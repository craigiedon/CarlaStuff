import dataclasses
import queue
from dataclasses import dataclass
from typing import List, Tuple, Mapping, Any, Callable, Optional

import carla
import numpy as np
import torch
from carla import World, Client, Actor, Transform, Location, Rotation, Vehicle, Vector3D
from scipy.spatial.transform import Rotation as R
from torch import nn
from torch.nn import functional as F
import json
from json import JSONEncoder

from render_utils import world_to_cam_trans, world_to_cam_viewport, cam_bb, amount_occluded_simple, \
    viewport_to_vehicle_depth


@dataclass
class Detector_Outputs:
    true_centre: List[int]
    true_distance: Optional[float]

    predicted_centre: Optional[List[float]]
    predicted_distance: Optional[float]

    true_det: bool
    model_det: bool


def set_weather(w: World, cloud: float, prec: float, prec_dep: float, wind: float, sun_az: float, sun_alt: float):
    weather = w.get_weather()
    weather.cloudiness = cloud
    weather.precipitation = prec
    weather.precipitation_deposits = prec_dep
    weather.wind_intensity = wind
    weather.sun_azimuth_angle = sun_az
    weather.sun_altitude_angle = sun_alt
    w.set_weather(weather)


def set_sync(w: World, client: carla.Client, delta: float):
    # Set synchronous mode settings
    new_settings = w.get_settings()
    new_settings.synchronous_mode = True
    new_settings.fixed_delta_seconds = delta
    w.apply_settings(new_settings)

    client.reload_world(False)

    # Set up traffic manager
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)


def set_rendering(w: World, client: carla.Client, render: bool):
    new_settings = w.get_settings()
    new_settings.no_rendering_mode = not render
    w.apply_settings(new_settings)
    client.reload_world(False)


def delete_actors(client: Client, actor_list: List[Actor]):
    print("Actors to destroy: ", actor_list)
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])


def setup_actors(world: World, blueprints: List[carla.ActorBlueprint], transforms: List[Transform]):
    assert len(blueprints) == len(transforms)
    actor_list = [world.spawn_actor(bp, trans) for bp, trans in zip(blueprints, transforms)]
    world.tick()
    return actor_list


def create_cam(world: carla.World, vehicle: carla.Vehicle, cam_dims: Tuple[int, int], fov: int,
               cam_location: Location, cam_rotation: Rotation, cam_type: str = 'rgb') -> Tuple[
    carla.Sensor, queue.Queue]:
    bpl = world.get_blueprint_library()
    camera_bp = bpl.find(f'sensor.camera.{cam_type}')
    camera_bp.set_attribute("image_size_x", str(cam_dims[0]))
    camera_bp.set_attribute("image_size_y", str(cam_dims[1]))
    camera_bp.set_attribute("fov", str(fov))
    cam_transform = carla.Transform(cam_location, cam_rotation)

    cam = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle,
                            attachment_type=carla.AttachmentType.Rigid)
    img_queue = queue.Queue()

    if not world.get_settings().no_rendering_mode:
        cam.listen(img_queue.put)
    # cam.stop()

    return cam, img_queue


@dataclass
class KITTI_Model_In:
    class_code: int
    truncation: float
    occ_code: int
    observation_angle: float
    dim_wlh: List[float]
    loc_kitti_cf: List[float]
    rot_y: float

    def as_tensor(self):
        return torch.tensor([self.class_code, self.truncation, self.occ_code, self.observation_angle, *self.dim_wlh,
                             *self.loc_kitti_cf, self.rot_y])


def to_data_in(cam_trans: Transform, cam_attributes: Mapping[str, Any], adv_vehicle: Vehicle) -> KITTI_Model_In:
    class_code = 0

    adv_x_min, adv_y_min, adv_x_max, adv_y_max = cam_bb(adv_vehicle, cam_trans, cam_attributes)
    full_area = (adv_x_max - adv_x_min) * (adv_y_max - adv_y_min)

    cam_w = int(cam_attributes["image_size_x"])
    cam_h = int(cam_attributes["image_size_y"])

    clamp_w = min(adv_x_max, cam_w) - max(0, adv_x_min)
    clamp_h = min(adv_y_max, cam_h) - max(0, adv_y_min)
    clamped_area = clamp_w * clamp_h

    truncation = 1.0 - clamped_area / full_area

    occlusion_prop = amount_occluded_simple(cam_trans, cam_attributes, adv_vehicle, [])
    if occlusion_prop < 0.1:
        occ_code = 0
    elif 0.1 <= occlusion_prop < 0.5:
        occ_code = 1
    else:
        occ_code = 2

    adv_trans_c = world_to_cam_trans(cam_trans, adv_vehicle.get_transform())

    dim_wlh = adv_vehicle.bounding_box.extent * 2.0
    adv_loc_c = adv_trans_c.location

    # KITTI dataset measures observation angle from side of the car rather than front
    cam_disp_unit = adv_loc_c.make_unit_vector()
    observation_angle = ccw_angle_to(adv_trans_c.get_forward_vector(), cam_disp_unit) - np.pi / 2.0

    # In CARLA, x-axis points forward, wheras in KITTI it points to the side
    adv_rot_y = np.deg2rad(adv_trans_c.rotation.yaw) - np.pi / 2.0

    # Initial Input Format:
    #   0: <Class Num>
    #   1: <Truncation>
    #   2: <Occlusion>
    #   3: <alpha>
    #   4-6: <dim_w> <dim_l> <dim_h>
    #   7-9: <loc_x> <loc_y> <loc_z>
    #   10: <rot_y>

    # KITTI Coords: x= right, y = down, z = forward
    # CARLA Coords: x= forward, y = right, z = up
    kt_x = adv_loc_c.y
    kt_y = -adv_loc_c.z
    kt_z = adv_loc_c.x

    # Note: The wlh thing might still be wrong...

    return KITTI_Model_In(class_code, truncation, occ_code, observation_angle, (dim_wlh.x, dim_wlh.y, dim_wlh.z),
                          (kt_x, kt_y, kt_z), adv_rot_y)


def to_salient_var(init_in: KITTI_Model_In, normalizing_func: Callable = None) -> torch.tensor:
    initial_in_tensor = init_in.as_tensor()

    assert len(initial_in_tensor == 11)

    if normalizing_func is not None:
        norm_dims = [1, 3, 4, 5, 6, 7, 8, 9, 10]
        normed_ins = normalizing_func(initial_in_tensor, norm_dims)
    else:
        normed_ins = initial_in_tensor

    assert len(normed_ins == 11)

    ## Final Indexing:
    # 0-6 Vehicle Cat One-hot
    # 7-9: Occlusion One-hot
    # 10,11,12: x,y,z cam loc
    # 13: Rot y

    class_num = F.one_hot(torch.tensor(init_in.class_code), 7)
    occlusion = F.one_hot(torch.tensor(init_in.occ_code), 3)

    salient_vars = torch.tensor([
        *class_num,
        *occlusion,
        *normed_ins[7:10],
        normed_ins[10]
    ])

    assert len(salient_vars) == 14

    return salient_vars.float()


def dummy_detector(salient_vars: torch.tensor, adv_vehicle: Vehicle, cam: carla.Sensor, world: World,
                   detection_rate: float) -> Tuple[
    bool, Optional[np.ndarray], Optional[float]]:
    r = np.random.sample()

    if r > detection_rate:
        return False, None, None

    assert 0 <= detection_rate <= 1
    cam_width = float(cam.attributes["image_size_x"])
    cam_height = float(cam.attributes["image_size_y"])

    # cc_orig = cam_frame_to_viewport(cam.attributes, salient_vars[[10, 11, 12]])
    cc_orig = world_to_cam_viewport(cam.get_transform(), cam.attributes,
                                    adv_vehicle.get_location() + Location(0, 0, adv_vehicle.bounding_box.extent.z))

    noise_scale = 2.0
    cc_pert = cc_orig + np.round(np.random.normal(0, noise_scale, 2))

    distance = viewport_to_vehicle_depth(world, cam, cc_pert)
    if distance is not None:
        return True, cc_pert, distance
    else:
        return False, None, None


def model_detector(salient_vars: torch.tensor, adv_vehicle: Vehicle, cam: carla.Sensor, world: World,
                   det_model: nn.Module,
                   reg_model: nn.Module) -> Tuple[
    bool, Optional[np.ndarray], Optional[float]]:
    r = np.random.sample()

    logits_dr = det_model(salient_vars.unsqueeze(0))
    detection_rate = torch.sigmoid(logits_dr).item()

    # print(detection_rate)
    assert 0 <= detection_rate <= 1

    if r > detection_rate:
        return False, None, None

    cc_orig = world_to_cam_viewport(cam.get_transform(), cam.attributes,
                                    adv_vehicle.get_location() + Location(0, 0, adv_vehicle.bounding_box.extent.z))

    if reg_model is not None:
        m_noise = reg_model(salient_vars.unsqueeze(0))
        mn_mu, mn_log_std = m_noise[0][0].detach().numpy(), m_noise[1][0].detach().numpy()
        cc_pert = cc_orig + np.random.normal(mn_mu, np.exp(mn_log_std), 2)
    else:
        cc_pert = cc_orig

    distance = viewport_to_vehicle_depth(world, cam, cc_pert)

    if distance is not None:
        return True, cc_pert, distance
    else:
        return False, None, None


def retrieve_data(data_queue: queue.Queue, world_frame: int, timeout: float):
    while True:
        data = data_queue.get(timeout=timeout)
        if data.frame == world_frame:
            return data


def norm_salient_input(s_inputs, in_mu, in_std, norm_dims):
    normed_inputs = torch.detach(s_inputs)
    normed_inputs[norm_dims] = (normed_inputs[norm_dims] - in_mu[norm_dims]) / in_std[norm_dims]
    return normed_inputs


def to_loc_tuple(t: carla.Transform) -> Tuple[float, float, float]:
    return t.location.x, t.location.y, t.location.z


def to_rot_tuple(t: carla.Transform) -> Tuple[float, float, float]:
    return t.rotation.pitch, t.rotation.yaw, t.rotation.roll


@dataclass
class SimSnapshot:
    time_step: int
    model_ins: KITTI_Model_In
    outs: Detector_Outputs
    ego_vel: float
    ego_acc: float
    adv_vel: float
    adv_acc: float


class SnapshotEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SimSnapshot):
            return dataclasses.asdict(obj)
        return JSONEncoder.default(self, obj)


def rollout_nll(rollout_snap: List[SimSnapshot], det_model, reg_model, n_func: Callable) -> float:
    nlls = []
    for ss in rollout_snap:
        print(ss.time_step)
        print(ss.model_ins)
        print(ss.outs)
        s_vars = to_salient_var(ss.model_ins, n_func)
        if ss.outs.model_det:
            nll_det = -torch.log(torch.sigmoid(det_model(s_vars.unsqueeze(0))))
            reg_mu, reg_log_sig = reg_model(s_vars.unsqueeze(0))
            centroid_error = (
                    torch.tensor(ss.outs.predicted_centre) - torch.tensor(ss.outs.true_centre)).float().unsqueeze(0)
            nll_reg = F.gaussian_nll_loss(centroid_error, reg_mu, torch.exp(2.0 * reg_log_sig))
        else:
            nll_det = -torch.log(1.0 - torch.sigmoid(det_model(s_vars.unsqueeze(0))))
            nll_reg = 0.0

        nlls.append(nll_det + nll_reg)

    full_nll = torch.sum(torch.vstack(nlls))
    print(full_nll)

    return full_nll.item()


def ccw_angle_to(v1, v2):
    dot = v1.dot(v2)
    det = Vector3D(0, 0, -1).dot(v1.cross(v2))
    return np.arctan2(det, dot)


def rot_2d(v, rads):
    rot_m = np.array([[np.cos(rads), -np.sin(rads)], [np.sin(rads), np.cos(rads)]])
    return rot_m @ v


def rot_rh_y(v, rads):
    rot = R.from_rotvec(rads * np.array([0, 1, 0]))
    return rot.apply(v)
