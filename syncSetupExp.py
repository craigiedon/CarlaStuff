from dataclasses import dataclass
import queue
from typing import Tuple, List, Any, Optional, Callable, Mapping
import pickle

import carla
from carla import Rotation, Location
import numpy as np
import pygame
import torch
import torch.nn.functional as F
from carla import ColorConverter, Vehicle, Transform, Vector3D
from scipy.spatial.transform import Rotation as R

from carlaSetupUtils import set_weather, set_sync, create_cam
from customAgent import CustomAgent
from pems import load_model_det, PEMClass_Deterministic, PEMReg_Aleatoric
from render_utils import world_to_cam_viewport, depth_array_to_distances, get_image_as_array, draw_image, \
    world_to_cam_trans


@dataclass
class KITTI_Model_In:
    class_code: int
    truncation: float
    occ_code: int
    observation_angle: float
    dim_wlh: Tuple[float, float, float]
    loc_kitti_cf: Tuple[float, float, float]
    rot_y: float

    def as_tensor(self):
        return torch.tensor([self.class_code, self.truncation, self.occ_code, self.observation_angle, *self.dim_wlh,
                             *self.loc_kitti_cf, self.rot_y])


@dataclass
class Detector_Outputs:
    true_centre: Tuple[int, int]
    predicted_centre: Optional[Tuple[int, int]]
    true_det: bool
    model_det: bool


@dataclass
class Sim_Snapshot:
    time_step: int
    model_ins: KITTI_Model_In
    outs: Detector_Outputs


def cam_bb(v: Vehicle, cam_trans: Transform, cam_attrs: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    v_trans = v.get_transform()
    v_bb = v.bounding_box
    v_bb_verts = v_bb.get_world_vertices(v_trans)

    adv_vp_corners = np.array([world_to_cam_viewport(cam_trans, cam_attrs, c) for c in v_bb_verts])
    x_min, y_min = np.min(adv_vp_corners, axis=0)
    x_max, y_max = np.max(adv_vp_corners, axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


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


def amount_occluded_simple(cam_trans: Transform, cam_attrs: Mapping[str, Any], target_v: carla.Vehicle,
                           possible_occluders: List[carla.Vehicle]) -> float:
    tv_tw = target_v.get_transform()

    tv_tc = world_to_cam_trans(cam_trans, tv_tw)
    tv_xmin, tv_ymin, tv_xmax, tv_ymax = cam_bb(target_v, cam_trans, cam_attrs)
    tv_area = (tv_xmax - tv_xmin) * (tv_ymax - tv_ymin)

    oc_bbs = []
    for poc in possible_occluders:
        poc_tc = world_to_cam_trans(cam_trans, poc.get_transform())
        # If the possible occluder is closer to the camera than the target vehicle
        if poc_tc.location.x < tv_tc.location.x:
            oc_bbs.append(cam_bb(poc, cam_trans, cam_attrs))

    overlap_props = [0.0]
    for (oc_xmin, oc_ymin, oc_xmax, oc_ymax) in oc_bbs:
        overlap_xmin = max(oc_xmin, tv_xmin)
        overlap_xmax = min(oc_xmax, tv_xmax)
        overlap_ymin = max(oc_ymin, tv_ymin)
        overlap_ymax = min(oc_ymax, tv_ymax)

        overlap_w = max(overlap_xmax - overlap_xmin, 0)
        overlap_h = max(overlap_ymax - overlap_ymin, 0)
        overlap_area = overlap_w * overlap_h
        overlap_props.append(overlap_area / tv_area)

    return np.max(overlap_props)


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


def dummy_detector(salient_vars: torch.tensor, adv_vehicle: Vehicle, cam: carla.Sensor, dist_array,
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
    cc_pert = cc_orig + np.round(np.random.normal(0, 2.0, 2))

    if 0 <= cc_pert[0] < cam_width and 0 <= cc_pert[1] < cam_height:
        distance = dist_array[int(cc_pert[1]), int(cc_pert[0])]
        return True, cc_pert, distance
    else:
        return False, None, None


def model_detector(salient_vars: torch.tensor, adv_vehicle: Vehicle, cam: carla.Sensor, dist_array, det_model,
                   reg_model) -> Tuple[
    bool, Optional[np.ndarray], Optional[float]]:
    r = np.random.sample()

    logits_dr = det_model(salient_vars.unsqueeze(0))
    detection_rate = torch.sigmoid(logits_dr).item()

    print(detection_rate)
    assert 0 <= detection_rate <= 1

    if r > detection_rate:
        return False, None, None

    cam_width = float(cam.attributes["image_size_x"])
    cam_height = float(cam.attributes["image_size_y"])

    # cc_orig = cam_frame_to_viewport(cam.attributes, salient_vars[[10, 11, 12]])
    cc_orig = world_to_cam_viewport(cam.get_transform(), cam.attributes,
                                    adv_vehicle.get_location() + Location(0, 0, adv_vehicle.bounding_box.extent.z))

    m_noise = reg_model(salient_vars.unsqueeze(0))
    mn_mu, mn_log_std = m_noise[0][0].detach().numpy(), m_noise[1][0].detach().numpy()
    cc_pert = cc_orig + np.random.normal(mn_mu, np.exp(mn_log_std), 2)

    if 0 <= cc_pert[0] < cam_width and 0 <= cc_pert[1] < cam_height:
        distance = dist_array[int(cc_pert[1]), int(cc_pert[0])]
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


# class Sim_Snapshot:
#     def __init__(self,
#                  time_step: int,
#                  model_det: bool,
#                  true_det: bool,
#                  true_centre: Optional[np.ndarray],
#                  model_centre: Optional[np.ndarray],
#                  cam_trans: carla.Transform,
#                  ego_v: carla.Vehicle,
#                  adv_v: carla.Vehicle):
#
#         self.time_step = time_step
#         self.model_det: model_det
#         self.true_det = true_det
#         if true_centre is not None:
#             self.true_centre: Tuple[float, float] = tuple(true_centre)
#         else:
#             self.true_centre = None
#
#         if model_centre is not None:
#             self.model_centre: Tuple[float, float] = tuple(model_centre)
#         self.cam_loc = to_loc_tuple(cam_trans)
#         self.cam_rot = to_rot_tuple(cam_trans)
#         self.ego_loc = to_loc_tuple(ego_v.get_transform())
#         self.ego_rot = to_rot_tuple(ego_v.get_transform())
#         self.adv_loc = to_loc_tuple(adv_v.get_transform())
#         self.adv_rot = to_rot_tuple(adv_v.get_transform())


# class SnapshotEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, Sim_Snapshot):
#             return obj.__dict__
#         return json.JSONEncoder.default(self, obj)


def to_loc_tuple(t: carla.Transform) -> Tuple[float, float, float]:
    return t.location.x, t.location.y, t.location.z


def to_rot_tuple(t: carla.Transform) -> Tuple[float, float, float]:
    return t.rotation.pitch, t.rotation.yaw, t.rotation.roll


def rollout_nll(rollout_snap: List[Sim_Snapshot], det_model, reg_model, n_func: Callable) -> float:
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


def run():
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        # Load desired map
        client.load_world("Town01")
        set_sync(world, client, 0.05)
        set_weather(world, 0, 0, 0, 0, 0, 75)

        bpl = world.get_blueprint_library()

        # Spawn the ego vehicle
        ego_bp = bpl.find('vehicle.lincoln.mkz_2017')
        ego_bp.set_attribute('role_name', 'ego')
        ego_vehicle = world.spawn_actor(ego_bp, carla.Transform(Location(207, 133, 0.5), Rotation(0, 0, 0)))
        world.tick()
        actor_list.append(ego_vehicle)

        # Load Perception model
        pem_class = load_model_det(PEMClass_Deterministic(14, 1, use_cuda=False),
                                   "models/det_baseline_full/pem_class_train_full")
        pem_reg = load_model_det(PEMReg_Aleatoric(14, 2, use_cuda=False), "models/al_reg_full/pem_reg_al_full")
        norm_stats = torch.load("models/norm_stats_mu.pt"), torch.load("models/norm_stats_std.pt")
        n_func = lambda s_inputs, norm_dims: norm_salient_input(s_inputs, norm_stats[0], norm_stats[1], norm_dims)

        # Create Cameras
        cam_w, cam_h = 1242, 375
        ego_cam, rgb_queue = create_cam(world, ego_vehicle, (cam_w, cam_h), 82, Location(2, 0, 1.76), Rotation())
        depth_cam, depth_queue = create_cam(world, ego_vehicle, (cam_w, cam_h), 82, Location(2, 0, 1.76), Rotation(),
                                            'depth')

        # Spawn other vehicle
        other_bp = bpl.find('vehicle.nissan.patrol')
        ego_pos = ego_vehicle.get_location()
        ego_forward = ego_vehicle.get_transform().get_forward_vector()
        ego_right = ego_vehicle.get_transform().get_right_vector()

        other_vehicle = world.spawn_actor(other_bp, carla.Transform(
            ego_vehicle.get_location() + ego_forward * 20, ego_vehicle.get_transform().rotation))
        world.tick()
        actor_list.append(other_vehicle)
        print(f'created {other_vehicle.type_id}')

        # Set ego vehicle behaviour
        # ego_vehicle.set_autopilot(True)
        # agent = BasicAgent(ego_vehicle)
        agent = CustomAgent(ego_vehicle)
        other_vehicle.set_autopilot(True)

        spectator = world.get_spectator()

        pygame.init()
        # py_display = pygame.display.set_mode((cam_w, cam_h * 2), pygame.HWSURFACE | pygame.DOUBLEBUF)
        py_display = pygame.display.set_mode((cam_w, cam_h), pygame.HWSURFACE | pygame.DOUBLEBUF)

        lights_list = world.get_actors().filter("*traffic_light*")
        adv_v = world.get_actors().filter("vehicle.nissan.patrol")[0]

        for l in lights_list:
            l.set_red_time(100)

        rollout_log = []

        for i in range(350):
            w_frame = world.tick()

            # Follow ego on server window
            spectator.set_transform(
                carla.Transform(ego_vehicle.get_transform().location + Location(z=30), Rotation(pitch=-90)))

            # Render sensor output
            data_timeout = 2.0
            current_rgb = retrieve_data(rgb_queue, w_frame, data_timeout)
            current_depth = retrieve_data(depth_queue, w_frame, data_timeout)

            # print(current_rgb.frame, current_depth.frame, w_frame)

            distance_array = depth_array_to_distances(get_image_as_array(current_depth))
            current_depth.convert(ColorConverter.LogarithmicDepth)
            depth_im_array = get_image_as_array(current_depth)

            draw_image(py_display, get_image_as_array(current_rgb))
            # draw_image(py_display, depth_im_array, offset=(0, cam_h))

            d_in = to_data_in(ego_cam.get_transform(), ego_cam.attributes, adv_v)
            salient_vars = to_salient_var(d_in, n_func)

            tru_adv_vp = world_to_cam_viewport(ego_cam.get_transform(), ego_cam.attributes,
                                               adv_v.get_location() + Location(0, 0,
                                                                               adv_v.bounding_box.extent.z)).astype(int)

            # detection, cam_centroid, obstacle_depth = dummy_detector(salient_vars, adv_v, ego_cam, distance_array, 0.9)
            # dummy_det, dummy_centroid, dummy_depth = dummy_detector(salient_vars, adv_v, ego_cam, distance_array, 0.9)
            # m_detection, m_centroid, m_depth = model_detector(salient_vars, adv_v, ego_cam, distance_array, pem_class,
            #                                                   pem_reg)
            m_detection, m_centroid, m_depth = dummy_detector(salient_vars, adv_v, ego_cam, distance_array, 1.0)

            d_outs = Detector_Outputs(tuple(tru_adv_vp),
                                      tuple(m_centroid) if m_centroid is not None else None,
                                      True,
                                      m_detection)

            pygame.draw.circle(py_display, (0, 255, 0), (tru_adv_vp[0], tru_adv_vp[1]), 5.0)

            rollout_log.append(Sim_Snapshot(w_frame, d_in, d_outs))

            if m_detection:
                # print("BB-dist: \t", np.min([av.distance(ev) for av in adv_bb_verts for ev in ego_bb_verts]))
                pygame.draw.circle(py_display, (255, 0, 0), (m_centroid[0], m_centroid[1]), 5.0)
                # pygame.draw.rect(py_display, (255, 0, 0), pygame.Rect(cam_centroid[0] - 5, cam_h + cam_centroid[1] - 5, 10, 10), 2)

            pygame.display.flip()
            ego_vehicle.apply_control(agent.run_step(m_centroid, m_depth))

        with open("data_outs/rollout_log.pickle", 'wb') as f:
            pickle.dump(rollout_log, f)

        with open("data_outs/rollout_log.pickle", 'rb') as f:
            loaded_roll = pickle.load(f)

        print(loaded_roll)

        nll = rollout_nll(rollout_log, pem_class, pem_reg, n_func)

    finally:
        ego_cam.destroy()
        depth_cam.destroy()

        print("Actors to destroy: ", actor_list)
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        pygame.quit()

        print("Done")


if __name__ == "__main__":
    run()


