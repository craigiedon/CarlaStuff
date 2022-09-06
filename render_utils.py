import copy
from typing import Dict, Mapping, Any, Tuple, List, Optional

import carla
from carla import Vehicle, Transform, Vector3D, World
import numpy as np
import pygame
from scipy.spatial.transform import Rotation as R


def cam_frame_to_viewport(cam_attrs: Mapping[str, Any], loc_cf) -> np.ndarray:
    fov_rad = np.deg2rad(float(cam_attrs["fov"]))
    im_w = float(cam_attrs["image_size_x"])
    im_h = float(cam_attrs["image_size_y"])
    f_length = 0.5 * im_w / np.tan(fov_rad / 2.0)

    px = (im_w / 2.0) + (f_length * loc_cf[1] / loc_cf[0])
    py = (im_h / 2.0) - (f_length * loc_cf[2] / loc_cf[0])
    return np.array([px, py])


def viewport_to_cam_frame(cam_attrs: Mapping[str, Any], distance: float, point_vp: np.ndarray) -> carla.Location:
    fov_rad = np.deg2rad(float(cam_attrs["fov"]))
    im_w = float(cam_attrs["image_size_x"])
    im_h = float(cam_attrs["image_size_y"])
    f_length = 0.5 / np.tan(fov_rad / 2.0)

    vp_loc_euclid = np.array([f_length,
                              (point_vp[0] - 0.5 * im_w) / im_w,
                              (point_vp[1] - 0.5 * im_h) / -im_w])

    vp_loc_normed = vp_loc_euclid / np.linalg.norm(vp_loc_euclid)
    back_proj_cam = vp_loc_normed * distance
    return carla.Location(*back_proj_cam)


def viewport_to_ray(cam_attrs: Mapping[str, Any], point_vp: np.ndarray) -> np.ndarray:
    # Extract camera computation attributes
    fov_rad = np.deg2rad(float(cam_attrs["fov"]))
    im_w = float(cam_attrs["image_size_x"])
    im_h = float(cam_attrs["image_size_y"])
    f_length = 0.5 / np.tan(fov_rad / 2.0)
    vp_dir_euclid = np.array([f_length,
                              (point_vp[0] - 0.5 * im_w) / im_w,
                              (point_vp[1] - 0.5 * im_h) / -im_w])
    vp_dir_normed = vp_dir_euclid / np.linalg.norm(vp_dir_euclid)
    return vp_dir_normed


def viewport_to_vehicle_depth(world: World, cam: carla.Sensor, vp: np.ndarray) -> Optional[float]:
    ray_tru = viewport_to_ray(cam.attributes, vp)
    projection_result = world.project_point(cam.get_location(), Vector3D(*ray_tru), 100.0)

    if projection_result is not None and projection_result.label == carla.CityObjectLabel.Vehicles:
        return cam.get_location().distance(projection_result.location)

    return None


def viewport_to_world(cam_trans: carla.Transform, cam_attrs, distance: float, point_vp: np.ndarray) -> carla.Location:
    point_cf = viewport_to_cam_frame(cam_attrs, distance, point_vp)
    point_world = cam_trans.transform(point_cf)
    return point_world


def world_to_cam_loc(cam_trans: carla.Transform, world_loc: carla.Location) -> np.ndarray:
    inv_cmat = np.array(cam_trans.get_inverse_matrix()).reshape((4, 4))
    h_adv_loc = np.append(np.array([world_loc.x, world_loc.y, world_loc.z]), 1.0)
    adv_loc_cf = inv_cmat @ h_adv_loc
    return adv_loc_cf[:3]


def world_to_cam_trans(cam_trans: carla.Transform, world_trans: carla.Transform) -> carla.Transform:
    inv_cmat = np.array(cam_trans.get_inverse_matrix()).reshape((4, 4))
    world_mat = np.array(world_trans.get_matrix()).reshape((4, 4))
    world_cf = inv_cmat @ world_mat
    world_cf_loc = world_cf[0:3, 3]
    world_cf_rot = R.from_matrix(world_cf[0:3, 0:3]).as_euler('zyx', degrees=True)
    return carla.Transform(carla.Location(*world_cf_loc),
                           carla.Rotation(yaw=world_cf_rot[0], pitch=world_cf_rot[1], roll=world_cf_rot[2]))


def world_to_cam_viewport(cam_trans, cam_attrs, world_loc) -> np.ndarray:
    loc_cf = world_to_cam_loc(cam_trans, world_loc)
    return cam_frame_to_viewport(cam_attrs, loc_cf)


def draw_image(surface, array, offset=(0, 0), blend=False):
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, offset)


def get_image_as_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    array = array.astype(np.float32)
    # make the array writeable doing a deep copy
    array2 = copy.deepcopy(array)
    return array2


def depth_array_to_distances(depth_array) -> np.ndarray:
    R = depth_array[:, :, 0]
    G = depth_array[:, :, 1]
    B = depth_array[:, :, 2]
    normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
    return normalized * 1000.0


def cam_bb(v: Vehicle, cam_trans: Transform, cam_attrs: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    v_trans = v.get_transform()
    v_bb = v.bounding_box
    v_bb_verts = v_bb.get_world_vertices(v_trans)

    adv_vp_corners = np.array([world_to_cam_viewport(cam_trans, cam_attrs, c) for c in v_bb_verts])
    x_min, y_min = np.min(adv_vp_corners, axis=0)
    x_max, y_max = np.max(adv_vp_corners, axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


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
