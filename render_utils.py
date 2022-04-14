import copy

import carla
import numpy as np
import pygame


def cam_frame_to_viewport(cam_attrs, loc_cf) -> np.ndarray:
    fov_rad = np.deg2rad(float(cam_attrs["fov"]))
    im_w = float(cam_attrs["image_size_x"])
    im_h = float(cam_attrs["image_size_y"])
    f_length = 0.5 * im_w / np.tan(fov_rad / 2.0)

    px = (im_w / 2.0) + (f_length * loc_cf[1] / loc_cf[0])
    py = (im_h / 2.0) - (f_length * loc_cf[2] / loc_cf[0])
    return np.array([px, py])


def viewport_to_cam_frame(cam_attrs, distance: float, point_vp: np.ndarray) -> carla.Location:
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


def viewport_to_world(cam_trans: carla.Transform, cam_attrs, distance: float, point_vp: np.ndarray) -> carla.Location:
    point_cf = viewport_to_cam_frame(cam_attrs, distance, point_vp)
    point_world = cam_trans.transform(point_cf)
    return point_world


def world_to_cam_frame(cam_trans: carla.Transform, world_loc: carla.Location) -> np.ndarray:
    inv_cmat = np.array(cam_trans.get_inverse_matrix()).reshape((4, 4))
    h_adv_loc = np.append(np.array([world_loc.x, world_loc.y, world_loc.z]), 1.0)
    adv_loc_cf = inv_cmat @ h_adv_loc
    return adv_loc_cf[:3]


def world_to_cam_viewport(cam_trans, cam_attrs, world_loc) -> np.ndarray:
    loc_cf = world_to_cam_frame(cam_trans, world_loc)
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
