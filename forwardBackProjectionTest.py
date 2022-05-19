import carla
import numpy as np

from render_utils import world_to_cam_viewport, world_to_cam_loc, cam_frame_to_viewport, viewport_to_world

cam_attrs = {
    "fov": 80,
    "image_size_x": 1080,
    "image_size_y": 720,
}

cam_loc = carla.Location(131.0, 133, 0.05)
cam_trans = carla.Transform(cam_loc, carla.Rotation(0, 0, 0))
obs_loc = carla.Location(155.0, 130, 0.01)

dist = cam_loc.distance(obs_loc)
print("Camera Location: ", cam_loc)
print("Obstacle Location (World): ", obs_loc)

obs_loc_cf = world_to_cam_loc(cam_trans, obs_loc)
obs_loc_cf_normed = obs_loc_cf / np.linalg.norm(obs_loc_cf)
print("Obstacle Location (Cam-Frame): ", obs_loc_cf)
print("Obs Loc (Cam Frame & normalised)", obs_loc_cf_normed)
print("Distance: ", dist)

vp_loc_pix = world_to_cam_viewport(cam_trans, cam_attrs, obs_loc)
back_proj_world = viewport_to_world(cam_trans, cam_attrs, dist, vp_loc_pix)

print("(Backprojected) Obstacle Location (World): ", back_proj_world)

