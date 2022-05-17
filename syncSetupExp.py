import queue
import time
import random
from typing import Tuple

import carla
import numpy as np
import pygame
import torch
import torch.nn.functional as F
from carla import ColorConverter, World, Vehicle, Transform, Vector3D
from navigation.basic_agent import BasicAgent
from navigation.behavior_agent import BehaviorAgent

from customAgent import CustomAgent
from render_utils import world_to_cam_viewport, depth_array_to_distances, get_image_as_array, draw_image, \
    viewport_to_world, world_to_cam_frame


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
    new_settings.fixed_delta_seconds = 0.05
    w.apply_settings(new_settings)

    client.reload_world(False)

    # Set up traffic manager
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)


def cam_bb(v: Vehicle, cam: carla.Sensor) -> Tuple[int, int, int, int]:
    v_trans = v.get_transform()
    v_bb = v.bounding_box
    v_bb_verts = v_bb.get_world_vertices(v_trans)

    adv_vp_corners = np.array([world_to_cam_viewport(cam.get_transform(), cam.attributes, c) for c in v_bb_verts])
    x_min, y_min = np.min(adv_vp_corners, axis=0)
    x_max, y_max = np.max(adv_vp_corners, axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def cw_angle_to(v1, v2):
    det = np.linalg.det([v2, v1])
    dot = v2 @ v1
    return np.arctan2(det, dot)


def rot_2d(v, rads):
    rot_m = np.array([[np.cos(rads), -np.sin(rads)], [np.sin(rads), np.cos(rads)]])
    return rot_m @ v


def convert_to_salient(cam: carla.Sensor, adv_vehicle: Vehicle):
    # 0: <Class Num>
    #   1: <Truncation>
    #   2: <Occlusion>
    #   3: <alpha>
    #   4-6: <dim_w> <dim_l> <dim_h>
    #   7-9: <loc_x> <loc_y> <loc_z>
    #   10: <rot_y>

    class_num = F.one_hot(torch.tensor([0]), 7)

    adv_x_min, adv_y_min, adv_x_max, adv_y_max = cam_bb(adv_vehicle, cam)
    full_area = (adv_x_max - adv_x_min) * (adv_y_max - adv_y_min)

    cam_w = int(cam.attributes["image_size_x"])
    cam_h = int(cam.attributes["image_size_y"])

    clamp_w = min(adv_x_max, cam_w) - max(0, adv_x_min)
    clamp_h = min(adv_y_max, cam_h) - max(0, adv_y_min)
    clamped_area = clamp_w * clamp_h

    truncation = 1.0 - clamped_area / full_area

    occlusion = F.one_hot(torch.tensor([0]), 3)
    c_loc = world_to_cam_frame(cam.get_transform(), adv_vehicle.get_transform().location)
    adv_fv = adv_vehicle.get_transform().get_forward_vector()

    adv_fv = rot_2d(np.array([1.0, 0.0]), 1.58)
    print(adv_fv)

    cam_disp = np.array([34.38, -3.18])
    cam_disp = cam_disp / np.linalg.norm(cam_disp)
    print(cam_disp)

    ang = cw_angle_to(adv_fv, cam_disp)
    print(ang)

    # So put in some simple examples here:
    # fw_cam = np.array([1, 0])
    # ex_fw = np.array([1, 0])
    # ex_bw = rot(ex_fw, np.pi)
    # ex_left = rot(ex_fw, np.pi / 2.0)
    # ex_right = rot(ex_fw, -np.pi / 2.0)
    # ex_anti = rot(ex_fw, np.pi / 4.0)
    # ex_c = rot(ex_fw, -np.pi / 4.0)
    #
    # print("Forward", np.rad2deg(cw_angle_to(ex_fw, fw_cam)))
    # print("Backward", np.rad2deg(cw_angle_to(ex_bw, fw_cam)))
    # print("Left", np.rad2deg(cw_angle_to(ex_left, fw_cam)))
    # print("Right", np.rad2deg(cw_angle_to(ex_right, fw_cam)))
    # print("Anti", np.rad2deg(cw_angle_to(ex_anti, fw_cam)))
    # print("Clock", np.rad2deg(cw_angle_to(ex_c, fw_cam)))

    # Okay! Now thats cleared up...is the rot_y calculation even correct?

    alpha = 0.0

    dim_wlh = adv_vehicle.bounding_box.extent * 2.0
    rot = np.deg2rad(adv_vehicle.get_transform().rotation[1])

    return torch.tensor([
        *class_num,
        truncation,
        *occlusion,
        alpha,
        *dim_wlh,
        *c_loc,
        rot
    ])


# def dummy_detector(salient_vars: torch.tensor):
#     return detected, cam_loc, distance


def run():
    actor_list = []
    try:
        # Py Display Setup
        cam_width, cam_height = 1242, 375

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        # Load desired map
        client.load_world("Town01")

        set_sync(world, client, 0.05)

        # Set the weather
        set_weather(world, 0, 0, 0, 0, 0, 75)

        bpl = world.get_blueprint_library()

        # Spawn the ego vehicle
        ego_bp = bpl.find('vehicle.lincoln.mkz_2017')
        ego_bp.set_attribute('role_name', 'ego')
        ego_vehicle = world.spawn_actor(ego_bp, carla.Transform(carla.Location(187, 133, 0.5), carla.Rotation(0, 0, 0)))
        actor_list.append(ego_vehicle)
        print(f'created {ego_vehicle.type_id}')
        world.tick()

        # Create RGB Camera
        bpl = world.get_blueprint_library()
        camera_bp = bpl.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", str(cam_width))
        camera_bp.set_attribute("image_size_y", str(cam_height))
        camera_bp.set_attribute("fov", str(82))
        cam_location = carla.Location(2, 0, 1.76)
        cam_rotation = carla.Rotation(0, 0, 0)
        cam_transform = carla.Transform(cam_location, cam_rotation)
        ego_cam = world.spawn_actor(camera_bp, cam_transform, attach_to=ego_vehicle,
                                    attachment_type=carla.AttachmentType.Rigid)
        rgb_queue = queue.Queue()
        ego_cam.listen(rgb_queue.put)

        # Create Depth Camera
        depth_bp = bpl.find('sensor.camera.depth')
        depth_bp.set_attribute("image_size_x", str(cam_width))
        depth_bp.set_attribute("image_size_y", str(cam_height))
        depth_bp.set_attribute("fov", str(82))
        depth_loc = carla.Location(2, 0, 1.76)
        depth_rot = carla.Rotation(0, 0, 0)
        depth_transform = carla.Transform(depth_loc, depth_rot)
        depth_cam = world.spawn_actor(depth_bp, depth_transform, attach_to=ego_vehicle,
                                      attachment_type=carla.AttachmentType.Rigid)

        depth_queue = queue.Queue()
        # depth_cam.listen(lambda img: depth_queue.put(img.convert(ColorConverter.LogarithmicDepth)))
        depth_cam.listen(depth_queue.put)

        # Spawn other vehicle
        other_bp = bpl.find('vehicle.nissan.patrol')
        ego_pos = ego_vehicle.get_location()
        ego_forward = ego_vehicle.get_transform().get_forward_vector()

        other_vehicle = world.spawn_actor(other_bp, carla.Transform(ego_vehicle.get_location() + ego_forward * 25 + carla.Location(y = 1.0),
                                                                    ego_vehicle.get_transform().rotation))
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
        py_display = pygame.display.set_mode((cam_width, cam_height * 2), pygame.HWSURFACE | pygame.DOUBLEBUF)

        lights_list = world.get_actors().filter("*traffic_light*")
        adv_v = world.get_actors().filter("vehicle.nissan.patrol")[0]

        for l in lights_list:
            l.set_red_time(100)

        for i in range(500):
            world.tick()
            # print(i)

            # Follow ego on server window
            spectator.set_transform(
                carla.Transform(ego_vehicle.get_transform().location + carla.Location(z=30), carla.Rotation(pitch=-90)))

            # Render sensor output
            current_rgb = rgb_queue.get()
            current_depth = depth_queue.get()

            # adv_trans = adv_v.get_transform()
            # adv_bb = adv_v.bounding_box
            # adv_bb_verts = adv_bb.get_world_vertices(adv_trans)
            # ego_bb_verts = ego_vehicle.bounding_box.get_world_vertices(ego_vehicle.get_transform())

            # adv_vp_corners = np.array(
            #     [world_to_cam_viewport(ego_cam.get_transform(), ego_cam.attributes, c) for c in adv_bb_verts])
            # xmin, ymin = np.min(adv_vp_corners, axis=0)
            # xmax, ymax = np.max(adv_vp_corners, axis=0)

            distance_array = depth_array_to_distances(get_image_as_array(current_depth))
            current_depth.convert(ColorConverter.LogarithmicDepth)
            depth_im_array = get_image_as_array(current_depth)

            draw_image(py_display, get_image_as_array(current_rgb))
            draw_image(py_display, depth_im_array, offset=(0, cam_height))

            adv_centre = adv_v.get_transform().location + carla.Location(0.0, 0.0, adv_v.bounding_box.extent.z)

            cam_centroid = world_to_cam_viewport(ego_cam.get_transform(), ego_cam.attributes, adv_centre)
            obstacle_depth = None

            if 0 <= cam_centroid[0] < cam_width and 0 <= cam_centroid[1] < cam_height:
                obstacle_depth = distance_array[int(cam_centroid[1]), int(cam_centroid[0])]
                print("Obs Depth: ", obstacle_depth)
                # print("BB-dist: \t", np.min([av.distance(ev) for av in adv_bb_verts for ev in ego_bb_verts]))
                pygame.draw.rect(py_display, (255, 0, 0),
                                 pygame.Rect(cam_centroid[0] - 5, cam_centroid[1] - 5, 10, 10), 2)
                pygame.draw.rect(py_display, (255, 0, 0),
                                 pygame.Rect(cam_centroid[0] - 5, cam_height + cam_centroid[1] - 5, 10, 10), 2)
                pygame.display.flip()
            else:
                cam_centroid = None
                obstacle_depth = None
                # depth_im_array[int(cam_centroid[1]), int(cam_centroid[0]), :] = [0, 255, 0]
                # print("obs depth: \t", obstacle_depth)
                # print("Ego loc: ", ego_vehicle.get_location())
                # print("Adv loc true: ", adv_v.get_transform().location)
                # print("Location diffs: ", ego_vehicle.get_location().distance(adv_v.get_transform().location))
                # print("Centroid: ", centroid)
                # vp_coords = np.array([centroid[1], centroid[0]]).astype(float)
                # print("VP Coords: ", vp_coords)
                # print("Guess from sensors: ",
                #       viewport_to_world(ego_cam.get_transform(), ego_cam.attributes, obstacle_depth, vp_coords))

            # Note: Should make this so that it is bounded by being inside the camera

            convert_to_salient(ego_cam, adv_v)

            ego_vehicle.apply_control(agent.run_step(cam_centroid, obstacle_depth))


    finally:
        ego_cam.destroy()
        depth_cam.destroy()

        print("Actors to destroy: ", actor_list)
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        pygame.quit()

        print("Done")


if __name__ == "__main__":
    run()
