import queue
import time
import random

import carla
import numpy as np
import pygame
from carla import ColorConverter

from render_utils import world_to_cam_viewport, depth_array_to_distances, get_image_as_array, draw_image, \
    viewport_to_world

actor_list = []

try:
    # Py Display Setup
    cam_width = 1242
    cam_height = 375
    pygame.init()
    py_display = pygame.display.set_mode((cam_width, cam_height * 2), pygame.HWSURFACE | pygame.DOUBLEBUF)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Load desired map
    client.load_world("Town01")

    # Set synchronous mode settings
    new_settings = world.get_settings()
    new_settings.synchronous_mode = True
    new_settings.fixed_delta_seconds = 0.05
    world.apply_settings(new_settings)

    client.reload_world(False)

    # Set up traffic manager
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    # Set the weather
    weather = world.get_weather()
    weather.cloudiness = 0
    weather.precipitation = 0
    weather.precipitation_deposits = 0
    weather.wind_intensity = 0
    weather.sun_azimuth_angle = 0
    weather.sun_altitude_angle = 75
    world.set_weather(weather)

    bpl = world.get_blueprint_library()

    # Spawn the ego vehicle
    ego_bp = bpl.find('vehicle.lincoln.mkz_2017')
    ego_bp.set_attribute('role_name', 'ego')
    ego_vehicle = world.spawn_actor(ego_bp, carla.Transform(carla.Location(107, 133, 0.5), carla.Rotation(0, 0, 0)))
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

    other_vehicle = world.spawn_actor(other_bp, carla.Transform(ego_vehicle.get_location() + ego_forward * 25,
                                                                ego_vehicle.get_transform().rotation))
    actor_list.append(other_vehicle)
    print(f'created {other_vehicle.type_id}')

    # Set ego vehicle behaviour
    ego_vehicle.set_autopilot(True)
    other_vehicle.set_autopilot(True)

    spectator = world.get_spectator()
    for i in range(1000):
        world.tick()
        print(i)

        # Follow ego on server window
        spectator.set_transform(
            carla.Transform(ego_vehicle.get_transform().location + carla.Location(z=30), carla.Rotation(pitch=-90)))

        # Render sensor output
        current_rgb = rgb_queue.get()
        current_depth = depth_queue.get()

        adv_v = world.get_actors().filter("vehicle.nissan.patrol")[0]
        adv_trans = adv_v.get_transform()
        adv_bb = adv_v.bounding_box

        adv_bb_verts = adv_bb.get_world_vertices(adv_trans)
        ego_bb_verts = ego_vehicle.bounding_box.get_world_vertices(ego_vehicle.get_transform())

        adv_vp_corners = np.array(
            [world_to_cam_viewport(ego_cam.get_transform(), ego_cam.attributes, c) for c in adv_bb_verts])
        xmin, ymin = np.min(adv_vp_corners, axis=0)
        xmax, ymax = np.max(adv_vp_corners, axis=0)

        distance_array = depth_array_to_distances(get_image_as_array(current_depth))
        current_depth.convert(ColorConverter.LogarithmicDepth)
        depth_im_array = get_image_as_array(current_depth)

        draw_image(py_display, get_image_as_array(current_rgb))
        centroid = int((ymax + ymin) / 2), int((xmax + xmin) / 2)
        if 0 <= centroid[0] < cam_height and 0 <= centroid[1] < cam_width:
            obstacle_depth = distance_array[centroid]
            depth_im_array[centroid[0], centroid[1], :] = [0, 255, 0]
            print("obs depth: \t", obstacle_depth)
            # print("Ego loc: ", ego_vehicle.get_location())
            print("Adv loc true: ", adv_v.get_transform().location)
            # print("Location diffs: ", ego_vehicle.get_location().distance(adv_v.get_transform().location))
            print("Centroid: ", centroid)
            vp_coords = np.array([centroid[1], centroid[0]]).astype(float)
            print("VP Coords: ", vp_coords)
            print("Guess from sensors: ", viewport_to_world(ego_cam.get_transform(), ego_cam.attributes, obstacle_depth, vp_coords))

            print("BB-dist: \t", np.min([av.distance(ev) for av in adv_bb_verts for ev in ego_bb_verts]))

        # Note: Should make this so that it is bounded by being inside the camera

        draw_image(py_display, depth_im_array, offset=(0, cam_height))

        pygame.draw.rect(py_display, (255, 0, 0),
                         pygame.Rect(xmin, ymin, xmax - xmin, ymax - ymin), 2)
        pygame.draw.rect(py_display, (255, 0, 0),
                         pygame.Rect(xmin, cam_height + ymin, xmax - xmin, ymax - ymin), 2)

        pygame.display.flip()

finally:
    ego_cam.destroy()
    depth_cam.destroy()

    print("Actors to destroy: ", actor_list)
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    pygame.quit()

    print("Done")
