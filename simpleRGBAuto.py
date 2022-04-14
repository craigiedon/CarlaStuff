import queue
import time
from datetime import datetime

import carla
import numpy as np
import pygame
from carla import ColorConverter
from navigation.behavior_agent import BehaviorAgent

from render_utils import draw_image, get_image_as_array, depth_array_to_distances, \
    world_to_cam_viewport


def run_carla():
    cam_width = 1242
    cam_height = 375
    pygame.init()
    py_display = pygame.display.set_mode((cam_width, cam_height * 2), pygame.HWSURFACE | pygame.DOUBLEBUF)
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        world = client.get_world()

        ego_vehicle = None

        # Get the ego vehicle
        while ego_vehicle is None:
            print("Waiting for the ego vehicle...")
            time.sleep(1)
            possible_vehicles = world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == "hero":
                    print(f"Ego vehicle found with id: {vehicle.id} and type_id: {vehicle.type_id}")
                    ego_vehicle = vehicle
                    break

        # Set ego auto behaviour
        agent = BehaviorAgent(ego_vehicle, behavior="aggressive")
        # Set the agent destination
        spawn_points = world.get_map().get_spawn_points()
        destination = spawn_points[0].location
        agent.set_destination(destination)

        ts = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

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

        spectator = world.get_spectator()
        # spectator.set_transform(carla.Transform(ego_vehicle.get_transform().location + carla.Location(z=25),
        #                                         carla.Rotation(pitch=-90)))

        for i in range(2000):
            world_snapshot = world.wait_for_tick()
            spectator.set_transform(carla.Transform(ego_vehicle.get_transform().location + carla.Location(z=25),
                                                    carla.Rotation(pitch=-90)))

            if not world.get_actor(ego_vehicle.id).is_alive:
                return

            current_rgb = rgb_queue.get()
            current_depth = depth_queue.get()
            adv_v = world.get_actors().filter("vehicle.nissan.patrol")[0]
            adv_trans = adv_v.get_transform()
            adv_bb = adv_v.bounding_box
            adv_bb_verts = adv_bb.get_world_vertices(adv_trans)

            adv_vp_corners = np.array([world_to_cam_viewport(ego_cam.get_transform(), ego_cam.attributes, c) for c in adv_bb_verts])
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
                print("obs depth: ", obstacle_depth)
                print("Ego loc: ", ego_vehicle.get_location())
                print("Adv loc: ", adv_v.get_transform().location)
                print("Actual distance", ego_vehicle.get_location().distance(adv_v.get_transform().location))

            # Note: Should make this so that it is bounded by being inside the camera

            draw_image(py_display, depth_im_array, offset=(0, cam_height))

            pygame.draw.rect(py_display, (255, 0, 0),
                             pygame.Rect(xmin, ymin, xmax - xmin, ymax - ymin), 2)
            pygame.draw.rect(py_display, (255, 0, 0),
                             pygame.Rect(xmin, cam_height + ymin, xmax - xmin, ymax - ymin), 2)

            pygame.display.flip()
            print(i)

            control = agent.run_step()
            ego_vehicle.apply_control(control)

    except Exception as e:
        print(e)
    finally:
        ego_cam.destroy()
        depth_cam.destroy()
        pygame.quit()


if __name__ == '__main__':
    run_carla()
