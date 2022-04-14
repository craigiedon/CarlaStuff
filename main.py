import copy
import queue
import random
import time
from datetime import datetime

import carla
import numpy as np
import scenic
import pygame


def draw_image(surface, array, blend=False):
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_image_as_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    # make the array writeable doing a deep copy
    array2 = copy.deepcopy(array)
    return array2


def run_carla():
    cam_width = 1242
    cam_height = 375
    pygame.init()
    py_display = pygame.display.set_mode((cam_width, cam_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.load_world("Town01")

        world = client.get_world()
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)

        # client.reload_world(False)  # Reload map keeping the world settings

        traffic_man = client.get_trafficmanager()
        traffic_man.set_synchronous_mode(True)
        traffic_man.set_random_device_seed(3)

        bpl = world.get_blueprint_library()

        # Spawn the ego vehicle
        ego_bp = bpl.find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'ego')
        ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
        ego_bp.set_attribute('color', ego_color)
        transform = world.get_map().get_spawn_points()[0]
        ego_vehicle = world.spawn_actor(ego_bp, transform)
        actor_list.append(ego_vehicle)
        print('created %s' % ego_vehicle.type_id)

        ego_vehicle.set_autopilot(True)
        ts = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

        # Create RGB Camera
        camera_bp = bpl.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", str(cam_width))
        camera_bp.set_attribute("image_size_y", str(cam_height))
        camera_bp.set_attribute("fov", str(82))
        cam_location = carla.Location(2, 0, 1.76)
        cam_rotation = carla.Rotation(0, 0, 0)
        cam_transform = carla.Transform(cam_location, cam_rotation)
        ego_cam = world.spawn_actor(camera_bp, cam_transform, attach_to=ego_vehicle,
                                    attachment_type=carla.AttachmentType.Rigid)
        actor_list.append(ego_cam)
        print('created %s' % ego_cam.type_id)

        rgb_queue = queue.Queue()
        # ego_cam.listen(lambda image: image.save_to_disk(f'data_outs/{ts}/rgb/{image.frame:.6f}.png'))
        ego_cam.listen(rgb_queue.put)

        # transform.location += carla.Location(x=40, y=-3.2)
        # transform.rotation.yaw = -180.0
        # for _ in range(0, 10):
        #     transform.location.x += 8.0
        #     bp = random.choice(bpl.filter('vehicle'))
        #
        #     npc = world.try_spawn_actor(bp, transform)
        #     if npc is not None:
        #         actor_list.append(npc)
        #         npc.set_autopilot(True)
        #         print(f'created {npc.type_id}')

        for i in range(1000):
            world.tick()
            current_im = rgb_queue.get()
            draw_image(py_display, get_image_as_array(current_im))
            pygame.display.flip()
            # current_im.save_to_disk(f'data_outs/{ts}/rgb/{current_im.frame:.6f}.png')
            print(i)
            spectator = world.get_spectator()
            spectator.set_transform(carla.Transform(ego_vehicle.get_transform().location + carla.Location(z=30),
                                                    carla.Rotation(pitch=-90)))

    except Exception as e:
        print(e)
    finally:
        print("destroying actors")
        ego_cam.destroy()
        print("Actors to destroy: ", actor_list)
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')
        pygame.quit()


if __name__ == '__main__':
    run_carla()
