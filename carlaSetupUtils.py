import queue
from typing import List, Tuple

import carla
from carla import World, Client, Actor, Transform, Location, Rotation


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
    cam.listen(img_queue.put)

    return cam, img_queue
