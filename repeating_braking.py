import time
from dataclasses import dataclass

import carla
import torch
from carla import Location, Rotation, Vehicle
from navigation.basic_agent import BasicAgent
from navigation.local_planner import LocalPlanner

from carlaSetupUtils import set_sync, set_weather, delete_actors, setup_actors, create_cam
from customAgent import CustomAgent
from pems import load_model_det, PEMClass_Deterministic, PEMReg_Aleatoric
from syncSetupExp import norm_salient_input


# TODO: Can we include info about wheel spin, velocity, throttle, acceleration etc?
@dataclass
class VehicleStat:
    location: Location
    dist_travelled: float


def update_vehicle_stats(actor: Vehicle, vs: VehicleStat) -> VehicleStat:
    new_loc = actor.get_location()
    new_dist = vs.dist_travelled + new_loc.distance(vs.location)
    return VehicleStat(new_loc, new_dist)


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

        # PHASE 1:

        # Load Perception model
        # pem_class = load_model_det(PEMClass_Deterministic(14, 1, use_cuda=False),
        #                            "models/det_baseline_full/pem_class_train_full")
        # pem_reg = load_model_det(PEMReg_Aleatoric(14, 2, use_cuda=False), "models/al_reg_full/pem_reg_al_full")
        # norm_stats = torch.load("models/norm_stats_mu.pt"), torch.load("models/norm_stats_std.pt")
        # n_func = lambda s_inputs, norm_dims: norm_salient_input(s_inputs, norm_stats[0], norm_stats[1], norm_dims)
        #
        # # Create Cameras
        # cam_w, cam_h = 1242, 375
        # ego_cam, rgb_queue = create_cam(world, ego_vehicle, (cam_w, cam_h), 82, Location(2, 0, 1.76), Rotation())
        # depth_cam, depth_queue = create_cam(world, ego_vehicle, (cam_w, cam_h), 82, Location(2, 0, 1.76), Rotation(),
        #                                     'depth')
        #
        # other_bp = bpl.find('vehicle.dodge.charger_2020')
        #
        # other_start_trans = carla.Transform(
        #     ego_start_trans.location + ego_start_trans.get_forward_vector() * 15,
        #     ego_start_trans.rotation
        # )
        #
        # bps = [ego_bp, other_bp]
        # transforms = [ego_start_trans, other_start_trans]

        # Create Camera to follow them
        spectator = world.get_spectator()

        num_episodes = 10
        num_timesteps = 1000
        for ep_id in range(num_episodes):

            ego_bp = bpl.find('vehicle.mercedes.coupe_2020')
            ego_bp.set_attribute('role_name', 'ego')
            ego_start_trans = carla.Transform(Location(207, 133, 0.1), Rotation(0, 0, 0))
            ego_v = world.spawn_actor(ego_bp, ego_start_trans)

            actor_list = [ego_v]

            # ego_agent = CustomAgent(ego_v)
            # ego_agent = BasicAgent(ego_v)
            ego_v.set_autopilot(True)
            vehicle_stats = [VehicleStat(actor.get_location(), 0.0) for actor in actor_list]

            # for actor in others:
            #     actor.get_autopilot(True)

            for i in range(num_timesteps):
                # print(f"Time Step {i}")
                w_frame = world.tick()

                new_vehicle_stats = [update_vehicle_stats(actor, vs) for actor, vs in zip(actor_list, vehicle_stats)]

                spectator.set_transform(carla.Transform(actor_list[0].get_transform().location + Location(z=30),
                                                        Rotation(pitch=-90)))

                # ego_v.apply_control(ego_agent.run_step(None, None))
                # ego_v.apply_control(ego_agent.run_step())

            delete_actors(client, actor_list)

    finally:
        delete_actors(client, actor_list)
        print("Done")


if __name__ == "__main__":
    run()
