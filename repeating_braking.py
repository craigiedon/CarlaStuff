import dataclasses
import json
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime

import carla
import pygame
import torch
from carla import Location, Rotation, Vehicle, ColorConverter, Vector3D
from navigation.basic_agent import BasicAgent
from navigation.local_planner import LocalPlanner

from CEMCarData import one_step_cem
from adaptiveImportanceSampler import FFPolicy
from carlaUtils import set_sync, set_weather, delete_actors, setup_actors, create_cam, norm_salient_input, \
    retrieve_data, to_data_in, to_salient_var, dummy_detector, Detector_Outputs, SimSnapshot, set_rendering, \
    model_detector, SnapshotEncoder, proposal_model_detector
from customAgent import CustomAgent
from pems import load_model_det, PEMClass_Deterministic, PEMReg_Aleatoric
from render_utils import depth_array_to_distances, get_image_as_array, draw_image, world_to_cam_viewport, \
    viewport_to_ray, viewport_to_vehicle_depth


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
        set_rendering(world, client, False)
        set_weather(world, 0, 0, 0, 0, 0, 75)

        is_rendered = not world.get_settings().no_rendering_mode
        # is_rendered = False
        print("Is it being rendered?:", is_rendered)

        bpl = world.get_blueprint_library()

        # Load Perception model
        pem_class = load_model_det(PEMClass_Deterministic(14, 1),
                                   "models/det_baseline_full/pem_class_train_full").cuda()
        pem_reg = load_model_det(PEMReg_Aleatoric(14, 2), "models/al_reg_full/pem_reg_al_full").cuda()
        norm_stats = torch.load("models/norm_stats_mu.pt"), torch.load("models/norm_stats_std.pt")
        n_func = lambda s_inputs, norm_dims: norm_salient_input(s_inputs, norm_stats[0], norm_stats[1], norm_dims)

        # Load proposal sampler
        proposal_model = load_model_det(FFPolicy(1, norm_tensor=torch.tensor([12.0], device="cuda")),
                                        "models/CEMs/pretrain_e100_PEM.pyt").cuda()

        ego_bp = bpl.find('vehicle.mercedes.coupe_2020')
        ego_bp.set_attribute('role_name', 'ego')
        ego_start_trans = carla.Transform(Location(257, 133, 0.1), Rotation(0, 0, 0))

        other_bp = bpl.find('vehicle.dodge.charger_2020')
        other_start_trans = carla.Transform(
            ego_start_trans.location + ego_start_trans.get_forward_vector() * 15,
            ego_start_trans.rotation
        )
        # transforms = [ego_start_trans, other_start_trans]

        # Create Camera to follow them
        spectator = world.get_spectator()

        cam_w, cam_h = 1242, 375

        if is_rendered:
            pygame.init()
            # py_display = pygame.display.set_mode((cam_w, cam_h * 2), pygame.HWSURFACE | pygame.DOUBLEBUF)
            py_display = pygame.display.set_mode((cam_w, cam_h), pygame.HWSURFACE | pygame.DOUBLEBUF)

        num_episodes = 10000
        num_timesteps = 200
        num_cem_stages = 1
        vel_burn_in_time = 100

        models_ts = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        data_folder_name = os.path.join("sim_data", models_ts)
        print(data_folder_name)
        os.makedirs(data_folder_name, exist_ok=True)

        for c_stage in range(num_cem_stages):
            rollout_logs = []
            for ep_id in range(num_episodes):
                rollout = []
                start_time = time.time()

                ego_v = world.spawn_actor(ego_bp, ego_start_trans)
                other_v = world.spawn_actor(other_bp, other_start_trans)

                # bps = [ego_bp, other_bp]

                # Create Cameras
                ego_cam, rgb_queue = create_cam(world, ego_v, (cam_w, cam_h), 82, Location(2, 0, 1.76), Rotation())
                depth_cam, depth_queue = create_cam(world, ego_v, (cam_w, cam_h), 82, Location(2, 0, 1.76),
                                                    Rotation(),
                                                    'depth')

                actor_list = [ego_v, other_v]

                other_v.set_autopilot(True)

                vehicle_stats = [VehicleStat(actor.get_location(), 0.0) for actor in actor_list]
                lights = world.get_actors().filter("*traffic_light*")

                w_frame = world.tick()

                for l in lights:
                    l.set_state(carla.TrafficLightState.Red)
                    l.freeze(True)

                ego_v.set_autopilot(True)

                for i in range(num_timesteps):
                    w_frame = world.tick()

                    # Render sensor output
                    if is_rendered:
                        spectator.set_transform(carla.Transform(actor_list[0].get_transform().location + Location(z=30),
                                                                Rotation(pitch=-90)))
                        data_timeout = 2.0
                        current_rgb = retrieve_data(rgb_queue, w_frame, data_timeout)
                        current_depth = retrieve_data(depth_queue, w_frame, data_timeout)

                        distance_array = depth_array_to_distances(get_image_as_array(current_depth))
                        current_depth.convert(ColorConverter.LogarithmicDepth)
                        depth_im_array = get_image_as_array(current_depth)

                        draw_image(py_display, get_image_as_array(current_rgb))
                        # draw_image(py_display, depth_im_array, offset=(0, cam_h))

                    if i < vel_burn_in_time:
                        continue

                    if i == vel_burn_in_time:
                        ego_v.set_autopilot(False)
                        ego_agent = CustomAgent(ego_v)

                    new_vehicle_stats = [update_vehicle_stats(actor, vs) for actor, vs in
                                         zip(actor_list, vehicle_stats)]

                    d_in = to_data_in(ego_cam.get_transform(), ego_cam.attributes, other_v)
                    salient_vars = to_salient_var(d_in, n_func)

                    tru_adv_vp = world_to_cam_viewport(ego_cam.get_transform(), ego_cam.attributes,
                                                       other_v.get_location() + Location(0, 0,
                                                                                         other_v.bounding_box.extent.z)).astype(
                        int)

                    tru_depth = viewport_to_vehicle_depth(world, ego_cam, tru_adv_vp)

                    m_detection, m_centroid, m_depth = dummy_detector(salient_vars, other_v, ego_cam, world, 0.5)
                    # # m_detection, m_centroid, m_depth = model_detector(salient_vars, other_v, ego_cam, world, pem_class,
                    # #                                                   pem_reg)
                    # m_detection, m_centroid, m_depth = proposal_model_detector(tru_depth, other_v, ego_cam, world,
                    #                                                            proposal_model)

                    d_outs = Detector_Outputs(true_centre=tuple(tru_adv_vp.tolist()),
                                              true_distance=tru_depth,
                                              predicted_centre=tuple(m_centroid) if m_centroid is not None else None,
                                              predicted_distance=m_depth,
                                              true_det=True,
                                              model_det=m_detection)

                    # print(f"Tru Depth: {tru_depth} Model Depth: {m_depth}")

                    if is_rendered:
                        pygame.draw.circle(py_display, (0, 255, 0), (tru_adv_vp[0], tru_adv_vp[1]), 5.0)

                        if m_detection:
                            pygame.draw.circle(py_display, (255, 0, 0), (m_centroid[0], m_centroid[1]), 5.0)

                        pygame.display.flip()

                    rollout.append(SimSnapshot(w_frame, d_in, d_outs,
                                               ego_v.get_velocity().length(),
                                               ego_v.get_acceleration().length(),
                                               other_v.get_velocity().length(),
                                               other_v.get_acceleration().length()))

                    ego_v.apply_control(ego_agent.run_step(d_outs.predicted_centre, m_depth))

                delete_actors(client, actor_list)
                ego_cam.destroy()
                depth_cam.destroy()
                print(f"Ep {ep_id} time: {time.time() - start_time}")

                # Write the episode data to file
                stage_path = os.path.join(data_folder_name, f"s{c_stage}")
                os.makedirs(stage_path, exist_ok=True)
                with open(os.path.join(stage_path, f"e{ep_id}.json"), 'w') as fp:
                    json.dump([dataclasses.asdict(s) for s in rollout], fp)
                rollout_logs.append(rollout)

            # pem_class.cuda()
            # proposal_model.cuda()
            # one_step_cem(rollout_logs, proposal_model, pem_class, norm_stats, False, f"models/CEMs/full_loop_s{c_stage}.pyt")
            print("Done CEM")
    finally:
        delete_actors(client, actor_list)

        if is_rendered:
            pygame.quit()
        print("Done")


if __name__ == "__main__":
    run()
