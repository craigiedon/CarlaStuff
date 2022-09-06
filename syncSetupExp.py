import pickle

import carla
from carla import Rotation, Location
import pygame
import torch
from carla import ColorConverter

from carlaUtils import set_weather, set_sync, create_cam, to_data_in, to_salient_var, dummy_detector, retrieve_data, \
    norm_salient_input, rollout_nll, Detector_Outputs, SimSnapshot
from customAgent import CustomAgent
from pems import load_model_det, PEMClass_Deterministic, PEMReg_Aleatoric
from render_utils import world_to_cam_viewport, depth_array_to_distances, get_image_as_array, draw_image


# class Sim_Snapshot:
#     def __init__(self,
#                  time_step: int,
#                  model_det: bool,
#                  true_det: bool,
#                  true_centre: Optional[np.ndarray],
#                  model_centre: Optional[np.ndarray],
#                  cam_trans: carla.Transform,
#                  ego_v: carla.Vehicle,
#                  adv_v: carla.Vehicle):
#
#         self.time_step = time_step
#         self.model_det: model_det
#         self.true_det = true_det
#         if true_centre is not None:
#             self.true_centre: Tuple[float, float] = tuple(true_centre)
#         else:
#             self.true_centre = None
#
#         if model_centre is not None:
#             self.model_centre: Tuple[float, float] = tuple(model_centre)
#         self.cam_loc = to_loc_tuple(cam_trans)
#         self.cam_rot = to_rot_tuple(cam_trans)
#         self.ego_loc = to_loc_tuple(ego_v.get_transform())
#         self.ego_rot = to_rot_tuple(ego_v.get_transform())
#         self.adv_loc = to_loc_tuple(adv_v.get_transform())
#         self.adv_rot = to_rot_tuple(adv_v.get_transform())


# class SnapshotEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, Sim_Snapshot):
#             return obj.__dict__
#         return json.JSONEncoder.default(self, obj)


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

        # Spawn the ego vehicle
        ego_bp = bpl.find('vehicle.lincoln.mkz_2017')
        ego_bp.set_attribute('role_name', 'ego')
        ego_vehicle = world.spawn_actor(ego_bp, carla.Transform(Location(207, 133, 0.5), Rotation(0, 0, 0)))
        world.tick()
        actor_list.append(ego_vehicle)

        # Load Perception model
        pem_class = load_model_det(PEMClass_Deterministic(14, 1, use_cuda=False),
                                   "models/det_baseline_full/pem_class_train_full")
        pem_reg = load_model_det(PEMReg_Aleatoric(14, 2, use_cuda=False), "models/al_reg_full/pem_reg_al_full")
        norm_stats = torch.load("models/norm_stats_mu.pt"), torch.load("models/norm_stats_std.pt")
        n_func = lambda s_inputs, norm_dims: norm_salient_input(s_inputs, norm_stats[0], norm_stats[1], norm_dims)

        # Create Cameras
        cam_w, cam_h = 1242, 375
        ego_cam, rgb_queue = create_cam(world, ego_vehicle, (cam_w, cam_h), 82, Location(2, 0, 1.76), Rotation())
        depth_cam, depth_queue = create_cam(world, ego_vehicle, (cam_w, cam_h), 82, Location(2, 0, 1.76), Rotation(),
                                            'depth')

        # Spawn other vehicle
        other_bp = bpl.find('vehicle.nissan.patrol')
        ego_pos = ego_vehicle.get_location()
        ego_forward = ego_vehicle.get_transform().get_forward_vector()
        ego_right = ego_vehicle.get_transform().get_right_vector()

        other_vehicle = world.spawn_actor(other_bp, carla.Transform(
            ego_vehicle.get_location() + ego_forward * 20, ego_vehicle.get_transform().rotation))
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
        # py_display = pygame.display.set_mode((cam_w, cam_h * 2), pygame.HWSURFACE | pygame.DOUBLEBUF)
        py_display = pygame.display.set_mode((cam_w, cam_h), pygame.HWSURFACE | pygame.DOUBLEBUF)

        lights_list = world.get_actors().filter("*traffic_light*")
        adv_v = world.get_actors().filter("vehicle.nissan.patrol")[0]

        for l in lights_list:
            l.set_red_time(100)

        rollout_log = []

        for i in range(350):
            w_frame = world.tick()

            # Follow ego on server window
            spectator.set_transform(
                carla.Transform(ego_vehicle.get_transform().location + Location(z=30), Rotation(pitch=-90)))

            # Render sensor output
            data_timeout = 2.0
            current_rgb = retrieve_data(rgb_queue, w_frame, data_timeout)
            current_depth = retrieve_data(depth_queue, w_frame, data_timeout)

            # print(current_rgb.frame, current_depth.frame, w_frame)

            distance_array = depth_array_to_distances(get_image_as_array(current_depth))
            current_depth.convert(ColorConverter.LogarithmicDepth)
            depth_im_array = get_image_as_array(current_depth)

            draw_image(py_display, get_image_as_array(current_rgb))
            # draw_image(py_display, depth_im_array, offset=(0, cam_h))

            d_in = to_data_in(ego_cam.get_transform(), ego_cam.attributes, adv_v)
            salient_vars = to_salient_var(d_in, n_func)

            tru_adv_vp = world_to_cam_viewport(ego_cam.get_transform(), ego_cam.attributes,
                                               adv_v.get_location() + Location(0, 0,
                                                                               adv_v.bounding_box.extent.z)).astype(int)

            # detection, cam_centroid, obstacle_depth = dummy_detector(salient_vars, adv_v, ego_cam, distance_array, 0.9)
            # dummy_det, dummy_centroid, dummy_depth = dummy_detector(salient_vars, adv_v, ego_cam, distance_array, 0.9)
            # m_detection, m_centroid, m_depth = model_detector(salient_vars, adv_v, ego_cam, distance_array, pem_class,
            #                                                   pem_reg)
            m_detection, m_centroid, m_depth = dummy_detector(salient_vars, adv_v, ego_cam, distance_array, 1.0)

            d_outs = Detector_Outputs(tuple(tru_adv_vp),
                                      tuple(m_centroid) if m_centroid is not None else None,
                                      True,
                                      m_detection)

            pygame.draw.circle(py_display, (0, 255, 0), (tru_adv_vp[0], tru_adv_vp[1]), 5.0)

            rollout_log.append(SimSnapshot(w_frame, d_in, d_outs))

            if m_detection:
                # print("BB-dist: \t", np.min([av.distance(ev) for av in adv_bb_verts for ev in ego_bb_verts]))
                pygame.draw.circle(py_display, (255, 0, 0), (m_centroid[0], m_centroid[1]), 5.0)
                # pygame.draw.rect(py_display, (255, 0, 0), pygame.Rect(cam_centroid[0] - 5, cam_h + cam_centroid[1] - 5, 10, 10), 2)

            pygame.display.flip()
            ego_vehicle.apply_control(agent.run_step(m_centroid, m_depth))

        with open("data_outs/rollout_log.pickle", 'wb') as f:
            pickle.dump(rollout_log, f)

        with open("data_outs/rollout_log.pickle", 'rb') as f:
            loaded_roll = pickle.load(f)

        print(loaded_roll)

        nll = rollout_nll(rollout_log, pem_class, pem_reg, n_func)

    finally:
        ego_cam.destroy()
        depth_cam.destroy()

        print("Actors to destroy: ", actor_list)
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        pygame.quit()

        print("Done")


if __name__ == "__main__":
    run()


