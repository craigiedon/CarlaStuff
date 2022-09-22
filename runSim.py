import argparse

import stl
from CEMCarData import extract_dist, range_norm
from repeating_braking import car_braking_CEM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adaptive Importance Sampling for CARLA Automated Braking')
    parser.add_argument('pem',
                        help="File path for the perception error model used for obstacle detection during simulation")
    parser.add_argument('--stages', help="Number of cross entropy importance sampling stages to run", default=10,
                        type=int)
    parser.add_argument('--sims', help="Number of simulations to run per adaptation stage", default=100, type=int)
    parser.add_argument('--timesteps', help="Number of timesteps to run each simulation", default=100, type=int)
    parser.add_argument('-m', '--metric',
                        choices=['classic', 'agm', 'smooth-cumulative'], default='classic', dest='metric',
                        help="The STL Robustness Metric to be used in evaluating trajectories. Choose from classic "
                             "spatial robustness, arithmetic-geomegeric mean, and smooth cumulative robustness.")
    parser.add_argument('--e-name',
                        help="Name of experiment, used when creating folders for simulation and model outputs")
    parser.add_argument('--render', '-r', action='store_true', help="Whether to render simluation window or not")

    args = parser.parse_args()

    classic_stl_spec = stl.G(stl.GEQ0(lambda x: extract_dist(x) - 2.0), 0, args.timesteps - 1)
    classic_rob_f = lambda rollout: stl.stl_rob(classic_stl_spec, rollout, 0)

    agm_stl_spec = stl.G(stl.GEQ0(lambda x: (range_norm(extract_dist(x), 0, 13.0) - range_norm(2.0, 0.0, 13.0))), 0, args.timesteps - 1)
    agm_rob_f = lambda rollout: stl.agm_rob(agm_stl_spec, rollout, 0)

    sc_rob_f = lambda rollout: stl.sc_rob_pos(classic_stl_spec, rollout, 0, 50)

    if args.metric == 'classic':
        safety_f = classic_rob_f
    elif args.metric == 'agm':
        safety_f = agm_rob_f
    elif args.metric == 'smooth-cumulative':
        safety_f = sc_rob_f

    car_braking_CEM(args.pem, args.stages, args.sims, args.timesteps + 100, 100, safety_f, args.e_name)
