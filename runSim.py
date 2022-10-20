import argparse

import carla

from repeating_braking import car_experiment_from_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adaptive Importance Sampling for CARLA Automated Braking')
    parser.add_argument('config_path', help="Path to the experiment configuration file")
    args = parser.parse_args()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    car_experiment_from_file(client, args.config_path)