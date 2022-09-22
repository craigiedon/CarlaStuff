import argparse

from CEMCarData import analyze_rollouts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze simulation rollouts from the adaptive importance sampler for number/likelihood of failure cases")

    parser.add_argument("rollouts-folder")
    parser.add_argument("pem-path")
    parser.add_argument("cem-path")
    parser.add_argument('-m', '--metric',
                    choices=['classic', 'agm', 'smooth-cumulative'], default='classic', dest='metric',
                    help="The STL Robustness Metric to be used in evaluating trajectories. Choose from classic "
                         "spatial robustness, arithmetic-geomegeric mean, and smooth cumulative robustness.")
    args = parser.parse_args()

    analyze_rollouts(args.rollouts_folder, args.pem_path, args.cem_path, args.metric)
