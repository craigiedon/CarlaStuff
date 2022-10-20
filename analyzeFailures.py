import argparse

from CEMCarData import chart_avg_rollout_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze simulation rollouts from the adaptive importance sampler for number/likelihood of failure cases")
    parser.add_argument("experiment_folders", nargs="+")

    args = parser.parse_args()

    chart_avg_rollout_metrics(args.experiment_folders)