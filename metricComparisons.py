# Load in 100 / 1000 rollouts from a random policy (e.g., naive 50)
from matplotlib import pyplot as plt

import stl
from CEMCarData import load_rollouts, extract_dist
from repeating_braking import create_safety_func


def run():
    ep_rollouts = load_rollouts("sim_data/mixed_dummy05_baseline_10000/s0", 1000)

    stl_spec = stl.G(stl.GEQ0(lambda x: extract_dist(x) - 2.0), 0, 99)
    metric_names = ["classic", "agm", "smooth_cumulative"]
    metrics = [create_safety_func(mn, stl_spec) for mn in metric_names]

    # Not interested in ones which fail for purposes of adaptive analysis, only ones close to failing
    filtered_rollouts = list(filter(lambda r: metrics[0](r) > 0.0, ep_rollouts))
    print(len(filtered_rollouts))

    metric_bests = []
    metric_top_5ps = []

    for mn, metric in zip(metric_names, metrics):
        indexed_safety_vals = list(enumerate([metric(rollout) for rollout in filtered_rollouts]))
        sorted_vals = sorted(indexed_safety_vals, key=lambda x: x[1])
        best_rollout = filtered_rollouts[sorted_vals[0][0]]
        metric_bests.append(best_rollout)
        top_5p = [i for i, v in sorted_vals[0:int(0.05 * len(sorted_vals))]]
        metric_top_5ps.append(set(top_5p))
        print(top_5p)
        plt.plot(range(len(best_rollout)), [s.outs.true_distance for s in best_rollout], label=mn)

    print(metric_top_5ps[0].difference(metric_top_5ps[1], metric_top_5ps[2]))
    print(metric_top_5ps[1].difference(metric_top_5ps[0], metric_top_5ps[2]))
    print(metric_top_5ps[2].difference(metric_top_5ps[0], metric_top_5ps[1]))
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.show()

    # Histogram / CDF of safety value for each metric
    # Create index set for top 95% percentile of each. Check for set equality. Check for top value. Check for bottom value. Check for set subtraction / uniquenesses

    # For a given rollout: Plot the distance from car in front v time. Should be doable by looking at the SimSnapshot fields


if __name__ == "__main__":
    run()
