from dataclasses import dataclass
from math import ceil
from typing import Tuple, List, Callable, Any, Sequence
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Car simulation with probability p of detection
# Start at distance 100. For 100 steps -
import scipy
import torch
from matplotlib import pyplot as plt
from scipy.stats import sem


@dataclass
class SimParams:
    total_dist: int
    braking_dist: int
    safe_dist: int
    tru_det_p: float


def car_sim(sp: SimParams) -> Tuple[torch.tensor, torch.tensor]:
    assert 0 < sp.tru_det_p < 1
    assert 0 < sp.braking_dist < 100

    current_dist = sp.total_dist
    stopped = False
    detections = []
    dists = []

    for t in range(sp.total_dist):
        r = np.random.random()
        detected = r < sp.tru_det_p

        detections.append(detected)
        dists.append(current_dist)

        if not stopped:
            if detected and current_dist <= sp.braking_dist:
                stopped = True
            else:
                current_dist -= 1

    return torch.tensor(detections, dtype=torch.bool), torch.tensor(dists, dtype=torch.float)


def cheat_sim(log_det_func: Callable, sp: SimParams) -> Tuple[torch.tensor, torch.tensor]:
    current_dist = sp.total_dist
    stopped = False

    detections = []
    dists = []

    for t in range(sp.total_dist):
        r = np.random.random()
        threshold = log_det_func(current_dist).exp()
        detected = r < threshold

        detections.append(detected)
        dists.append(current_dist)

        if not stopped:
            if detected and current_dist <= sp.braking_dist:
                stopped = True
            else:
                current_dist -= 1

    return torch.tensor(detections, dtype=torch.bool), torch.tensor(dists, dtype=torch.float)


def rule_based_det(current_dist: float, safe_dist: float, braking_dist: float, det_p: float) -> float:
    return 0.0 if safe_dist <= current_dist <= braking_dist else det_p


def crude_monte_carlo(sim: Callable, safety_func: Callable, num_samples: int) -> float:
    assert num_samples > 0
    safety_vals = np.array([1.0 if safety_func(sim()) else 0.0 for i in range(num_samples)])
    print(f"Num Unsafe: {len(safety_vals[safety_vals == 0.0])} / {len(safety_vals)}")
    # print("Total Samples: ", num_samples)
    unsafe_prob = 1.0 - np.mean(safety_vals)
    unsafe_sem = sem(safety_vals)
    print("Unsafe prob: ", unsafe_prob)
    # print("sem: ", unsafe_sem)
    print(f"Confidence Interval \u00B1{1.96 * unsafe_sem}")
    return unsafe_prob


def rho_quantile(vals: Sequence[Any], rho: float, min_val: float) -> float:
    descending_safety = sorted(vals, reverse=True)
    rho_q = descending_safety[ceil((1.0 - rho) * len(vals))]
    return max(rho_q, min_val)


def CE_Recursive(model, dets, dists, lls_tru, lls_model, sp: SimParams, num_sims: int, solver_iterations: int):
    # Compute the quantile yt of performances provided yt <= y, else yt = y
    performances = torch.amin(dists, dim=1)
    threshold = rho_quantile(performances, 0.01, sp.safe_dist)
    print(f"Threshold: {threshold}, Safe_Dist: {sp.safe_dist}")

    # Solve stochastic program
    indicator_f = (performances < threshold).detach()
    imp_weights = (lls_tru - lls_model).exp().detach()

    optimizer = torch.optim.Adam(model.parameters())
    for solver_i in range(solver_iterations):
        optimizer.zero_grad()

        ll_params = batch_trace_ll(dets, dists, model)

        loss = -(imp_weights[indicator_f] * ll_params[indicator_f]).sum() / num_sims

        if torch.isnan(loss).item() or torch.isinf(loss).item():
            # rare_ll_params = ll_params[indicator_f]
            # rare_imp_weights = imp_weights[indicator_f]
            print("Nan!")

        loss.backward()

        for name, param in model.named_parameters():
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print("Got a divergence!")
                print(f"Name {name}, Params: {param}, Grads: {param.grad}")

        optimizer.step()
        if solver_i % 500 == 0:
            print(f"i: {solver_i}, Loss - {loss}")

    xs = np.linspace(1, sp.total_dist)
    ys = model(torch.tensor(xs, dtype=torch.float).reshape(-1, 1)).exp().detach().numpy().reshape(-1)
    plt.plot(xs, ys)
    plt.show()

    if threshold == sp.safe_dist:
        return

    traces = [cheat_sim(lambda s: model(torch.tensor(s, dtype=torch.float).unsqueeze(0)), sp) for _ in range(num_sims)]
    next_dets = torch.stack([t[0] for t in traces])
    next_dists = torch.stack([t[1] for t in traces])
    next_lls_tru = batch_trace_ll_const(next_dets, next_dists, sp.tru_det_p)
    next_lls_model = batch_trace_ll(next_dets, next_dists, model)

    CE_Recursive(model, next_dets, next_dists, next_lls_tru, next_lls_model, sp, num_sims, solver_iterations)


class CEModel(nn.Module):
    def __init__(self, norm_val):
        super().__init__()
        # self.ff_1 = nn.Linear(1, 10)
        # self.ff_2 = nn.Linear(10, 10)
        # self.ff_3 = nn.Linear(10, 1)
        self.ff = nn.Linear(4,1)
        self.ls = nn.LogSigmoid()
        self.norm_val = norm_val

    def forward(self, x):
        x = x / self.norm_val
        x2 = x.pow(2.0)
        x3 = x.pow(3.0)
        x4 = x.pow(4.0)
        bases = torch.cat((x, x2, x3, x4), dim=-1)
        x = self.ls(self.ff(bases))
        # x = F.leaky_relu(self.ff_1(x))
        # x = F.leaky_relu(self.ff_2(x))
        # x = self.ls(self.ff_3(x))
        return x


def cross_entropy_method(sp: SimParams, sims_adaptive_stage: int, sims_final: int, solve_its: int):
    # Model instantiation
    model = CEModel(sp.total_dist)

    # shitty_det_p = 0.1
    shitty_sp = SimParams(sp.total_dist, sp.braking_dist, sp.safe_dist, 0.5)

    traces = [car_sim(shitty_sp) for _ in range(sims_adaptive_stage)]
    first_dets = torch.stack([t[0] for t in traces])
    first_dists = torch.stack([t[1] for t in traces])
    first_lls_tru = batch_trace_ll_const(first_dets, first_dists, sp.tru_det_p)
    first_lls_model = batch_trace_ll_const(first_dets, first_dists, shitty_sp.tru_det_p)
    # first_lls_model = batch_trace_ll(first_dets, first_dists, model)

    CE_Recursive(model, first_dets, first_dists, first_lls_tru, first_lls_model, sp, sims_adaptive_stage, solve_its)

    xs = np.linspace(1, sp.total_dist)
    ys = model(torch.tensor(xs, dtype=torch.float).reshape(-1, 1)).exp().detach().numpy().reshape(-1)
    plt.plot(xs, ys)
    plt.show()

    # Estimate rare-event probability using final params and new sample set
    new_sample = [cheat_sim(lambda s: model(torch.tensor(s, dtype=torch.float).unsqueeze(0)), sp) for _ in
                  range(sims_final)]
    new_dets = torch.stack([t[0] for t in new_sample])
    new_dists = torch.stack([t[1] for t in new_sample])
    new_performance = torch.amin(new_dists, dim=1)
    new_indicator = new_performance < sp.safe_dist

    num_rare_events = new_indicator.sum()
    print(f"Rare Events: {num_rare_events}")

    new_ll_trus = batch_trace_ll_const(new_dets, new_dists, sp.tru_det_p)
    new_ll_model = batch_trace_ll(new_dets, new_dists, model)
    rare_ev_probs = new_indicator * (new_ll_trus - new_ll_model).exp()
    rare_ev_prob = rare_ev_probs.mean()

    print(f"Rare Event Prob: {rare_ev_prob}")

    return rare_ev_prob


def importance_sampling(importance_sim: Callable, real_log_pdf: Callable, importance_log_pdf: Callable,
                        safety_func: Callable,
                        num_samples: int) -> float:
    assert num_samples > 0
    weights = []
    for i in range(num_samples):
        trace = importance_sim()
        crash_value = 0.0 if safety_func(trace) else 1.0
        log_px = real_log_pdf(trace)
        log_qx = importance_log_pdf(trace)
        weights.append(crash_value * np.exp(log_px - log_qx))

    unsafe_prob = np.mean(weights)
    unsafe_sem = sem(weights)
    print("Unsafe prob: ", np.mean(weights))
    # print("SEM: ", sem(weights))
    print(f"Confidence Interval \u00B1{1.96 * unsafe_sem}")
    return unsafe_prob


def log1mexp(x):
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -x.abs()
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def batch_trace_ll(dets_batch, dists_batch, log_dp_func: nn.Module) -> torch.tensor:
    num_sims = len(dets_batch)
    model_log_dps = log_dp_func(dists_batch.view(-1, 1)).view(num_sims, -1)
    return (dets_batch * model_log_dps + ~dets_batch * log1mexp(model_log_dps)).sum(dim=1)


def batch_trace_ll_const(dets_batch, dists_batch, det_p: float) -> torch.tensor:
    return (det_p * dets_batch + ~dets_batch * (1.0 - det_p)).log().sum(dim=1)


def trace_ll_tensor(trace, det_p_func: nn.Module) -> torch.tensor:
    dets = torch.tensor([s[0] for s in trace], dtype=torch.bool)
    dists = torch.tensor([s[1] for s in trace], dtype=torch.float).view(-1, 1)
    model_det_ps = det_p_func(dists).view(-1)

    return (dets * model_det_ps + ~dets * (1.0 - model_det_ps)).log().sum()


def trace_ll(trace: List[Tuple[bool, int]], det_p: float) -> float:
    log_probs = [np.log(det_p) if x[0] else np.log((1.0 - det_p)) for x in trace]
    log_likelihood = np.sum(log_probs)
    # print("LL: ", log_likelihood)
    # print("Prob: ", np.exp(log_likelihood))
    return log_likelihood


# def cheat_ll(trace: List[Tuple[bool, int]], det_p: float, braking_dist: int, safe_dist: int) -> float:
#     locked_in = braking_dist - (safe_dist - 1)
#
#     if is_safe(trace, safe_dist):
#         return np.log(0.0)
#
#     log_probs = [np.log(det_p) if x[0] else np.log((1.0 - det_p)) for x in trace]
#     log_likelihood = np.sum(log_probs) - locked_in * np.log(1.0 - det_p)
#     return log_likelihood


# im_weight = 0.2
sp = SimParams(50, 25, 10, 0.9)
tru_sim = lambda: car_sim(sp)
# im_sim = lambda: cheat_sim(lambda s: stateless_det(s, im_weight), 10, 5, 3)
#

# print()
# print("IS")
# im_prob = importance_sampling(im_sim, lambda t: trace_ll(t, tru_weight), lambda t: trace_ll(t, im_weight),
#                               lambda s: is_safe(s, 3), 1000)

print("Cross Entropy: ")
cross_entropy_method(sp, 1000, 1000, 10000)

print("Crude")
prob = crude_monte_carlo(tru_sim, lambda s: torch.amin(s[1]) >= sp.safe_dist, 100000)
