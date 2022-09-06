import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import average
from torch import nn, autograd, BoolTensor, FloatTensor, IntTensor, LongTensor
import torch.nn.functional as F


@dataclass
class SimParams:
    total_dist: int
    braking_dist: int
    safe_dist: int
    tru_det_p: float


@dataclass
class SimState:
    distance: int
    stopped: bool

    def to_tensor(self, use_cuda=True):
        if use_cuda:
            return torch.tensor([self.distance, self.stopped], dtype=torch.float).cuda()

        return torch.tensor([self.distance, self.stopped], dtype=torch.float)


def step_state(detect: bool, s: SimState, sp: SimParams) -> SimState:
    if s.stopped or (detect and s.distance <= sp.braking_dist):
        return SimState(s.distance, True)

    return SimState(s.distance - 1, False)


SimTraj = List[Tuple[SimState, bool]]
SS_Policy = Callable[[bool, SimState], float]


class FFPolicy(nn.Module):
    def __init__(self, inp_dim: int, norm_tensor=None):
        super().__init__()
        hidden_dims = 32
        self.ff_1 = nn.Linear(inp_dim, hidden_dims)
        self.ff_2 = nn.Linear(hidden_dims, hidden_dims)
        self.ff_3 = nn.Linear(hidden_dims, 2)
        self.log_sfm = nn.LogSoftmax(1)

        self.norm_tensor = norm_tensor

        assert len(norm_tensor) == inp_dim

    def forward(self, x):
        if self.norm_tensor is not None:
            x = x / self.norm_tensor

        x = F.relu(self.ff_1(x))
        x = F.relu(self.ff_2(x))
        x = self.log_sfm(self.ff_3(x))

        return x


def car_rollout(policy: SS_Policy, sp: SimParams) -> SimTraj:
    sim_traj = []
    current_state = SimState(sp.total_dist, False)

    for t in range(sp.total_dist):
        r = np.random.random()

        threshold = policy(True, current_state)
        detected = r < threshold

        sim_traj.append((current_state, detected))

        # Failure states are terminal states
        if current_state.distance < sp.safe_dist:
            return sim_traj

        # Next state
        current_state = step_state(detected, current_state, sp)
    return sim_traj


# def policy_from_log_nn(a: bool, s: SimState, log_nn: nn.Module) -> float:
#     pos_det_prob = log_nn(s.to_tensor().unsqueeze(0)).exp()
#     return pos_det_prob if a else 0.0 - pos_det_prob

def policy_from_log_sfm(a: bool, s: SimState, log_sfm: nn.Module) -> float:
    l_sfm = log_sfm(s.to_tensor().unsqueeze(0))
    return l_sfm[0, (1 if a else 0)].exp()


def safety_value(tau: SimTraj) -> float:
    return min([s.distance for s, a in tau])


def disturbance_log_prob(actions: torch.BoolTensor, sp: SimParams) -> torch.tensor:
    det_p = torch.full((len(actions),), sp.tru_det_p, device=actions.device)
    return (actions * det_p + ~actions * (1.0 - det_p)).log()


def simple_policy(det: bool, sp: SimParams) -> float:
    return sp.tru_det_p if det else (1.0 - sp.tru_det_p)


def best_var_policy(det: bool, s: SimState, sp: SimParams) -> float:
    if sp.safe_dist <= s.distance <= sp.braking_dist:
        return 0.0 if det else 1.0
    else:
        return sp.tru_det_p if det else (1.0 - sp.tru_det_p)


# Pre-training stage (minimizing the KL-Divergence between them?)
def grad_clipper(grad, clip_value: float) -> float:
    return torch.clamp(grad, -clip_value, clip_value)


def get_quantile(vals: List[Any], quantile: float, lower_bound: Optional[float] = None):
    assert 0.0 <= quantile <= 1.0
    quantile_ind = int(quantile * len(vals))
    safety_desc = sorted(vals, reverse=True)

    fail_thresh = safety_desc[quantile_ind]

    if lower_bound is not None:
        return max(fail_thresh, lower_bound)

    return fail_thresh


def pre_train_cem(model: nn.Module, rollout_fn: Callable[[], SimTraj], n_eps: int, sp: SimParams) -> nn.Module:
    model.eval()
    with torch.no_grad():
        pre_train_rollouts = [rollout_fn() for _ in range(n_eps)]
        pre_s_tensors = torch.concat([torch.stack([s.to_tensor() for s, _ in sr[:-1]]) for sr in pre_train_rollouts])
        pre_a_tensors = torch.concat(
            [torch.tensor([a for _, a in sr[:-1]], device=torch.device("cuda")) for sr in
             pre_train_rollouts])

    pre_optim = torch.optim.Adam(model.parameters())

    model.train()
    for epoch in range(1000):
        if epoch % 100 == 0:
            print(f"Pre E: {epoch}")
        pre_lsfms = model(pre_s_tensors).view(-1, 2)
        # pre_nlqs = log1mexp(pre_plqs)
        pre_log_qs = pre_lsfms[:, 0] * ~pre_a_tensors + pre_lsfms[:, 1] * pre_a_tensors
        pre_loss = -pre_log_qs.sum() / n_eps

        pre_optim.zero_grad()
        pre_loss.backward()
        pre_optim.step()

    model.eval()
    return model


def weighted_log_sum_exp(values, weights, dim=None):
    eps = 1e-20
    m = torch.max(values, dim=dim, keepdim=True)[0].squeeze(dim)
    weight_vec = weights.unsqueeze(-1)
    weighed_exps = torch.exp(values - m) * weight_vec
    exp_sum = torch.sum(weighed_exps, dim=dim) + eps
    result = m + torch.log(exp_sum)
    return result


def cross_entropy_train(model: nn.Module,
                        s_tensors: FloatTensor,
                        a_tensors: BoolTensor,
                        ep_strides: LongTensor,
                        log_ps: FloatTensor,
                        fail_indicator: torch.Tensor,
                        n_epochs: int,
                        n_eps: int):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(n_epochs):
        log_sfms = model(s_tensors).view(-1, 2)

        log_qs = (~a_tensors * log_sfms[:, 0]) + (a_tensors * log_sfms[:, 1])
        log_ratios = log_ps - log_qs
        strided_ratios = log_ratios.tensor_split(ep_strides)

        log_weights = torch.concat([rs.sum(0).expand(len(rs)) for rs in strided_ratios])

        smoothing_param = 0.5
        weights = (log_weights * smoothing_param).exp()

        losses = fail_indicator * -1.0 * weights * log_qs
        average_loss = losses.sum() / n_eps

        if weights.isinf().any() or weights.isnan().any() or average_loss.isnan():
            raise RuntimeError("Some sort of Numerical Stability Problem")

        if epoch % 100 == 0:
            print(f"E: {epoch} - {average_loss}")

        optimizer.zero_grad()
        average_loss.backward()
        optimizer.step()


def get_ep_strides(rollouts: List[List[Any]]) -> torch.Tensor:
    # Ignore final logged action --- has no effect
    stride_cum_sum = torch.tensor([len(r) - 1 for r in rollouts]).cumsum(0)
    # Ignore final stride set (as this is just the full length of array)
    # See tensor_split doc for reasoning
    return stride_cum_sum[:-1]


def policy_gradient_AIS(n_levels: int, n_eps: int, n_epochs: int, sp: SimParams):
    q_sfm = FFPolicy(2, torch.tensor([sp.total_dist, 1.0], device="cuda")).cuda()
    for p in q_sfm.parameters():
        p.register_hook(lambda grad: grad_clipper(grad, 1.0))

    q_sfm = pre_train_cem(q_sfm,
                          lambda: car_rollout(lambda a, s: simple_policy(s, dataclasses.replace(sp, tru_det_p=0.9)),
                                              sp), 1000, sp)

    fig, axs = plt.subplots(1, 1)
    for level in range(n_levels):
        q_sfm.eval()
        with torch.no_grad():
            sim_rollouts = [car_rollout(lambda a, s: policy_from_log_sfm(a, s, q_sfm), sp) for _ in range(n_eps)]

        safety_vals = [safety_value(rollout) for rollout in sim_rollouts]
        failure_thresh = get_quantile(safety_vals, 0.98, sp.safe_dist - 1)
        num_fails = len([s for s in safety_vals if s <= failure_thresh])
        print(f"Failure Thresh: {failure_thresh} ({num_fails} / {n_eps} episodes)")

        # Ignore final action: It has no effect as there is no subsequent state, and it does not affect return...
        ep_strides = get_ep_strides(sim_rollouts)
        s_tensors = torch.concat([torch.stack([s.to_tensor() for s, _ in sr[:-1]]) for sr in sim_rollouts])
        a_tensors = torch.concat(
            [torch.tensor([a for _, a in sr[:-1]], device=torch.device("cuda")) for sr in sim_rollouts])
        fail_indicator = torch.concat(
            [torch.full((len(sr) - 1,), sv <= failure_thresh, device=torch.device("cuda")) for sr, sv in
             zip(sim_rollouts, safety_vals)])
        log_ps = disturbance_log_prob(a_tensors, sp)

        assert len(s_tensors) == len(a_tensors) == len(log_ps) == len(fail_indicator)

        cross_entropy_train(q_sfm, s_tensors, a_tensors, ep_strides, log_ps, fail_indicator, n_epochs, n_eps)

        with torch.no_grad():
            ds = torch.arange(sp.safe_dist - 1, sp.total_dist + 1, dtype=torch.float).cuda()
            go_states = torch.column_stack([ds, torch.full_like(ds, 0)]).cuda()
            qs_go = q_sfm(go_states).exp()
            axs.plot(ds.cpu().numpy(), qs_go[:, 1].cpu().numpy(), color='b', alpha=(1.0 + level) / (1.0 + n_levels))

    padding = 0.01
    axs.set_ylim(0, 1)
    axs.set_xlabel("Distance")
    axs.set_ylabel("P(True | dist)")
    axs.axvline(sp.braking_dist, ymin=padding, ymax=1 - padding, color='r', ls='--', alpha=0.5)
    plt.show()
    return q_sfm


# def log1mexp(x):
#     # Computes log(1-exp(-|x|))
#     # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
#     x = -x.abs()
#     return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def rollout_weights(model: nn.Module, rollouts: List[SimTraj], sp: SimParams) -> torch.FloatTensor:
    s_tensors = torch.concat([torch.stack([s.to_tensor() for s, _ in sr[:-1]]) for sr in rollouts])
    a_tensors = torch.concat(
        [torch.tensor([a for _, a in sr[:-1]], device=torch.device("cuda")) for sr in
         rollouts])

    ep_strides = torch.tensor([len(sr) - 1 for sr in rollouts]).cumsum(0)[:-1]
    log_ps = disturbance_log_prob(a_tensors, sp)

    log_sfms = model(s_tensors).view(-1, 2)
    log_qs = ~a_tensors * log_sfms[:, 0] + a_tensors * log_sfms[:, 1]

    log_ratios = log_ps - log_qs
    strided_ratios = log_ratios.tensor_split(ep_strides)
    log_weights = torch.stack([rs.sum(0) for rs in strided_ratios])
    weights = log_weights.exp()

    return weights


if __name__ == "__main__":
    sp = SimParams(100, 50, 30, 0.9)

    cem_model = policy_gradient_AIS(10, 500, 500, sp)
    cem_rollouts = [car_rollout(lambda a, s: policy_from_log_sfm(a, s, cem_model), sp) for _ in range(1000)]

    weights = rollout_weights(cem_model, cem_rollouts, sp)
    safety_vals = [safety_value(rollout) for rollout in cem_rollouts]
    fail_ind = torch.tensor([sv < sp.safe_dist for sr, sv in zip(cem_rollouts, safety_vals)], device=weights.device)
    cem_fail_prob = (weights * fail_ind).mean()

    best_model = lambda a, s: best_var_policy(a, s, sp)
    best_rollouts = [car_rollout(best_model, sp) for _ in range(1000)]

    best_weights = []
    for rollout in best_rollouts:
        log_qs = torch.tensor([best_var_policy(a, s, sp) for s, a in rollout[:-1]]).log()
        log_ps = disturbance_log_prob(torch.tensor([a for _, a in rollout[:-1]]), sp)
        log_ratios = log_ps - log_qs
        weight = log_ratios.sum().exp()
        best_weights.append(weight)

    best_weights = torch.tensor(best_weights)

    best_svs = [safety_value(rollout) for rollout in best_rollouts]
    best_fails = torch.tensor([sv < sp.safe_dist for sr, sv in zip(best_rollouts, best_svs)])
    best_fail_prob = (best_weights * best_fails).mean()

    print(f"Fail Prob Best-Variance-Estimator: {best_fail_prob}")

    print(f"Fail Prob CEM: {cem_fail_prob}")
    print(f"Num Fails: {fail_ind.sum()}")
