import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import average
from torch import nn
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
        self.ff_1 = nn.Linear(inp_dim, 32)
        self.ff_2 = nn.Linear(32, 32)
        self.ff_3 = nn.Linear(32, 1)
        self.ls = nn.LogSigmoid()

        self.norm_tensor = norm_tensor

        assert len(norm_tensor) == inp_dim

    def forward(self, x):
        if self.norm_tensor is not None:
            x = x / self.norm_tensor

        x = F.relu(self.ff_1(x))
        x = F.relu(self.ff_2(x))
        x = self.ls(self.ff_3(x))

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


def policy_from_log_nn(a: bool, s: SimState, log_nn: nn.Module) -> float:
    tru_det_prob = log_nn(s.to_tensor().unsqueeze(0)).exp()
    return tru_det_prob if a else 1.0 - tru_det_prob


def safety_value(tau: SimTraj) -> float:
    return min([s.distance for s, a in tau])


def disturbance_log_prob(actions: torch.tensor, sp: SimParams) -> torch.tensor:
    det_p = torch.full_like(actions, sp.tru_det_p)
    return (actions * det_p + (1.0 - actions) * (1.0 - det_p)).log()


def simple_policy(det: bool, sp: SimParams) -> float:
    return sp.tru_det_p if det else (1.0 - sp.tru_det_p)


# Pre-training stage (minimizing the KL-Divergence between them?)
def grad_clipper(grad, clip_value: float) -> float:
    return torch.clamp(grad, -clip_value, clip_value)


def policy_gradient_AIS(n_levels: int, n_eps: int, n_epochs: int, sp: SimParams):
    q_log = FFPolicy(2, torch.tensor([sp.total_dist, 1.0], device=torch.device("cuda"))).cuda()
    for p in q_log.parameters():
        p.register_hook(lambda grad: grad_clipper(grad, 1.0))

    ### Pre-training stage
    # 100 epochs, 10000 data points?

    q_log.eval()
    with torch.no_grad():
        pre_train_rollouts = [car_rollout(lambda a, s: simple_policy(s, sp), sp) for _ in range(10000)]
        pre_s_tensors = torch.concat([torch.stack([s.to_tensor() for s, _ in sr[:-1]]) for sr in pre_train_rollouts])
        pre_a_tensors = torch.concat(
            [torch.tensor([1.0 if a else 0.0 for _, a in sr[:-1]], device=torch.device("cuda")) for sr in
             pre_train_rollouts])

    pre_optim = torch.optim.Adam(q_log.parameters())

    q_log.train()
    for epoch in range(1000):
        if epoch % 100 == 0:
            print(f"Pre E: {epoch}")
        pre_plqs = q_log(pre_s_tensors).view(-1)
        pre_nlqs = log1mexp(pre_plqs)
        pre_log_qs = (pre_a_tensors * pre_plqs) + (1.0 - pre_a_tensors) * pre_nlqs
        pre_loss = -pre_log_qs.sum() / 10000

        pre_optim.zero_grad()
        pre_loss.backward()
        pre_optim.step()

    q_log.eval()
    fig, axs = plt.subplots(1, 1)
    with torch.no_grad():
        ds = torch.arange(sp.safe_dist - 1, sp.total_dist + 1, dtype=torch.float).cuda()
        go_states = torch.column_stack([ds, torch.full_like(ds, 0)]).cuda()
        qs_go = q_log(go_states).exp()
        axs.plot(ds.cpu().numpy(), qs_go[:, 0].cpu().numpy())
        padding = 0.01
        axs.axvline(sp.braking_dist, ymin=padding, ymax=1 - padding, color='r', ls='--', alpha=0.5)
        axs.set_ylim(0, 1)
        axs.set_xlabel("Distance")
        axs.set_ylabel("P(True | dist)")

    plt.show()

    fig, axs = plt.subplots(1, 1)
    for level in range(n_levels):
        q_log.eval()
        with torch.no_grad():
            sim_rollouts = [car_rollout(lambda a, s: policy_from_log_nn(a, s, q_log), sp) for _ in range(n_eps)]
            safety_vals = [safety_value(rollout) for rollout in sim_rollouts]

            quantile_ind = int(0.95 * n_eps)
            safety_desc = sorted(safety_vals, reverse=True)
            failure_thresh = max(safety_desc[quantile_ind], sp.safe_dist - 1)
            num_fails = len([s for s in safety_vals if s <= failure_thresh])
            print(f"Failure Thresh: {failure_thresh} ({num_fails} / {n_eps} episodes)")

        optimizer = torch.optim.Adam(q_log.parameters())

        # Ignore final action: It has no effect as there is no subsequent state, and it does not affect return...
        ep_strides = torch.tensor([len(sr) - 1 for sr in sim_rollouts]).cumsum(0)[:-1]
        s_tensors = torch.concat([torch.stack([s.to_tensor() for s, _ in sr[:-1]]) for sr in sim_rollouts])
        a_tensors = torch.concat(
            [torch.tensor([(1.0 if a else 0.0) for _, a in sr[:-1]], device=torch.device("cuda")) for sr in
             sim_rollouts])
        fail_indicator = torch.concat(
            [torch.full((len(sr) - 1,), sv <= failure_thresh, device=torch.device("cuda")) for sr, sv in
             zip(sim_rollouts, safety_vals)])
        log_ps = disturbance_log_prob(a_tensors, sp)

        assert len(s_tensors) == len(a_tensors) == len(log_ps) == len(fail_indicator)

        q_log.train()
        for epoch in range(n_epochs):
            p_lqs = q_log(s_tensors).view(-1)
            n_lqs = log1mexp(p_lqs)

            log_qs = (a_tensors * p_lqs) + (1.0 - a_tensors) * n_lqs

            log_ratios = log_ps - log_qs
            strided_ratios = log_ratios.tensor_split(ep_strides)
            log_weights = torch.concat([rs.sum(0).expand(len(rs)) for rs in strided_ratios])
            max_log_weight = torch.max(log_weights)
            weights = (log_weights - max_log_weight).exp()

            losses = fail_indicator * -1.0 * weights * log_qs
            average_loss = losses.sum() / n_eps

            if epoch % 100 == 0:
                print(f"E: {epoch} - {average_loss}")

            optimizer.zero_grad()
            average_loss.backward()
            optimizer.step()

        # Sample using q_theta, then sort!
        q_log.eval()
        with torch.no_grad():
            ds = torch.arange(sp.safe_dist - 1, sp.total_dist + 1, dtype=torch.float).cuda()
            go_states = torch.column_stack([ds, torch.full_like(ds, 0)]).cuda()
            qs_go = q_log(go_states).exp()
            axs.plot(ds.cpu().numpy(), qs_go[:, 0].cpu().numpy(), color='b', alpha=(1.0 + level) / (1.0 + n_levels))

    padding = 0.01
    axs.set_ylim(0, 1)
    axs.set_xlabel("Distance")
    axs.set_ylabel("P(True | dist)")
    axs.axvline(sp.braking_dist, ymin=padding, ymax=1 - padding, color='r', ls='--', alpha=0.5)
    plt.show()


def log1mexp(x):
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -x.abs()
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


if __name__ == "__main__":
    policy_gradient_AIS(20, 500, 1000, SimParams(100, 50, 30, 0.9))
