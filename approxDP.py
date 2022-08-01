from dataclasses import dataclass
from typing import Callable, Tuple, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff_1 = nn.Linear(1, 10)
        self.ff_2 = nn.Linear(10, 10)
        self.ff_3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.ff_1(x))
        x = F.relu(self.ff_2(x))
        x = torch.sigmoid(self.ff_3(x))
        return x


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

    def to_dist_tensor(self):
        return torch.tensor(self.distance, dtype=torch.float)


SimTraj = List[Tuple[SimState, bool]]


def mc_policy_eval(n_iter, n_samples, learning_rate):
    sp = SimParams(10, 5, 3, 0.9)
    model = DPModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for _ in range(n_iter):
        optimizer.zero_grad()
        episodes: List[SimTraj] = [car_rollout(model, sp) for _ in range(n_samples)]
        episode_gs = []

        # TODO: Flatten all this into somethign nice for torch, so that the gradients and additions and means etc. make sense
        for ep in episodes:
            end_state = ep[-1][0]
            g_estimates = torch.zeros(len(ep))
            if end_state.distance < sp.safe_dist:
                for t in range(len(ep)):
                    state_probs = torch.tensor([disturbance_prob(ep[i][1], sp) for i in range(t, len(ep))])
                    policy_probs = torch.tensor([policy_prob(ep[i][1], ep[i][0], model, sp) for i in range(t, len(ep))])
                    g_estimates[t] = torch.prod(state_probs / policy_probs)
            episode_gs.append(g_estimates)

        Js = []
        for i in range(len(episodes)):
            for j in range(len(episodes[i])):
                mse = (episode_gs[i][j] - model(episodes[i][j][0].to_dist_tensor().unsqueeze(0))) ** 2
                Js.append(mse)

        J = torch.mean(torch.tensor(Js))

        J.backward()
        optimizer.step()
    return model


def disturbance_prob(detected: bool, sp: SimParams) -> float:
    return sp.tru_det_p if detected else (1.0 - sp.tru_det_p)


def policy_prob(detection_act: bool, state: SimState, val_func: Callable, sp: SimParams) -> float:
    v_s = val_func(state.to_dist_tensor())
    p_x = sp.tru_det_p

    next_state = step_state(detection_act, state, sp)
    v_s_next = val_func(next_state)

    action_prob = (p_x * v_s_next) / v_s

    return action_prob


def step_state(detection_act: bool, state: SimState, sp: SimParams) -> SimState:
    if state.stopped or (detection_act and state.distance <= sp.braking_dist):
        return SimState(state.distance, True)

    return SimState(state.distance - 1, False)


def car_rollout(value_function: Callable, sp: SimParams) -> SimTraj:
    current_dist = sp.total_dist
    stopped = False

    sim_traj = []

    for t in range(sp.total_dist):
        r = np.random.random()

        # Do the policy calculation stuff here
        current_state = SimState(current_dist, stopped)
        next_s_if_tru = step_state(True, current_state, sp)
        v_s_next = value_function(next_s_if_tru.to_dist_tensor().unsqueeze(0))
        v_s = value_function(current_state.to_dist_tensor().unsqueeze(0))
        threshold = (sp.tru_det_p * v_s_next) / v_s
        # threshold = det_func(current_dist)
        detected = r < threshold

        sim_traj.append((current_state, detected))

        if not stopped:
            if detected and current_dist <= sp.braking_dist:
                stopped = True
            else:
                current_dist -= 1

    return sim_traj


if __name__ == "__main__":
    mc_policy_eval(10, 10, 0.1)
