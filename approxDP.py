import dataclasses
import time
from dataclasses import dataclass
from typing import Callable, Tuple, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR


class DPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff_1 = nn.Linear(2, 20)
        self.ff_2 = nn.Linear(20, 20)
        self.ff_3 = nn.Linear(20, 1)
        self.ls = nn.LogSigmoid()

    def forward(self, x):
        x = F.relu(self.ff_1(x))
        x = F.relu(self.ff_2(x))
        x = self.ls(self.ff_3(x))
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

    # def to_dist_tensor(self):
    #     return torch.tensor(self.distance, dtype=torch.float)

    def to_tensor(self, use_cuda=True):
        if use_cuda:
            return torch.tensor([self.distance, self.stopped], dtype=torch.float).cuda()

        return torch.tensor([self.distance, self.stopped], dtype=torch.float)


SimTraj = List[Tuple[SimState, bool]]


def mc_policy_eval(n_iter, n_samples):
    sp = SimParams(100, 50, 30, 0.9)
    cuda = torch.device('cuda')
    has_been_filled = False
    fig, axs = plt.subplots(1, 2)

    # TODO: Setup initial params so that there are actually some failures!
    model = DPModel().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    for iteration in range(n_iter):
        with torch.no_grad():
            if iteration == 0:
                crap_car = dataclasses.replace(sp, tru_det_p=0.01)
                episodes: List[SimTraj] = [car_rollout(model, crap_car) for _ in range(n_samples)]
            else:
                episodes: List[SimTraj] = [car_rollout(model, sp) for _ in range(n_samples)]
            unsafe_count = len([ep for ep in episodes if ep[-1][0].distance < sp.safe_dist])

            episode_gs = []

            for ep in episodes:

                # Failure state
                if ep[-1][0].distance < sp.safe_dist:
                    state_probs = torch.tensor([disturbance_prob(s[1], sp) for s in ep])
                    policy_probs = torch.tensor([policy_prob(s[1], s[0], model, sp) for s in ep])
                    ratios = state_probs / policy_probs

                    # In the failure state, the probability of failure is certain
                    ratios[-1] = 1.0

                    g_estimates = torch.cumprod(ratios.flip([0]), 0).flip([0])
                else:
                    g_estimates = torch.zeros(len(ep))

                episode_gs.append(g_estimates)

        episode_tensors = torch.concat([torch.stack([s[0].to_tensor() for s in ep]) for ep in episodes]).cuda()
        gs = torch.concat(episode_gs).cuda().view(-1, 1)

        solve_start = time.time()
        print(f"Num Unsafe: {unsafe_count}")
        for i in range(10000):
            optimizer.zero_grad()
            vs = model(episode_tensors).exp()
            J = F.mse_loss(vs, gs)
            J.backward()
            optimizer.step()
        if torch.isnan(J):
            print("Nan!")
        # print("Learning Rate: ", lr_scheduler.get_last_lr())
        print(f"Loss: {J.item()}")
        print(f"Solve Time: {time.time() - solve_start}")

        # p_100 = policy_prob(True, SimState(100, False), model, sp)
        # p_49 = policy_prob(True, SimState(49, False), model, sp)

        # with torch.no_grad():
        #     ds = torch.arange(sp.safe_dist - 1, sp.total_dist + 1, dtype=torch.float)
        #     go_states = torch.column_stack([ds, torch.full_like(ds, 0)])
        #     vs_go = model(go_states).exp()
        #     axs[0].plot(ds.detach().numpy(), vs_go.detach().numpy(), color="blue", alpha=(iteration / n_iter))
        #
        #     stop_states = torch.column_stack([ds, torch.full_like(ds, 1)])
        #     vs_stop = model(stop_states).exp()
        #     axs[0].plot(ds.detach().numpy(), vs_stop.detach().numpy(), color="red", alpha=(iteration / n_iter))

    with torch.no_grad():
        ds = torch.arange(sp.safe_dist - 1, sp.total_dist + 1, dtype=torch.float).cuda()
        go_states = torch.column_stack([ds, torch.full_like(ds, 0)]).cuda()
        vs_go = model(go_states).exp()
        axs[0].plot(ds.cpu().numpy(), vs_go.cpu().numpy(), color="blue",
                    label="No Brakes")  # , alpha=(iteration / n_iter))
        axs[0].set_xlabel("Distance")
        axs[0].set_ylabel("Value Function")

        stop_states = torch.column_stack([ds, torch.full_like(ds, 1)]).cuda()
        vs_stop = model(stop_states).exp()
        axs[0].plot(ds.cpu().numpy(), vs_stop.cpu().numpy(), color="red",
                    label="Brakes")  # , alpha=(iteration / n_iter))
        axs[0].legend()

        # I want to plot the policy decisions at each (go) state. Probability of detection?
        axs[1].plot(ds.cpu().numpy(),
                    torch.tensor([policy_prob(True, SimState(d, False), model, sp) for d in ds]).cpu().numpy())
        axs[1].set_xlabel("Distance")
        axs[1].set_ylabel("\u03c0(det | distance)")
        fig.suptitle(f"Total Dist: {sp.total_dist}, Braking Dist: {sp.braking_dist}, Safe Dist: {sp.safe_dist}")
        plt.show()

    return model


def disturbance_prob(detected: bool, sp: SimParams) -> float:
    return sp.tru_det_p if detected else (1.0 - sp.tru_det_p)



def q_policy_prob(detection_act: bool, state: SimState, log_q_func: Callable, sp: SimParams) -> float:
    p_a = torch.tensor(sp.tru_det_p if detection_act else (1.0 - sp.tru_det_p))
    next_s_a = step_state(detection_act, state, sp)
    next_s_not_a = step_state(not detection_act, state, sp)

    log_q_s_a = log_q_func(state.to_tensor().unsqueeze(0), detection_act)
    log_q_s_na = log_q_func(state.to_tensor().unsqueeze(0), detection_act)

    log_ap = p_a.log() + log_v_s_a
    log_not_ap = (1.0 - p_a).log() + log_v_s_not_a

    log_norm = torch.logsumexp(torch.tensor([log_ap, log_not_ap]), 0)

    action_prob = (log_ap - log_norm).exp()
    return action_prob



def policy_prob(detection_act: bool, state: SimState, log_val_func: Callable, sp: SimParams) -> float:
    p_a = torch.tensor(sp.tru_det_p if detection_act else (1.0 - sp.tru_det_p))

    next_s_a = step_state(detection_act, state, sp)
    next_s_not_a = step_state(not detection_act, state, sp)

    log_v_s_a = log_val_func(next_s_a.to_tensor().unsqueeze(0))
    log_v_s_not_a = log_val_func(next_s_not_a.to_tensor().unsqueeze(0))

    log_ap = p_a.log() + log_v_s_a
    log_not_ap = (1.0 - p_a).log() + log_v_s_not_a

    log_norm = torch.logsumexp(torch.tensor([log_ap, log_not_ap]), 0)

    action_prob = (log_ap - log_norm).exp()
    return action_prob

    # if torch.is_nonzero(normalization):
    #     action_prob = ap_numerator / normalization
    #     return action_prob
    # else:
    #     return torch.full_like(ap_numerator, p_a)


def step_state(detection_act: bool, state: SimState, sp: SimParams) -> SimState:
    if state.stopped or (detection_act and state.distance <= sp.braking_dist):
        return SimState(state.distance, True)

    return SimState(state.distance - 1, False)


def car_rollout(value_function: Callable, sp: SimParams) -> SimTraj:
    sim_traj = []
    current_state = SimState(sp.total_dist, False)

    for t in range(sp.total_dist):
        r = np.random.random()

        # Do the policy calculation stuff here
        if current_state.distance == sp.braking_dist:
            x = 1

        threshold = policy_prob(True, current_state, value_function, sp)
        detected = r < threshold

        sim_traj.append((current_state, detected))

        # Failure states are terminal states
        if current_state.distance < sp.safe_dist:
            return sim_traj

        # Next state
        current_state = step_state(detected, current_state, sp)

    return sim_traj


"""
def iterative_policy_evaluation(max_its : int, sp: SimParams):
    vs = [0.0 for s in S]
    for i in range(max_its):
        vs = [lookahead() for s in S]
    return 
"""

if __name__ == "__main__":
    mc_policy_eval(10, 100)
