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


class DQN(nn.Module):
    def __init__(self, state_dims, num_actions, norm_tensor=None):
        super().__init__()
        self.ff_1 = nn.Linear(state_dims, 20)
        self.ff_2 = nn.Linear(20, 20)
        self.ff_3 = nn.Linear(20, num_actions)
        self.ls = nn.LogSigmoid()
        self.norm_tensor = norm_tensor

    def forward(self, x):
        if self.norm_tensor is not None:
            x = x / self.norm_tensor
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


# def mc_policy_eval(n_iter, n_samples):
#     sp = SimParams(100, 50, 30, 0.9)
#     cuda = torch.device('cuda')
#     fig, axs = plt.subplots(1, 2)
#
#     # TODO: Setup initial params so that there are actually some failures!
#     model = DPModel().cuda()
#     optimizer = torch.optim.Adam(model.parameters())
#     for iteration in range(n_iter):
#         with torch.no_grad():
#             if iteration == 0:
#                 crap_car = dataclasses.replace(sp, tru_det_p=0.01)
#                 episodes: List[SimTraj] = [car_rollout(model, crap_car) for _ in range(n_samples)]
#             else:
#                 episodes: List[SimTraj] = [car_rollout(model, sp) for _ in range(n_samples)]
#             unsafe_count = len([ep for ep in episodes if ep[-1][0].distance < sp.safe_dist])
#
#             episode_gs = []
#
#             for ep in episodes:
#
#                 # Failure state
#                 if ep[-1][0].distance < sp.safe_dist:
#                     state_probs = torch.tensor([disturbance_prob(s[1], sp) for s in ep])
#                     policy_probs = torch.tensor([policy_prob(s[1], s[0], model, sp) for s in ep])
#                     ratios = state_probs / policy_probs
#
#                     # In the failure state, the probability of failure is certain
#                     ratios[-1] = 1.0
#
#                     g_estimates = torch.cumprod(ratios.flip([0]), 0).flip([0])
#                 else:
#                     g_estimates = torch.zeros(len(ep))
#
#                 episode_gs.append(g_estimates)
#
#         episode_tensors = torch.concat([torch.stack([s[0].to_tensor() for s in ep]) for ep in episodes]).cuda()
#         gs = torch.concat(episode_gs).cuda().view(-1, 1)
#
#         solve_start = time.time()
#         print(f"Num Unsafe: {unsafe_count}")
#         for i in range(10000):
#             optimizer.zero_grad()
#             vs = model(episode_tensors).exp()
#             J = F.mse_loss(vs, gs)
#             J.backward()
#             optimizer.step()
#         if torch.isnan(J):
#             print("Nan!")
#         # print("Learning Rate: ", lr_scheduler.get_last_lr())
#         print(f"Loss: {J.item()}")
#         print(f"Solve Time: {time.time() - solve_start}")
#
#     with torch.no_grad():
#         ds = torch.arange(sp.safe_dist - 1, sp.total_dist + 1, dtype=torch.float).cuda()
#         go_states = torch.column_stack([ds, torch.full_like(ds, 0)]).cuda()
#         vs_go = model(go_states).exp()
#         axs[0].plot(ds.cpu().numpy(), vs_go.cpu().numpy(), color="blue",
#                     label="No Brakes")  # , alpha=(iteration / n_iter))
#         axs[0].set_xlabel("Distance")
#         axs[0].set_ylabel("Value Function")
#
#         stop_states = torch.column_stack([ds, torch.full_like(ds, 1)]).cuda()
#         vs_stop = model(stop_states).exp()
#         axs[0].plot(ds.cpu().numpy(), vs_stop.cpu().numpy(), color="red",
#                     label="Brakes")  # , alpha=(iteration / n_iter))
#         axs[0].legend()
#
#         # I want to plot the policy decisions at each (go) state. Probability of detection?
#         axs[1].plot(ds.cpu().numpy(),
#                     torch.tensor([policy_prob(True, SimState(d, False), model, sp) for d in ds]).cpu().numpy())
#         axs[1].set_xlabel("Distance")
#         axs[1].set_ylabel("\u03c0(det | distance)")
#         fig.suptitle(f"Total Dist: {sp.total_dist}, Braking Dist: {sp.braking_dist}, Safe Dist: {sp.safe_dist}")
#         plt.show()
#
#     return model

def simple_crap_prob(const_det_prob: float, detection_act: bool, s: SimState, sp: SimParams) -> float:
    # if s.distance <= sp.braking_dist:
    return const_det_prob if detection_act else (1.0 - const_det_prob)

    # return sp.tru_det_p if detection_act else (1.0 - sp.tru_det_p)


def q_learning(n_iter, n_samples):
    sp = SimParams(10, 5, 3, 0.9)
    fig, axs = plt.subplots(1, 2)
    bar_fig, bar_axs = plt.subplots()

    model = DQN(2, 2, torch.tensor([sp.total_dist, 1.0], device=torch.device("cuda"))).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for iteration in range(n_iter):
        with torch.no_grad():
            # if iteration == 0:
            #     crap_car = dataclasses.replace(sp, tru_det_p=0.01)
            #     episodes: List[SimTraj] = [car_rollout_q(model, crap_car) for _ in range(n_samples)]
            # else:

            crap_policy_f = lambda a, s, sp: simple_crap_prob(0.05, a, s, sp)

            episodes: List[SimTraj] = [car_rollout_q(crap_policy_f, sp) for _ in range(n_samples)]
            episode_gs = [weighted_cum_q_return(ep, crap_policy_f, sp) for ep in episodes]

        # TODO: Just have the data-oriented approach from the start, with simulation trajectories grouped as tensors?
        # The terminal state is meaningless for the q-function, as the final action is never actually taken
        episode_tensors = torch.concat([torch.stack([s[0].to_tensor() for s in ep[:-1]]) for ep in episodes]).cuda()
        action_tensors = torch.concat(
            [torch.tensor([s[1] for s in ep[:-1]], dtype=torch.long) for ep in episodes]).cuda().view(-1, 1)
        gs = torch.concat(episode_gs).cuda().view(-1, 1)

        assert action_tensors.size() == gs.size()
        assert len(episode_tensors) == len(gs)

        solve_start = time.time()
        print(f"Num Unsafe: {len([ep for ep in episodes if ep[-1][0].distance < sp.safe_dist])}")
        for i in range(10000):
            optimizer.zero_grad()
            qs = model(episode_tensors).exp().gather(-1, action_tensors)

            assert qs.size() == gs.size()

            J = F.mse_loss(qs, gs)
            J.backward()
            optimizer.step()

            if torch.isnan(J):
                print("Nan!")

        print(f"Loss: {J.item()}")
        print(f"Solve Time: {time.time() - solve_start}")

    with torch.no_grad():
        ds = torch.arange(sp.safe_dist - 1, sp.total_dist + 1, dtype=torch.float).cuda()
        go_states = torch.column_stack([ds, torch.full_like(ds, 0)]).cuda()
        qs_go = model(go_states).exp()
        axs[0].plot(ds.cpu().numpy(), qs_go[:, 0].cpu().numpy(), color="blue",
                    label="Mis-detect")  # , alpha=((1.0 + iteration) / n_iter))
        axs[0].set_xlabel("Distance")
        axs[0].set_ylabel("Q(Action | State)")

        # stop_states = torch.column_stack([ds, torch.full_like(ds, 1)]).cuda()
        # qs_stop = model(stop_states).exp()
        axs[0].plot(ds.cpu().numpy(), qs_go[:, 1].cpu().numpy(), color="red",
                    label="Detect")  # alpha=((1.0 + iteration) / n_iter))

        # I want to plot the policy decisions at each (go) state. Probability of detection?
        axs[1].plot(ds.cpu().numpy(),
                    torch.tensor([q_policy_prob(True, SimState(d, False), model, sp) for d in ds]).cpu().numpy(),
                    color="green")  # alpha=((1.0 + iteration) / n_iter))
        axs[1].set_xlabel("Distance")
        axs[1].set_ylabel("\u03c0(det | distance)")
        fig.suptitle(f"Total Dist: {sp.total_dist}, Braking Dist: {sp.braking_dist}, Safe Dist: {sp.safe_dist}")

        det_dist_counts = torch.zeros_like(ds.cpu())
        ndet_dist_counts = torch.zeros_like(ds.cpu())
        dlabels = [str(int(d.item())) for d in ds]

        for s, a in zip(episode_tensors, action_tensors):
            if s[1] == 1:
                continue

            # if s[0] <= sp.braking_dist and a == 1:
            #     print("Found one!")

            d_idx = int(s[0] - ds[0])

            if a == 0:
                ndet_dist_counts[d_idx] += 1
            else:
                det_dist_counts[d_idx] += 1

        bar_axs.bar(ds.cpu(), det_dist_counts, 0.35, label='Detect | Dist, brake')
        bar_axs.bar(ds.cpu(), ndet_dist_counts, 0.35, bottom=det_dist_counts, label='¬Detect | Dist, no-brake')
        bar_axs.legend()
    plt.show()

    torch.tensor([q_policy_prob(True, SimState(d, False), model, sp) for d in ds])

    return model


def tabular_q_learning(n_iter, n_samples):
    sp = SimParams(10, 5, 3, 0.9)

    # Initialize for all s, a
    # Q[dist, brakes, detection]
    # Q = torch.zeros((10, 2, 2))
    WG = torch.zeros((10, 2, 2))
    C = torch.zeros((10, 2, 2))
    # C(s, a) <- 0

    crap_policy_f = lambda a, s, sp: simple_crap_prob(0.1, a, s, sp)
    episodes = [car_rollout_q(crap_policy_f, sp) for _ in range(n_samples)]

    for ep in episodes:
        # if ep[-1][0].distance >= sp.safe_dist:
        #     continue
        # weights = retrace_cum_return(ep, crap_policy_f, sp, 1.0)
        # weights = weighted_cum_q_return(ep, crap_policy_f, sp)
        weights = importance_weights(ep, crap_policy_f, sp)

        for t in reversed(range(len(ep) - 1)):
            d_idx = ep[t][0].distance - 1
            b_idx = 1 if ep[t][0].stopped else 0
            a_idx = 1 if ep[t][1] else 0

            C[d_idx, b_idx, a_idx] += 1

            G = 1.0 if ep[-1][0].distance < sp.safe_dist else 0.0

            WG[d_idx, b_idx, a_idx] += G * weights[t]

        Q = WG / C

    print("Done")
    print(Q[2, 0, :])
    print(Q[3, 0, :])
    print(Q[8, 0, :])
    print(Q[9, 0, :])

    ds = np.arange(sp.safe_dist, sp.total_dist + 1)
    plt.plot(ds, Q[ds - 1, 0, 0], label="Not Detect")
    plt.plot(ds, Q[ds - 1, 0, 1], label="Detect")
    plt.legend()
    plt.show()


def disturbance_prob(detected: bool, sp: SimParams) -> float:
    return sp.tru_det_p if detected else (1.0 - sp.tru_det_p)


def q_policy_prob(detection_act: bool, state: SimState, log_q_func: nn.Module, sp: SimParams) -> float:
    log_pas = torch.tensor([1.0 - sp.tru_det_p, sp.tru_det_p], device=torch.device("cuda")).log()
    log_qs = log_q_func(state.to_tensor().unsqueeze(0))[0]

    log_aps = log_pas + log_qs
    log_norm = torch.logsumexp(log_aps, 0)

    action_probs = (log_aps - log_norm).exp()
    action_choice = 1 if detection_act else 0
    return action_probs[action_choice]


SS_Policy = Callable[[bool, SimState, SimParams], float]


def retrace_cum_return(ep: SimTraj, policy_prob_func: SS_Policy, sp: SimParams, ratio_ceiling: float) -> torch.tensor:
    assert ratio_ceiling > 0.0

    if ep[-1][0].distance >= sp.safe_dist:
        return torch.zeros(len(ep) - 1)
    state_probs = torch.tensor([disturbance_prob(ep[t][1], sp) for t in range(1, len(ep) - 1)])
    policy_probs = torch.tensor([policy_prob_func(ep[t][1], ep[t][0], sp) for t in range(1, len(ep) - 1)])

    ratios = torch.ones(len(ep) - 1)
    ratios[:-1] = state_probs / policy_probs
    ratios[:-1] = ratios[:-1].clamp(0, ratio_ceiling)
    g_estimates = torch.cumprod(ratios.flip([0]), 0).flip([0])

    assert len(g_estimates) == len(ep) - 1
    assert g_estimates[-1] == 1.0

    return g_estimates


def importance_weights(ep: SimTraj, policy_prob_func: SS_Policy, sp: SimParams) -> torch.tensor:
    state_probs = torch.tensor([disturbance_prob(ep[t][1], sp) for t in range(1, len(ep) - 1)])

    policy_probs = torch.tensor([policy_prob_func(ep[t][1], ep[t][0], sp) for t in range(1, len(ep) - 1)])

    ratios = torch.ones(len(ep) - 1)
    ratios[:-1] = state_probs / policy_probs

    g_estimates = torch.cumprod(ratios.flip([0]), 0).flip([0])

    assert len(g_estimates) == len(ep) - 1
    assert g_estimates[-1] == 1.0

    return g_estimates


def weighted_cum_q_return(ep: SimTraj, policy_prob_func: SS_Policy, sp: SimParams) -> torch.tensor:
    # Not a Failure State
    if ep[-1][0].distance >= sp.safe_dist:
        return torch.zeros(len(ep) - 1)

    state_probs = torch.tensor([disturbance_prob(ep[t][1], sp) for t in range(1, len(ep) - 1)])

    policy_probs = torch.tensor([policy_prob_func(ep[t][1], ep[t][0], sp) for t in range(1, len(ep) - 1)])
    # policy_probs = torch.tensor([q_policy_prob(ep[t][1], ep[t][0], log_q_func, sp) for t in range(1, len(ep) - 1)])

    # The probability of failure by taking failing action from penultimate state is certain
    ratios = torch.ones(len(ep) - 1)
    ratios[:-1] = state_probs / policy_probs

    g_estimates = torch.cumprod(ratios.flip([0]), 0).flip([0])

    assert len(g_estimates) == len(ep) - 1
    assert g_estimates[-1] == 1.0

    return g_estimates


# def policy_prob(detection_act: bool, state: SimState, log_val_func: Callable, sp: SimParams) -> float:
#     p_a = torch.tensor(sp.tru_det_p if detection_act else (1.0 - sp.tru_det_p))
#
#     next_s_a = step_state(detection_act, state, sp)
#     next_s_not_a = step_state(not detection_act, state, sp)
#
#     log_v_s_a = log_val_func(next_s_a.to_tensor().unsqueeze(0))
#     log_v_s_not_a = log_val_func(next_s_not_a.to_tensor().unsqueeze(0))
#
#     log_ap = p_a.log() + log_v_s_a
#     log_not_ap = (1.0 - p_a).log() + log_v_s_not_a
#
#     log_norm = torch.logsumexp(torch.tensor([log_ap, log_not_ap]), 0)
#
#     action_prob = (log_ap - log_norm).exp()
#     return action_prob
#
# if torch.is_nonzero(normalization):
#     action_prob = ap_numerator / normalization
#     return action_prob
# else:
#     return torch.full_like(ap_numerator, p_a)


def step_state(detection_act: bool, state: SimState, sp: SimParams) -> SimState:
    if state.stopped or (detection_act and state.distance <= sp.braking_dist):
        return SimState(state.distance, True)

    return SimState(state.distance - 1, False)


# def car_rollout(value_function: Callable, sp: SimParams) -> SimTraj:
#     sim_traj = []
#     current_state = SimState(sp.total_dist, False)
#
#     for t in range(sp.total_dist):
#         r = np.random.random()
#
#         # Do the policy calculation stuff here
#         if current_state.distance == sp.braking_dist:
#             x = 1
#
#         threshold = policy_prob(True, current_state, value_function, sp)
#         detected = r < threshold
#
#         sim_traj.append((current_state, detected))
#
#         # Failure states are terminal states
#         if current_state.distance < sp.safe_dist:
#             return sim_traj
#
#         # Next state
#         current_state = step_state(detected, current_state, sp)
#
#     return sim_traj


def car_rollout_q(policy_prob_f: Callable[[bool, SimState, SimParams], float], sp: SimParams) -> SimTraj:
    sim_traj = []
    current_state = SimState(sp.total_dist, False)

    for t in range(sp.total_dist):
        r = np.random.random()

        # threshold = q_policy_prob(True, current_state, log_q_func, sp)
        threshold = policy_prob_f(True, current_state, sp)
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
    # mc_policy_eval(10, 100)
    tabular_q_learning(5, 10000)
    # q_learning(5, 1000)
