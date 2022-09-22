import os
import os.path
import json
import time
from dataclasses import dataclass

import dacite
from typing import Callable, List, Any, Tuple, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import torch
from torch import nn, FloatTensor, BoolTensor
import torch.nn.functional as F

import stl
from adaptiveImportanceSampler import FFPolicy, get_quantile, cross_entropy_train, get_ep_strides
from carlaUtils import SimSnapshot, to_salient_var, norm_salient_input
from crossEntExperiment import log1mexp
from pems import load_model_det, PEMClass_Deterministic, save_model_det


def pre_train_cem(model: nn.Module, pre_s_tensors: FloatTensor, pre_a_tensors: torch.BoolTensor,
                  n_eps: int) -> nn.Module:
    pre_optim = torch.optim.Adam(model.parameters())

    model.train()
    for epoch in range(1000):
        pre_lsfms = model(pre_s_tensors).view(-1, 2)

        pre_log_qs = pre_lsfms[:, 0] * ~pre_a_tensors + pre_lsfms[:, 1] * pre_a_tensors
        pre_loss = -pre_log_qs.sum() / n_eps

        pre_optim.zero_grad()
        pre_loss.backward()
        pre_optim.step()
        if epoch % 100 == 0:
            print(f"Pre E: {epoch}, Loss: {pre_loss}")

    model.eval()
    return model


def tensors_from_rollouts(rollouts: List[List[SimSnapshot]]) -> Tuple[FloatTensor, BoolTensor]:
    # Ignore final action as it has no effect on reward / subsequent state
    s_tensors = torch.concat(
        [torch.tensor([s.outs.true_distance for s in r[:-1]], device="cuda") for r in rollouts]).view(-1, 1)
    a_tensors = torch.concat(
        [torch.tensor([s.outs.model_det for s in r[:-1]], device="cuda") for r in rollouts])
    return s_tensors, a_tensors


# def episodic_tensors_from_rollouts(rollouts: List[List[SimSnapshot]]) -> Tuple[List[FloatTensor], List[BoolTensor]]:
#     # Ignore final action as it has no effect on reward / subsequent state
#     s_tensors = [torch.tensor([s.outs.true_distance for s in r[:-1]], device="cuda") for r in rollouts]
#     a_tensors = [torch.tensor([s.outs.model_det for s in r[:-1]], device="cuda") for r in rollouts]
#     return s_tensors, a_tensors


def chart_det_probs(model: nn.Module):
    fig, ax = plt.subplots(figsize=(3, 3))
    with torch.no_grad():
        pre_dists = torch.linspace(0, 13, 100, device="cuda").view(-1, 1)
        pre_probs = model(pre_dists)[:, 1].exp()
        ax.plot(pre_dists.detach().cpu(), pre_probs.detach().cpu())
        # plt.ylim([0, 1])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Detection Probability")
        ax.set_xlabel("Distance (Metres)")
        plt.tight_layout()
        plt.show()


def chart_multistage_probs(cem_folder: str, color: str = 'b'):
    fig, ax = plt.subplots(figsize=(3, 3))
    cem_paths = sorted([os.path.join(cem_folder, cp) for cp in os.listdir(cem_folder)])
    print(cem_paths)
    cem_models = [load_model_det(FFPolicy(1, torch.tensor([12.0], device="cuda")).cuda(), cem_path) for cem_path in
                  cem_paths]

    with torch.no_grad():
        pre_dists = torch.linspace(0, 13, 100, device="cuda").view(-1, 1)
        for i, model in enumerate(cem_models):
            pre_probs = model(pre_dists)[:, 1].exp()
            ax.plot(pre_dists.detach().cpu(), pre_probs.detach().cpu(), color=color,
                    alpha=np.sqrt((1.0 + i) / (len(cem_models) + 1.0)))

    ax.set_ylim(0, 1)
    ax.set_ylabel("Detection Probability")
    ax.set_xlabel("Distance (Metres)")
    plt.tight_layout()
    plt.show()


def chart_multiple_multistage(cem_folders: List[str]):
    cem_models = []
    for cem_folder in cem_folders:
        cem_paths = sorted([os.path.join(cem_folder, cp) for cp in os.listdir(cem_folder)])
        cem_models.append(
            [load_model_det(FFPolicy(1, torch.tensor([12.0], device="cuda")).cuda(), cem_path) for cem_path in
             cem_paths])

    pre_dists = torch.linspace(0, 13, 100, device="cuda").view(-1, 1)

    for i, mid_models in enumerate(zip(*cem_models)):
        fig, ax = plt.subplots(figsize=(3, 3))

        for cem_name, model in zip(cem_folders, mid_models):
            pre_probs = model(pre_dists)[:, 1].exp()
            name_map = {"AGM": "$r_a$", "Classic": "$r_c$", "Smooth-Cumulative": "$r_s$"}
            ax.plot(pre_dists.detach().cpu(), pre_probs.detach().cpu(), label=name_map[os.path.basename(cem_name).split("_")[1]])
            ax.legend(loc="best")
            ax.set_title(f"$\kappa = {i + 1}$")

        ax.set_ylim(0, 1)
        ax.set_ylabel("Detection Probability")
        ax.set_xlabel("Distance (Metres)")
        plt.tight_layout()
        plt.show()


def load_rollouts(folder_path: str, subset_amount: Optional[int] = None) -> List[List[SimSnapshot]]:
    ep_fns = os.listdir(folder_path)
    if subset_amount is not None:
        ep_fns = ep_fns[:subset_amount]

    print(ep_fns)
    ep_rollouts = []
    for ep_fn in ep_fns:
        with open(os.path.join(folder_path, ep_fn), 'r') as f:
            ep_rollouts.append([dacite.from_dict(data_class=SimSnapshot, data=s) for s in json.load(f)])
    return ep_rollouts


def one_step_cem(ep_rollouts: List[List[SimSnapshot]], cem_model: nn.Module, pem_model: nn.Module, norm_stats,
                 safety_func,
                 chart: bool, model_save_path: Optional[str] = None) -> nn.Module:
    s_tensors, a_tensors = tensors_from_rollouts(ep_rollouts)

    safety_vals = [safety_func(rollout) for rollout in ep_rollouts]
    failure_thresh = get_quantile(safety_vals, 0.95, 0.0)

    num_fails = len([s for s in safety_vals if s <= failure_thresh])
    print(f"Failure Thresh: {failure_thresh} ({num_fails} / {len(ep_rollouts)} episodes)")

    n_func = lambda s_inputs, norm_dims: norm_salient_input(s_inputs, norm_stats[0], norm_stats[1], norm_dims)

    state_pem_ins = torch.stack(
        [to_salient_var(s.model_ins, n_func) for rollout in ep_rollouts for s in rollout[:-1]]).to(device="cuda")

    with torch.no_grad():
        pem_logits = pem_model(state_pem_ins).view(-1)
        log_tru_ps = F.logsigmoid(pem_logits)
        log_neg_ps = log1mexp(log_tru_ps)
        log_ps = (a_tensors * log_tru_ps) + (~a_tensors * log_neg_ps)
        ep_strides = get_ep_strides(ep_rollouts)
        fail_indicator = torch.concat([torch.full((len(sr) - 1,), sv <= failure_thresh, device="cuda") for sr, sv in
                                       zip(ep_rollouts, safety_vals)])

    cross_entropy_train(cem_model, s_tensors, a_tensors, ep_strides, log_ps, fail_indicator, 1000, len(ep_rollouts))

    if model_save_path is not None:
        save_model_det(cem_model, model_save_path)

    if chart:
        chart_det_probs(cem_model)

    return cem_model


def cem_run(sim_data_folder: str, model_load_path: str):
    ep_rollouts = load_rollouts(sim_data_folder)

    # Single variable of distance right now? But could also add velocity and accelerations of both vehicles (this is just to check that *anything* works...)
    cem_model = FFPolicy(1, torch.tensor([12.0], device="cuda")).cuda()
    cem_model = load_model_det(cem_model, model_load_path)

    pem_class = load_model_det(PEMClass_Deterministic(14, 1), "models/det_baseline_full/pem_class_train_full").cuda()
    norm_stats = torch.load("models/norm_stats_mu.pt"), torch.load("models/norm_stats_std.pt")

    # n_func = lambda s_inputs, norm_dims: norm_salient_input(s_inputs, norm_stats[0], norm_stats[1], norm_dims)

    # rollout_weights(pem_class, n_func, cem_model, ep_rollouts)

    # Pre- Training
    # print("Pre-training")
    # pre_train_cem(cem_model, s_tensors, a_tensors, len(ep_rollouts))
    # save_model_det(cem_model, "models/CEMs/pretrain_e100_PEM.pyt")
    # chart_det_probs(cem_model)

    cem_model = one_step_cem(ep_rollouts, cem_model, pem_class, norm_stats, True)


def rollout_weights(pem: nn.Module, n_func, proposal: nn.Module, rollouts: List[List[SimSnapshot]]) -> FloatTensor:
    s_tensors, a_tensors = tensors_from_rollouts(rollouts)
    state_pem_ins = torch.stack([to_salient_var(s.model_ins, n_func) for rollout in rollouts for s in rollout[:-1]]).to(
        device="cuda")
    ep_strides = get_ep_strides(rollouts)

    pem_logits = pem(state_pem_ins).view(-1)
    log_tru_ps = F.logsigmoid(pem_logits)
    log_neg_ps = log1mexp(log_tru_ps)
    log_ps = (a_tensors * log_tru_ps) + (~a_tensors * log_neg_ps)

    log_q_sfms = proposal(s_tensors).view(-1, 2)
    log_qs = (~a_tensors * log_q_sfms[:, 0]) + (a_tensors * log_q_sfms[:, 1])

    log_ratios = log_ps - log_qs
    strided_ratios = log_ratios.tensor_split(ep_strides)
    log_weights = torch.stack([rs.sum(0) for rs in strided_ratios])
    # log_weights = torch.concat([rs.sum(0).expand(len(rs)) for rs in strided_ratios])
    weights = log_weights.exp()
    return weights


def pem_loglikelihoods(pem: nn.Module, n_func, rollouts: List[List[SimSnapshot]]) -> FloatTensor:
    s_tensors, a_tensors = tensors_from_rollouts(rollouts)
    state_pem_ins = torch.stack([to_salient_var(s.model_ins, n_func) for rollout in rollouts for s in rollout[:-1]]).to(
        device="cuda")
    ep_strides = get_ep_strides(rollouts)

    pem_logits = pem(state_pem_ins).view(-1)
    log_tru_ps = F.logsigmoid(pem_logits)
    log_neg_ps = log1mexp(log_tru_ps)
    log_ps = (a_tensors * log_tru_ps) + (~a_tensors * log_neg_ps)

    strided_lps = log_ps.tensor_split(ep_strides)
    ep_log_ps = torch.stack([lps.sum(0) for lps in strided_lps])
    return ep_log_ps


def avg_fail_nll(nlls: FloatTensor, safety_func, fail_thresh: float, rollouts: List[List[SimSnapshot]]) -> float:
    safety_vals = torch.tensor([safety_func(rollout) for rollout in rollouts], device=nlls.device)
    fail_indicator = safety_vals < fail_thresh
    fail_nlls = nlls[fail_indicator]
    return fail_nlls.mean()


def rollout_weights_mixed(pem: nn.Module, n_func, dummy_prob: float, rollouts: List[List[SimSnapshot]]) -> FloatTensor:
    s_tensors, a_tensors = tensors_from_rollouts(rollouts)
    state_pem_ins = torch.stack([to_salient_var(s.model_ins, n_func) for rollout in rollouts for s in rollout[:-1]]).to(
        device="cuda")
    ep_strides = get_ep_strides(rollouts)

    pem_logits = pem(state_pem_ins).view(-1)
    log_tru_ps = F.logsigmoid(pem_logits)
    log_neg_ps = log1mexp(log_tru_ps)
    log_ps = (a_tensors * log_tru_ps) + (~a_tensors * log_neg_ps)

    log_dummy_tru = torch.full(log_ps.shape, dummy_prob, device=log_ps.device).log()
    log_dummy_neg = torch.full(log_ps.shape, 1.0 - dummy_prob, device=log_ps.device).log()
    log_dummy_qs = (a_tensors * log_dummy_tru) + (~a_tensors * log_dummy_neg)

    dummy_condition = (s_tensors < 10.0).view(-1)
    # Places which do not meet condition use the pem to sample, so their log-ratios are zero'd out (log 1.0 == 0.0)
    log_ratios = dummy_condition * (log_ps - log_dummy_qs)

    strided_ratios = log_ratios.tensor_split(ep_strides)
    log_weights = torch.stack([rs.sum(0) for rs in strided_ratios])
    weights = log_weights.exp()
    return weights


def rollout_weights_dummy(pem: nn.Module, n_func, dummy_prob: float, rollouts: List[List[SimSnapshot]]) -> FloatTensor:
    s_tensors, a_tensors = tensors_from_rollouts(rollouts)
    state_pem_ins = torch.stack([to_salient_var(s.model_ins, n_func) for rollout in rollouts for s in rollout[:-1]]).to(
        device="cuda")
    ep_strides = get_ep_strides(rollouts)

    pem_logits = pem(state_pem_ins).view(-1)
    log_tru_ps = F.logsigmoid(pem_logits)
    log_neg_ps = log1mexp(log_tru_ps)
    log_ps = (a_tensors * log_tru_ps) + (~a_tensors * log_neg_ps)

    log_dummy_tru = torch.full(log_ps.shape, dummy_prob, device=log_ps.device).log()
    log_dummy_neg = torch.full(log_ps.shape, 1.0 - dummy_prob, device=log_ps.device).log()
    log_qs = (a_tensors * log_dummy_tru) + (~a_tensors * log_dummy_neg)

    log_ratios = log_ps - log_qs
    strided_ratios = log_ratios.tensor_split(ep_strides)
    log_weights = torch.stack([rs.sum(0) for rs in strided_ratios])
    # log_weights = torch.concat([rs.sum(0).expand(len(rs)) for rs in strided_ratios])
    # max_log = log_weights.max()
    weights = log_weights.exp()
    return weights


def dist_safety_val(rollout: List[SimSnapshot]) -> float:
    return np.min([extract_dist(s) for s in rollout])


def extract_dist(s: SimSnapshot) -> float:
    return s.outs.true_distance


def fail_prob_eval(ep_rollouts: List[List[SimSnapshot]], pem: nn.Module, n_func, proposal_model: nn.Module, safety_func,
                   fail_thresh: float) -> float:
    ep_weights = rollout_weights(pem, n_func, proposal_model, ep_rollouts)
    safety_vals = torch.tensor([safety_func(rollout) for rollout in ep_rollouts], device=ep_weights.device)
    fail_indicator = safety_vals < fail_thresh
    print(f"Num Fails: {fail_indicator.sum().item()} / {len(fail_indicator)}")
    fail_prob = (fail_indicator * ep_weights).mean()
    return fail_prob


def fail_prob_eval_dummy(ep_rollouts: List[List[SimSnapshot]], pem: nn.Module, n_func, dummy_prob: float, safety_func,
                         fail_thresh: float) -> float:
    ep_weights = rollout_weights_dummy(pem, n_func, dummy_prob, ep_rollouts)
    safety_vals = torch.tensor([safety_func(rollout) for rollout in ep_rollouts], device=ep_weights.device)
    fail_indicator = safety_vals < fail_thresh
    print(f"Num Fails: {fail_indicator.sum().item()} / {len(fail_indicator)}")
    fail_prob = (fail_indicator * ep_weights).mean()
    return fail_prob


def fail_prob_eval_mixed(ep_rollouts: List[List[SimSnapshot]], pem: nn.Module, n_func, dummy_prob: float, safety_func,
                         fail_thresh: float) -> float:
    ep_weights = rollout_weights_mixed(pem, n_func, dummy_prob, ep_rollouts)
    safety_vals = torch.tensor([safety_func(rollout) for rollout in ep_rollouts], device=ep_weights.device)
    fail_indicator = safety_vals < fail_thresh
    print(f"Num Fails: {fail_indicator.sum().item()} / {len(fail_indicator)}")
    fail_prob = (fail_indicator * ep_weights).mean()
    return fail_prob


# Normalizes between -1 and 1
def range_norm(x: float, min_v: float, max_v: float) -> float:
    assert max_v > min_v
    return 2.0 * (x - min_v) / (max_v - min_v) - 1.0


def run():
    # ep_rollouts = load_rollouts("sim_data/mixed_dummy05_baseline_10000/s0", 10000)
    # ep_rollouts = load_rollouts("sim_data/22-09-08-17-02-51/s9")
    # ep_rollouts = load_rollouts("sim_data/STL_AGM/22-09-08-17-29-09/s9")
    ep_rollouts = load_rollouts("sim_data/STL_Smooth_Cumulative/22-09-08-18-33-10/s9")

    cem_model = FFPolicy(1, torch.tensor([12.0], device="cuda")).cuda()
    # cem_model = load_model_det(cem_model, "models/CEMs/full_loop_s8.pyt")
    # cem_model = load_model_det(cem_model, "models/CEMs/full_loop_s8.pyt")
    # cem_model = load_model_det(cem_model, "models/CEMs/STL_AGM/full_loop_s8.pyt")
    cem_model = load_model_det(cem_model, "models/CEMs/STL_Smooth-Cumulative/full_loop_s8.pyt")

    # chart_det_probs(cem_model)
    # chart_multistage_probs("models/CEMs/STL_Classic", "b")
    # chart_multistage_probs("models/CEMs/STL_Smooth-Cumulative/", "g")
    # chart_multistage_probs("models/CEMs/STL_AGM/", "r")
    # chart_multiple_multistage(["models/CEMs/STL_Classic", "models/CEMs/STL_AGM", "models/CEMs/STL_Smooth-Cumulative"])

    pem_class = load_model_det(PEMClass_Deterministic(14, 1), "models/det_baseline_full/pem_class_train_full").cuda()
    norm_stats = torch.load("models/norm_stats_mu.pt"), torch.load("models/norm_stats_std.pt")
    n_func = lambda s_inputs, norm_dims: norm_salient_input(s_inputs, norm_stats[0], norm_stats[1], norm_dims)

    classic_stl_spec = stl.G(stl.GEQ0(lambda x: extract_dist(x) - 2.0), 0, 99)
    classic_rob_f = lambda rollout: stl.stl_rob(classic_stl_spec, rollout, 0)

    # Normalized between -1 and 1
    agm_stl_spec = stl.G(stl.GEQ0(lambda x: (range_norm(extract_dist(x), 0, 13.0) - range_norm(2.0, 0.0, 13.0))), 0, 99)
    agm_rob_f = lambda rollout: stl.agm_rob(agm_stl_spec, rollout, 0)
    sc_rob_f = lambda rollout: stl.sc_rob_pos(classic_stl_spec, rollout, 0, 500)

    # print("Prev distance safety")
    # safety_fail_prob = fail_prob_eval(ep_rollouts, pem_class, n_func, cem_model, dist_safety_val, 2.0)

    # print("Classic STL Safety:")
    nlls = pem_loglikelihoods(pem_class, n_func, ep_rollouts)
    avg_nll = avg_fail_nll(nlls, classic_rob_f, 0.0, ep_rollouts)
    fail_prob = fail_prob_eval(ep_rollouts, pem_class, n_func, cem_model, classic_rob_f, 0.0)
    # fail_prob = fail_prob_eval_dummy(ep_rollouts, pem_class, n_func, 0.5, classic_rob_f, 0.0)
    print("Fail prob: ", fail_prob)
    print("Avg NLL: ", avg_nll)

    # print("Smooth Cumulative STL Safety:")
    # stl_cum_fail_prob = fail_prob_eval(ep_rollouts, pem_class, n_func, cem_model, sc_rob_f, 0.0)
    #
    # print("Old Failure Prob: ", safety_fail_prob)
    # print("STL Fail Prob: ", stl_fail_prob)
    # print("Smooth Cumulative Prob: ", stl_cum_fail_prob)


if __name__ == "__main__":
    run()
    # cem_run(os.path.join("sim_data", "22-09-05-16-48-21"), "models/CEMs/pretrain_e100_PEM.pyt")
