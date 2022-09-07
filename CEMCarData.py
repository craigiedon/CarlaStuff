import os
import os.path
import json
import time
from dataclasses import dataclass

import dacite
from typing import Callable, List, Any, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, FloatTensor, BoolTensor
import torch.nn.functional as F

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
    with torch.no_grad():
        pre_dists = torch.linspace(0, 13, 100, device="cuda").view(-1, 1)
        pre_probs = model(pre_dists)[:, 1].exp()
        plt.plot(pre_dists.detach().cpu(), pre_probs.detach().cpu())
        plt.ylim([0, 1])
        plt.ylabel("Detection Probability")
        plt.xlabel("Distance")
        plt.show()


def load_rollouts(folder_path: str) -> List[List[SimSnapshot]]:
    ep_fns = os.listdir(folder_path)
    print(ep_fns)
    ep_rollouts = []
    for ep_fn in ep_fns:
        with open(os.path.join(folder_path, ep_fn), 'r') as f:
            ep_rollouts.append([dacite.from_dict(data_class=SimSnapshot, data=s) for s in json.load(f)])
    return ep_rollouts


def one_step_cem(ep_rollouts: List[List[SimSnapshot]], cem_model: nn.Module, pem_model: nn.Module, norm_stats,
                 chart: bool, model_save_path: Optional[str] = None) -> nn.Module:
    s_tensors, a_tensors = tensors_from_rollouts(ep_rollouts)

    safety_vals = [np.min([s.outs.true_distance for s in rollout]) for rollout in ep_rollouts]
    failure_thresh = get_quantile(safety_vals, 0.95, 2.0)

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


def dist_safety_val(rollout: List[List[SimSnapshot]]) -> float:
    return np.min([s.outs.true_distance for s in rollout])


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


def run():
    ep_rollouts = load_rollouts("sim_data/22-09-07-19-21-32/s0")

    # cem_model = FFPolicy(1, torch.tensor([12.0], device="cuda")).cuda()
    # cem_model = load_model_det(cem_model, "models/CEMs/full_loop_s8.pyt")
    #
    # chart_det_probs(cem_model)

    pem_class = load_model_det(PEMClass_Deterministic(14, 1), "models/det_baseline_full/pem_class_train_full").cuda()
    norm_stats = torch.load("models/norm_stats_mu.pt"), torch.load("models/norm_stats_std.pt")
    n_func = lambda s_inputs, norm_dims: norm_salient_input(s_inputs, norm_stats[0], norm_stats[1], norm_dims)

    fail_prob = fail_prob_eval_dummy(ep_rollouts, pem_class, n_func, 0.5, dist_safety_val, 2.0)
    print("Failure Prob: ", fail_prob)


if __name__ == "__main__":
    run()
    # cem_run(os.path.join("sim_data", "22-09-05-16-48-21"), "models/CEMs/pretrain_e100_PEM.pyt")
