import os
import os.path
import json

import dacite
from typing import List, Tuple, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.special

from experiment_config import ExpConfig

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import torch
from torch import nn, FloatTensor, BoolTensor, Tensor
import torch.nn.functional as F

import stl
from adaptiveImportanceSampler import FFPolicy, get_quantile, cross_entropy_train, get_ep_strides
from carlaUtils import SimSnapshot, to_salient_var, norm_salient_input
from utils import log1mexp, range_norm
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
            ax.plot(pre_dists.detach().cpu(), pre_probs.detach().cpu(),
                    label=name_map[os.path.basename(cem_name).split("_")[1]])
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
                 chart: bool, model_save_path: Optional[str] = None) -> Tuple[nn.Module, float, float, int, float]:
    s_tensors, a_tensors = tensors_from_rollouts(ep_rollouts)

    safety_vals = np.array([safety_func(rollout) for rollout in ep_rollouts])
    failure_thresh = get_quantile(safety_vals, 0.95, 0.0)

    num_fails = len(safety_vals[safety_vals <= failure_thresh])
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

        ep_target_lls = torch.stack([lps.sum(0) for lps in log_ps.tensor_split(ep_strides)])

        # Even though we are doing adaptive thresholding, we want to record the Log likelihoods the current stage is getting on the *true* failure thresh
        avg_f_ll = avg_fail_lls(ep_target_lls, safety_func, 0.0, ep_rollouts)
        est_fail_prob = fail_prob_eval(ep_rollouts, pem_model, n_func, cem_model, safety_func, 0.0)
        num_tru_fails = len(safety_vals[safety_vals <= 0.0])

    cross_entropy_train(cem_model, s_tensors, a_tensors, ep_strides, log_ps, fail_indicator, 1000, len(ep_rollouts))

    if model_save_path is not None:
        save_model_det(cem_model, model_save_path)

    if chart:
        chart_det_probs(cem_model)

    return cem_model, failure_thresh, est_fail_prob, num_tru_fails, avg_f_ll


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

    one_step_cem(ep_rollouts, cem_model, pem_class, norm_stats, True)


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


def avg_fail_lls(nlls: FloatTensor, safety_func, fail_thresh: float, rollouts: List[List[SimSnapshot]]) -> float:
    safety_vals = torch.tensor([safety_func(rollout) for rollout in rollouts], device=nlls.device)
    fail_indicator = safety_vals <= fail_thresh
    fail_nlls = nlls[fail_indicator]

    if len(fail_nlls) == 0:
        return -torch.inf

    return (torch.logsumexp(fail_nlls, 0) - np.log(len(fail_nlls))).item()


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
    return fail_prob.item()


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


def averaged_rollout_metrics(experiment_folder: str):
    # Load folder
    rf_paths = [os.path.join(experiment_folder, r) for r in sorted(next(os.walk(experiment_folder))[1])[:-1]]

    fail_nlls = np.stack([np.loadtxt(os.path.join(rfp, "fail_nlls.txt")) for rfp in rf_paths])
    fail_threshes = np.stack([np.loadtxt(os.path.join(rfp, "failure_threshes.txt")) for rfp in rf_paths])
    fails_per_stage = np.stack([np.loadtxt(os.path.join(rfp, "num_fails.txt")) for rfp in rf_paths])
    fail_probs = np.stack([np.loadtxt(os.path.join(rfp, "fail_probs.txt")) for rfp in rf_paths])

    fail_threshes_mu = fail_threshes.mean(0)
    fail_threshes_std = fail_threshes.std(0)
    # fails_per_stage_mu = fails_per_stage.mean(0)

    fail_nlls_mu = scipy.special.logsumexp(fail_nlls, 0) - np.log(len(fail_nlls))  # Need special mean for logs
    fail_nlls_std = None

    fail_prob_mu = fail_probs.mean(0)

    return fail_prob_mu, fail_threshes_mu, fail_threshes_std, fail_nlls_mu


def avg_dist_v_prob(models_folder: str, stage: int) -> Tuple[Tensor, Tensor, Tensor]:
    model_paths = [os.path.join(models_folder, r, f"full_loop_s{stage}.pyt") for r in
                   sorted(next(os.walk(models_folder))[1])[:-1]]

    pre_dists = torch.linspace(0, 13, 100, device="cuda").view(-1, 1)
    model_probs = []
    for model_path in model_paths:
        cem_model = load_model_det(FFPolicy(1, torch.tensor([12.0], device="cuda")).cuda(), model_path)
        model_probs.append(cem_model(pre_dists)[:, 1].exp())

    model_probs = torch.stack(model_probs)
    model_probs_mu = torch.mean(model_probs, 0)
    model_probs_std = torch.std(model_probs, 0)

    return pre_dists.view(-1).cpu().detach(), model_probs_mu.cpu().detach(), model_probs_std.cpu().detach()

    # plt.plot(pre_dists.view(-1).cpu().detach(), model_probs_mu.cpu().detach())
    # plt.ylim(0, 1)
    # plt.fill_between(pre_dists.view(-1).cpu().detach(), model_probs_mu.cpu().detach() - model_probs_std.cpu().detach(), model_probs_mu.cpu().detach() + model_probs_std.cpu().detach(), alpha=0.4)
    # plt.show()


def analyze_rollouts(rollout_folder: str, pem_path: str, cem_path: str, metric: str):
    ep_rollouts = load_rollouts(rollout_folder)
    timesteps = len(ep_rollouts[0])

    cem_model = FFPolicy(1, torch.tensor([12.0], device="cuda")).cuda()
    cem_model = load_model_det(cem_model, cem_path)

    pem_model = load_model_det(PEMClass_Deterministic(14, 1), pem_path).cuda()
    norm_stats = torch.load("models/norm_stats_mu.pt", torch.load("models/norm_stats_std.pt"))
    n_func = lambda s_inputs, norm_dims: norm_salient_input(s_inputs, norm_stats[0], norm_stats[1], norm_dims)

    classic_stl_spec = stl.G(stl.GEQ0(lambda x: extract_dist(x) - 2.0), 0, timesteps - 1)
    classic_rob_f = lambda rollout: stl.stl_rob(classic_stl_spec, rollout, 0)

    # Normalized between -1 and 1
    agm_stl_spec = stl.G(stl.GEQ0(lambda x: (range_norm(extract_dist(x), 0, 13.0) - range_norm(2.0, 0.0, 13.0))), 0,
                         timesteps - 1)
    agm_rob_f = lambda rollout: stl.agm_rob(agm_stl_spec, rollout, 0)
    sc_rob_f = lambda rollout: stl.sc_rob_pos(classic_stl_spec, rollout, 0, 500)

    if metric == "classic":
        safety_f = classic_rob_f
    elif metric == "agm":
        safety_f = agm_rob_f
    elif metric == "smooth-cumulative":
        safety_f = sc_rob_f

    lls = pem_loglikelihoods(pem_model, n_func, ep_rollouts)
    avg_ll = avg_fail_lls(lls, classic_rob_f, 0.0, ep_rollouts)
    fail_prob = fail_prob_eval(ep_rollouts, pem_model, n_func, cem_model, classic_rob_f, 0.0)

    print("Fail prob: ", fail_prob)
    print("Avg LL: ", avg_ll)


def failure_prob_from_experiment(model_folder, experiment_folder, exp_conf: ExpConfig):
    ep_rollouts = load_rollouts(os.path.join(experiment_folder, "r0", "s9"))
    cem_model = FFPolicy(1, torch.tensor([12.0], device="cuda")).cuda()
    cem_model = load_model_det(cem_model, os.path.join(model_folder, "r0", "full_loop_s8.pyt"))

    pem_class = load_model_det(PEMClass_Deterministic(14, 1), "models/det_baseline_full/pem_class_train_full").cuda()
    norm_stats = torch.load("models/norm_stats_mu.pt"), torch.load("models/norm_stats_std.pt")
    n_func = lambda s_inputs, norm_dims: norm_salient_input(s_inputs, norm_stats[0], norm_stats[1], norm_dims)

    stl_spec = stl.G(stl.GEQ0(lambda x: extract_dist(x) - 2.0), 0, exp_conf.timesteps - exp_conf.vel_burn_in_time - 1)
    # rob_f = create_safety_func(exp_conf.safety_func, stl_spec)
    rob_f = None
    raise NotImplementedError("Only got halfway to coding this")

    # nlls = pem_loglikelihoods(pem_class, n_func, ep_rollouts)
    # avg_nll = avg_fail_lls(nlls, rob_f, 0.0, ep_rollouts)
    fail_prob = fail_prob_eval(ep_rollouts, pem_class, n_func, cem_model, rob_f, 0.0)

    print("Fail prob: ", fail_prob)
    # print("Avg NLL: ", avg_nll)


def chart_avg_rollout_metrics(data_paths_labelled: List[Tuple[str, str]]):
    # NLLs, Failure Threshes, And Estimate Failure Probabilities
    for data_path, label in data_paths_labelled:
        fail_prob_mu, fts_mu, fts_std, nlls_mu = averaged_rollout_metrics(data_path)
        plt.plot(range(len(fts_mu)), fts_mu, label=label)
        plt.fill_between(range(len(fts_mu)), fts_mu - fts_std / np.sqrt(5), fts_mu + fts_std / np.sqrt(5), alpha=0.3)
        print(f"{label} Fail Prob: {fail_prob_mu[-1]}, NLL: {nlls_mu[-1]}")
    plt.legend(loc="best")
    plt.xlabel("$\kappa$")
    plt.ylabel("$\gamma_{\kappa}$")
    plt.show()


if __name__ == "__main__":

    model_paths_labelled = [
        # ("models/CEMs/STL_Classic/22-10-19-12-49-34", "Classic"),
        ("models/CEMs/STL_AGM/22-10-20-11-42-35", "AGM"),
        # ("models/CEMs/STL_Smooth_Cumulative/22-10-18-18-36-56", "SC")
    ]

    data_paths_labelled = [
        # ("sim_data/STL_Classic/22-10-19-12-49-34", "Classic"),
        ("sim_data/STL_AGM/22-10-20-11-42-35", "AGM"),
        # ("sim_data/STL_Smooth_Cumulative/22-10-18-18-36-56", "SC")
    ]

    chart_avg_rollout_metrics(data_paths_labelled)

    # Distance Visualizations
    fig, axs = plt.subplots(3, 3)
    axs = axs.reshape(-1)
    plt.rcParams['font.family'] = 'serif'

    for s in range(9):
        for model_path, label in model_paths_labelled:
            dists, mus, stds = avg_dist_v_prob(model_path, s)
            axs[s].plot(dists, mus, label=label)
            axs[s].fill_between(dists, mus - stds / np.sqrt(5), mus + stds / np.sqrt(5), alpha=0.3)
            axs[s].set_title(f"$\kappa = {s}$")

        axs[s].spines['top'].set_visible(False)
        axs[s].spines['right'].set_visible(False)
        axs[s].set_ylim(0, 1)
        axs[s].legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Failure Probability Estimates
    # run()
