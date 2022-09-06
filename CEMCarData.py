import os
import os.path
import json
import time
from dataclasses import dataclass

import dacite
from typing import Callable, List, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, FloatTensor, BoolTensor
import torch.nn.functional as F

from adaptiveImportanceSampler import FFPolicy, get_quantile, cross_entropy_train, get_ep_strides
from carlaUtils import SimSnapshot, to_salient_var, norm_salient_input
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


def run():
    # For each file in the specified folder
    sim_data_folder = os.path.join("sim_data", "22-09-05-16-48-21")
    ep_fns = os.listdir(sim_data_folder)
    print(ep_fns)

    ep_rollouts = []
    for ep_fn in ep_fns:
        with open(os.path.join(sim_data_folder, ep_fn), 'r') as f:
            ep_rollouts.append([dacite.from_dict(data_class=SimSnapshot, data=s) for s in json.load(f)])

    # Single variable of distance right now? But could also add velocity and accelerations of both vehicles (this is just to check that *anything* works...)
    print("Pre-training")
    cem_model = FFPolicy(1, torch.tensor([12.0], device="cuda")).cuda()

    s_tensors, a_tensors = tensors_from_rollouts(ep_rollouts)

    pre_train_cem(cem_model, s_tensors, a_tensors, len(ep_rollouts))

    save_model_det(cem_model, "models/CEMs/pretrain_e100_PEM.pyt")

    with torch.no_grad():
        pre_dists = torch.linspace(0, 13, 100, device="cuda").view(-1, 1)
        pre_probs = cem_model(pre_dists)[:, 1].exp()
        plt.plot(pre_dists.detach().cpu(), pre_probs.detach().cpu())
        plt.ylim([0, 1])
        plt.ylabel("Detection Probability")
        plt.xlabel("Distance")
        plt.show()

    safety_vals = [np.min([s.outs.true_distance for s in rollout]) for rollout in ep_rollouts]
    failure_thresh = get_quantile(safety_vals, 0.95, 2.0)

    num_fails = len([s for s in safety_vals if s <= failure_thresh])
    print(f"Failure Thresh: {failure_thresh} ({num_fails} / {len(ep_rollouts)} episodes)")

    pem_class = load_model_det(PEMClass_Deterministic(14, 1), "models/det_baseline_full/pem_class_train_full").cuda()
    norm_stats = torch.load("models/norm_stats_mu.pt"), torch.load("models/norm_stats_std.pt")
    n_func = lambda s_inputs, norm_dims: norm_salient_input(s_inputs, norm_stats[0], norm_stats[1], norm_dims)

    state_pem_ins = torch.stack([to_salient_var(s.model_ins, n_func) for rollout in ep_rollouts for s in rollout[:-1]]).to(device="cuda")

    assert len(state_pem_ins) == len(s_tensors)
    assert torch.all(state_pem_ins[5] == to_salient_var(ep_rollouts[0][5].model_ins, n_func).to(device="cuda"))

    with torch.no_grad():
        pem_logits = pem_class(state_pem_ins).view(-1)
        log_ps = F.logsigmoid(pem_logits)
        ep_strides = get_ep_strides(ep_rollouts)
        fail_indicator = torch.concat([torch.full((len(sr) - 1,), sv <= failure_thresh, device="cuda") for sr, sv in zip(ep_rollouts, safety_vals)])

    cross_entropy_train(cem_model, s_tensors, a_tensors, ep_strides, log_ps, fail_indicator, 1000, len(ep_rollouts))

    save_model_det(cem_model, "models/CEMs/cem_e100_s1.pyt")

    with torch.no_grad():
        pre_dists = torch.linspace(0, 13, 100, device="cuda").view(-1, 1)
        pre_probs = cem_model(pre_dists)[:, 1].exp()
        plt.plot(pre_dists.detach().cpu(), pre_probs.detach().cpu())
        plt.ylim([0, 1])
        plt.ylabel("Detection Probability")
        plt.xlabel("Distance")
        plt.show()


if __name__ == "__main__":
    run()
