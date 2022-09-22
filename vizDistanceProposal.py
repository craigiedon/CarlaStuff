import argparse

import torch

from CEMCarData import chart_det_probs
from adaptiveImportanceSampler import FFPolicy
from pems import load_model_det, PEMClass_Deterministic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization of proposal distributions")
    parser.add_argument("cem-path", help="File path of learned adaptive importance sampling proposal model")

    args = parser.parse_args()

    chart_det_probs(load_model_det(FFPolicy(1, torch.tensor([12.0])), args.cem_path))
