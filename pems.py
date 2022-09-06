import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F


class SalientObstacleDataset(Dataset):
    def __init__(self, s_inputs, s_labels):
        # Input Format:
        #   0: <Class Num>
        #   1: <Truncation>
        #   2: <Occlusion>
        #   3: <alpha>
        #   4-6: <dim_w> <dim_l> <dim_h>
        #   7-9: <loc_x> <loc_y> <loc_z>
        #   10: <rot_y>
        self.s_inp = s_inputs

        # Normalize 1, 3, 4,5,6,7,8,9,10
        norm_dims = [1, 3, 4, 5, 6, 7, 8, 9, 10]
        self.normed_ins, self.s_in_mus, self.s_in_stds = normalize_salient_data(s_inputs, norm_dims)

        # Pick Subset
        self.pose_subset = [7, 8, 9, 10]

        # One-hot encode 0, 2
        one_hot_cats = F.one_hot(self.s_inp[:, 0].to(dtype=torch.long), 7)
        one_hot_occlusion = F.one_hot(self.s_inp[:, 2].to(dtype=torch.long), 3)

        ## Final Indexing:
        # 0-6 Vehicle Cat One-hot
        # 7-9: Obscured One-hot
        # 10,11,12: x,y,z cam loc
        # 13: Rot y
        self.final_ins = torch.cat((one_hot_cats, one_hot_occlusion, self.normed_ins[:, self.pose_subset]), dim=1)
        assert self.final_ins.shape[0] == len(s_inputs)
        assert self.final_ins.shape[1] == 10 + len(self.pose_subset)

        # Label Format:
        #   0: <Detected>
        #   1-2: <bbox cx> <bbox cy>
        #   3-4: <bbox_w> <bbox_h>
        #   5-6: <err cx> <err cy>
        #   7-8: <err w> <err h>
        self.s_label = s_labels

    def __len__(self):
        return len(self.s_inp)

    def __getitem__(self, index):
        return self.final_ins[index], self.s_label[index]


def normalize_salient_data(s_inputs, norm_dims):
    in_mu = s_inputs.mean(0)
    in_std = s_inputs.std(0)

    normed_inputs = torch.detach(s_inputs)
    normed_inputs[:, norm_dims] = (normed_inputs[:, norm_dims] - in_mu[norm_dims]) / in_std[norm_dims]

    return normed_inputs, in_mu, in_std


class PEMClass_Deterministic(nn.Module):
    def __init__(self, in_d, out_d, h=20, use_cuda=False):
        super().__init__()
        self.ff_nn = nn.Sequential(
            nn.Linear(in_d, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),

            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),

            nn.Linear(h, out_d),
        )

        if use_cuda:
            self.cuda()

    def forward(self, x):
        cat_logits = self.ff_nn(x)
        return cat_logits


class PEMReg_Aleatoric(nn.Module):
    def __init__(self, in_d, out_d, h=20, use_cuda=True):
        super().__init__()
        self.ff_mu = nn.Sequential(
            nn.Linear(in_d, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),

            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),

            nn.Linear(h, out_d)
        )
        self.ff_log_sig = nn.Sequential(
            nn.Linear(in_d, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),

            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),

            nn.Linear(h, out_d)
        )
        if use_cuda:
            self.cuda()

    def forward(self, x):
        mu = self.ff_mu(x)
        log_sig = self.ff_log_sig(x)
        return mu, log_sig


class PEMReg_Deterministic(nn.Module):
    def __init__(self, in_d, out_d, h=20, use_cuda=True):
        super().__init__()
        self.ff_nn = nn.Sequential(
            nn.Linear(in_d, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),

            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),

            nn.Linear(h, out_d),
        )

        if use_cuda:
            self.cuda()

    def forward(self, x):
        output = self.ff_nn(x)
        return output


def save_model_det(model, f_name: str):
    torch.save(model.state_dict(), f_name)


def load_model_det(model_skeleton, model_path: str):
    model_skeleton.load_state_dict(torch.load(model_path))
    model_skeleton.eval()
    return model_skeleton
