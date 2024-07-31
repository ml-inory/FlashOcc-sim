from functools import partial
from typing import Any, Callable, List, Optional, Type, Union, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BEVPoolV2(nn.Module):

    __constants__ = ["bev_feat_shape"]
    # Attributes to match the plugin requirements.
    # Must follow the type annotations via PEP 526-style.
    # https://peps.python.org/pep-0526/#class-and-instance-variable-annotations
    bev_feat_shape: ClassVar[List[int]]

    def __init__(self, bev_feat_shape=(1,1,200,200,64)):
        super(BEVPoolV2, self).__init__()
        self.bev_feat_shape = list(bev_feat_shape)


    def forward(self, ranks_depth, ranks_feat, ranks_bev, n_points, depth, feat):
        ranks_depth = ranks_depth[:n_points]
        ranks_feat = ranks_feat[:n_points]
        ranks_bev = ranks_bev[:n_points]

        B, N, _, iH, iW = depth.shape
        C = feat.shape[-1]
        _, oD, oH, oW, _ = self.bev_feat_shape

        # flatten inputs
        depth_1d = depth.flatten()
        feat_2d = feat.reshape(B * N * iH * iW, C)

        # gather depth and feat
        gathered_depth_1d = torch.gather(input=depth_1d, dim=0, index=ranks_depth.long())
        ranks_feat = ranks_feat.reshape(ranks_feat.shape[0], 1).repeat(1, C)
        gathered_feat = torch.gather(input=feat_2d, dim=0, index=ranks_feat.long())

        # subtract zp and mul
        gathered_depth_2d = gathered_depth_1d.reshape(gathered_depth_1d.shape[0], 1)
        r_mul = gathered_depth_2d * gathered_feat

        # init with zeros
        r_scatter = torch.full(fill_value=0, size=(B * oD * oW * oH, C), dtype=torch.float32, device=r_mul.device)

        # scatter_add
        ranks_bev = ranks_bev.reshape(ranks_bev.shape[0], 1).repeat(1, C)
        r_scatter = torch.scatter_add(input=r_scatter, dim=0, index=ranks_bev.long(), src=r_mul)

        # reshape
        r = r_scatter.reshape(B, oD, oW, oH, C)

        return r