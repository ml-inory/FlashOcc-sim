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

    
    def forward(self, depth, feat, ranks_depth, ranks_feat, maxn):
        """
        Args:
            depth: (B, N, D, fH, fW)
            feat:  (B, N, fH, fW, C)
            ranks_depth: (D_Z * D_Y * D_X * maxn),
            ranks_feat:  (D_Z * D_Y * D_X * maxn),
            bev_feat_shape: (B, D_Z, D_Y, D_X, C)
        Returns:
            r: bev feature in shape (B, C, Dz, Dy, Dx)
        """
        B, N, D, iH, iW = depth.shape
        C = feat.shape[-1]
        _, oD, oW, oH, _ = self.bev_feat_shape
        # flatten inputs
        depth_2d = depth.reshape(B * N * D * iH * iW, 1)
        feat_2d = feat.reshape(B * N * iH * iW, C)
        depth_2d = torch.cat((depth_2d, torch.zeros([1, 1], dtype=torch.float32, device=depth_2d.device)), 0)
        feat_2d = torch.cat((feat_2d, torch.zeros([1, 64], dtype=torch.float32, device=feat_2d.device)), 0)
        # gather depth and feat
        # gathered_depth = torch.gather(input=depth_2d, dim=0, index=ranks_depth.long())
        # gathered_feat = torch.gather(input=feat_2d, dim=0, index=ranks_feat.long())
        gathered_depth = torch.index_select(depth_2d, 0, ranks_depth)
        gathered_feat = torch.index_select(feat_2d, 0, ranks_feat)
        # subtract zp and mul
        r_mul = gathered_depth * gathered_feat
        # scatter_add
        r_mul = r_mul.reshape(oD, oW, oH, maxn, C)
        r_scatter = r_mul.sum(dim=3, keepdim=True)
        # permute
        r = r_scatter.permute(3, 0, 1, 2, 4)
        return r
    