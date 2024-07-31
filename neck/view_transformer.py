from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LSSViewTransformer(nn.Module):
    def __init__(self, in_channels=256, out_channels=64):
        super(LSSViewTransformer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.D = 44
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.out_channels, kernel_size=1, padding=0
        )

    def forward(self, x):
        x = self.depth_net(x)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1).unsqueeze(0)
        feat = tran_feat.permute((0, 2, 3, 1)).unsqueeze(0)
        return depth, feat