from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CustomFPN(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_mode="nearest"):
        super(CustomFPN, self).__init__()

        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_channels[i], out_channels, kernel_size=1) for i in range(len(in_channels))])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1))])
        self.upsample_mode = upsample_mode

    def forward(self, inputs):
        laterals = [
            self.lateral_convs[i](inp) for i, inp in enumerate(inputs)
        ]

        laterals[0] += F.interpolate(laterals[1], size=laterals[0].shape[2:], mode=self.upsample_mode)
        return self.fpn_convs[0](laterals[0])