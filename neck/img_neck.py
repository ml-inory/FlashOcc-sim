from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CustomFPN(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_mode="nearest"):
        super(CustomFPN, self).__init__()

        self.lateral_conv0 = nn.Conv2d(in_channels[0], out_channels, (1, 1))
        self.lateral_conv1 = nn.Conv2d(in_channels[1], out_channels, (1, 1))
        self.fpn_conv0 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1))
        self.upsample_mode = upsample_mode

    def forward(self, x, y):
        laterals = [self.lateral_conv0(x), self.lateral_conv1(y)]
        laterals[0] += F.interpolate(laterals[-1], size=laterals[0].shape[2:], mode=self.upsample_mode)
        return self.fpn_conv0(laterals[0])