import torch
import torch.nn as nn
from torch.nn import functional as F

from backbone.resnet import ResNet50


"""
Simplified FlashOcc PyTorch model
"""
class PthModel(nn.Module):
    def __init__(self) -> None:
        super(PthModel, self).__init__()

        self.img_backbone = ResNet50()

    def forward(self, x):
        x = self.img_backbone(x)

        return x


if __name__ == "__main__":
    model = PthModel()