import torch
import torch.nn as nn
from torch.nn import functional as F

from backbone.resnet import ResNet50
from neck.img_neck import CustomFPN
from neck.view_transformer import LSSViewTransformer
from ops.bevpoolv2 import BEVPoolV2
from backbone.resnet import CustomResNet
from neck.fpn_lss import FPN_LSS
from head.bev_occ_head import BEVOCCHead2D


"""
Simplified FlashOcc PyTorch model
"""
class PthModel(nn.Module):
    def __init__(self) -> None:
        super(PthModel, self).__init__()

        self.numC_Trans = 64
        self.img_backbone = ResNet50()
        self.img_neck = CustomFPN([1024, 2048], 256)
        self.view_transformer = LSSViewTransformer(in_channels=256, out_channels=self.numC_Trans)
        self.bevpool = BEVPoolV2()
        self.img_bev_encoder_backbone = CustomResNet(numC_input=self.numC_Trans, num_channels=[self.numC_Trans * 2, self.numC_Trans * 4, self.numC_Trans * 8])
        self.img_bev_encoder_neck = FPN_LSS(in_channels=self.numC_Trans * 8 + self.numC_Trans * 2, out_channels=128)
        self.occ_head = BEVOCCHead2D(in_dim=128, out_dim=128, Dz=16, use_mask=True, num_classes=18, use_predicter=True)

    def forward(self, img, ranks_depth, ranks_feat, ranks_bev, n_points):
        x, y = self.img_backbone(img)
        x = self.img_neck(x, y)
        depth, feat = self.view_transformer(x)

        x = self.bevpool(ranks_depth, ranks_feat, ranks_bev, n_points, depth, feat)
        x = x.permute((0,4,1,2,3)).reshape(1, 64, 200, 200)

        feats = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(feats)
        x = self.occ_head(x)

        return x.argmax(dim=-1)


if __name__ == "__main__":
    model = PthModel()