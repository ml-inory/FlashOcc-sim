import torch
import torchvision
from pth_model import PthModel
from ops.bevpoolv2 import BEVPoolV2

 
img = torch.randn(6, 3, 256, 704)
ranks_depth = torch.randint(low=0, high=255, size=(185856,))
ranks_feat = torch.randint(low=0, high=255, size=(185856,))
ranks_bev = torch.randint(low=0, high=255, size=(185856,))
n_points = torch.randint(low=0, high=255, size=(1,))

onnx_inputs = (img, ranks_depth, ranks_feat, ranks_bev, n_points)

input_names = [
    'img', 'ranks_depth', 'ranks_feat', 'ranks_bev',
    'n_points'
]
output_names = ['cls_occ_label']

model = PthModel()
model.eval()
torch.onnx.export(
    model,
    onnx_inputs,
    "bevdet.onnx",
    input_names=input_names,
    output_names=output_names,
    export_params=True,
    do_constant_folding=True,
    opset_version=16,
    export_modules_as_functions={BEVPoolV2}
)
print("Export model to bevdev.onnx")