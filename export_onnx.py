import torch
import torchvision
from pth_model import PthModel
from ops.bevpoolv2 import BEVPoolV2
import onnx
from onnx import helper, TensorProto
from onnxsim import simplify


def fix_state_dict(state_dict):
    fix_pairs = [
        ("img_neck.lateral_convs.0.conv.weight", "img_neck.lateral_convs.0.weight"),
        ("img_neck.lateral_convs.1.conv.weight", "img_neck.lateral_convs.1.weight"),
        ("img_neck.lateral_convs.0.conv.bias", "img_neck.lateral_convs.0.bias"),
        ("img_neck.lateral_convs.1.conv.bias", "img_neck.lateral_convs.1.bias"),
        ("img_neck.fpn_convs.0.conv.weight", "img_neck.fpn_convs.0.weight"),
        ("img_neck.fpn_convs.0.conv.bias", "img_neck.fpn_convs.0.bias"),
        ("occ_head.final_conv.conv.weight", "occ_head.final_conv.weight"),
        ("occ_head.final_conv.conv.bias", "occ_head.final_conv.bias"),
    ]

    for old_key, new_key in fix_pairs:
        state_dict[new_key] = state_dict.pop(old_key)

    return state_dict


def load_pth_model(cpkt="flashocc-r50-M0-256x704.pth"):
    model = PthModel()
    checkpoint = torch.load("flashocc-r50-M0-256x704.pth", map_location=torch.device("cpu"))
    state_dict = fix_state_dict(checkpoint["state_dict"])
    model.load_state_dict(state_dict)
    model.eval()
    return model

 
img = torch.randn(6, 3, 256, 704)
indices_depth = torch.randint(low=0, high=185856, size=(800000,), dtype=torch.int32)
indices_feat = torch.randint(low=0, high=185856 // 64, size=(800000,), dtype=torch.int32)

onnx_inputs = (img, indices_depth, indices_feat,)

input_names = [
    'img', 'indices_depth', 'indices_feat'
]
output_names = ['cls_occ_label']
model_name = "bevdet_axmaxn.onnx"

model = load_pth_model()
torch.onnx.export(
    model,
    onnx_inputs,
    model_name,
    dynamic_axes=None,
    input_names=input_names,
    output_names=output_names,
    export_params=True,
    do_constant_folding=True,
    opset_version=16
)

onnx_model = onnx.load(model_name)
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"

onnx.save(model_simp, model_name)
print(f"Export model to {model_name}")   