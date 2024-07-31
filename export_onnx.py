import torch
import torchvision
from pth_model import PthModel
from ops.bevpoolv2 import BEVPoolV2
import onnx
from onnx import helper


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
    checkpoint = torch.load("flashocc-r50-M0-256x704.pth")
    state_dict = fix_state_dict(checkpoint["state_dict"])
    model.load_state_dict(state_dict)
    model.eval()
    return model

 
img = torch.randn(6, 3, 256, 704)
ranks_depth = torch.randint(low=0, high=185856, size=(185856,), dtype=torch.int32)
ranks_feat = torch.randint(low=0, high=256, size=(185856,), dtype=torch.int32)
ranks_bev = torch.randint(low=0, high=256, size=(185856,), dtype=torch.int32)
n_points = torch.randint(low=0, high=185856, size=(1,), dtype=torch.int32)

onnx_inputs = (img, ranks_depth, ranks_feat, ranks_bev, n_points)

input_names = [
    'img', 'ranks_depth', 'ranks_feat', 'ranks_bev',
    'n_points'
]
output_names = ['cls_occ_label']

model = load_pth_model()
torch.onnx.export(
    model,
    onnx_inputs,
    "bevdet_ax.onnx",
    input_names=input_names,
    output_names=output_names,
    export_params=True,
    do_constant_folding=True,
    opset_version=16,
    export_modules_as_functions={BEVPoolV2}
)


onnx_model = onnx.load("bevdet_ax.onnx")
graph = onnx_model.graph
for i, node in enumerate(graph.node):
    if node.op_type == "BEVPoolV2":
        ax_bev = helper.make_node(op_type="AxBevPool", 
                              name="bev_pool_v2", 
                              inputs=["/img_view_transformer/Unsqueeze_output_0", 
                                      "/img_view_transformer/Unsqueeze_1_output_0", 
                                      "ranks_depth", "ranks_feat", "ranks_bev", "n_points"], 
                              outputs=["/bevpool/BEVPoolV2_output_0"], 
                              domain="ai.onnx.contrib",
                              bev_feat_shape=(1,1,200,200,64))
        graph.node.append(ax_bev)
        del graph.node[i]

onnx.save(onnx_model, "bevdet_ax.onnx")
print("Export model to bevdet_ax.onnx")   