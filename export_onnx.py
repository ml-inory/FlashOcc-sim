import torch
import torchvision
from pth_model import PthModel
 
model = PthModel()
model.eval()
torch.onnx.export(
    model,
    torch.randn(6, 3, 256, 704),
    "bevdet.onnx",
    export_params=True,
    do_constant_folding=True,
    opset_version=11,
)
print("Export model to bevdev.onnx")