import os, argparse
import json
from dataloader import Dataloader
from model import Model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run FlashOcc with onnx')
    parser.add_argument('--config', help='test config file path', default='config.json')
    parser.add_argument('--onnx', help='onnx file', default='bevdet_ax.onnx')
    parser.add_argument('--data_root', help='data root', default='data/nuscenes')
    parser.add_argument('--img', help='image path', required=True)
    args = parser.parse_args()
    return args


def save_onnx_inputs(onnx_inputs):
    os.makedirs("inputs", exist_ok=True)
    for name, value in onnx_inputs.items():
        value.tofile(os.path.join("inputs", name + ".bin"))


def save_onnx_outputs(onnx_outputs):
    os.makedirs("outputs", exist_ok=True)
    onnx_outputs.tofile(os.path.join("outputs", "cls_occ_label.bin"))


def main():
    args = parse_args()

    with open(args.config) as f:
        model_config = json.load(f)

    model = Model(args.onnx, model_config)

    dataloader = Dataloader(args.data_root, model_config)
    inputs = dataloader.load(args.img)
    
    bev_inputs = model.get_bev_pool_input(inputs)

    onnx_inputs = {
        "img": inputs[0][0],
        "ranks_depth": bev_inputs[0],
        "ranks_feat": bev_inputs[1],
        "ranks_bev": bev_inputs[2],
        "n_points": bev_inputs[3]
    }
    save_onnx_inputs(onnx_inputs)

    onnx_outputs = model.forward(onnx_inputs)
    save_onnx_outputs(onnx_outputs)
    

if __name__ == '__main__':
    main()