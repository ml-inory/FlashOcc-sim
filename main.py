import os, argparse
import json
import numpy as np
from dataloader import Dataloader
from model import Model
from visualize import visualize
import torch
import cv2


colors_map = np.array(
    [
        [0, 0, 0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],  # 2 pedestrian  Blue
        [47, 79, 79, 255],  # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],  # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255],  # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],  # 9 motorcycle  Slategrey
        [222, 184, 135, 255],  # 10 building Burlywood
        [0, 175, 0, 255],  # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255],  # 14 walkable, sidewalk
        [255, 0, 0, 255],  # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run FlashOcc with onnx")
    parser.add_argument("--config", help="test config file path", default="config.json")
    parser.add_argument("--onnx", help="onnx file", default="bevdet_ax.onnx")
    parser.add_argument("--data_root", help="data root", default="data/nuscenes")
    parser.add_argument("--img", help="image path", required=True)
    args = parser.parse_args()
    return args


def save_onnx_inputs(onnx_inputs):
    os.makedirs("inputs", exist_ok=True)
    for name, value in onnx_inputs.items():
        value.tofile(os.path.join("inputs", name + ".bin"))
    print("Saved input data to inputs folder")


def save_onnx_outputs(onnx_outputs, name="cls_occ_label.bin"):
    os.makedirs("outputs", exist_ok=True)
    onnx_outputs.tofile(os.path.join("outputs", name))
    print("Saved output data to outputs folder")


def vis_occ(semantics):
    # simple visualization of result in BEV
    semantics_valid = np.logical_not(semantics == 17)
    d = np.arange(16).reshape(1, 1, 16)
    d = np.repeat(d, 200, axis=0)
    d = np.repeat(d, 200, axis=1).astype(np.float32)
    d = d * semantics_valid
    selected = np.argmax(d, axis=2)

    selected_torch = torch.from_numpy(selected)
    semantics_torch = torch.from_numpy(semantics)

    occ_bev_torch = torch.gather(
        semantics_torch, dim=2, index=selected_torch.unsqueeze(-1)
    )
    occ_bev = occ_bev_torch.numpy()

    occ_bev = occ_bev.flatten().astype(np.int32)
    occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
    occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
    occ_bev_vis = cv2.resize(occ_bev_vis, (400, 400))
    return occ_bev_vis


def main():
    args = parse_args()

    with open(args.config) as f:
        model_config = json.load(f)

    model = Model(args.onnx, model_config)

    dataloader = Dataloader(args.data_root, model_config)
    inputs, info = dataloader.load(args.img)

    bev_inputs = model.get_bev_pool_input(inputs)

    onnx_inputs = {
        "img": inputs[0][0],
        "ranks_depth": bev_inputs[0],
        "ranks_feat": bev_inputs[1],
        "ranks_bev": bev_inputs[2],
        "n_points": bev_inputs[3],
    }
    save_onnx_inputs(onnx_inputs)

    onnx_outputs = model.forward(onnx_inputs)

    save_onnx_outputs(onnx_outputs)

    # onnx_outputs = np.fromfile("outputs/cls_occ_label.bin", dtype=np.int32).reshape(
    #     (200, 200, 16)
    # )

    result = vis_occ(onnx_outputs)
    cv2.imwrite("sementics.jpg", result)
    print("Saved sementics to sementics.jpg")

    visualize(onnx_outputs, info, visible=False)


if __name__ == "__main__":
    main()
