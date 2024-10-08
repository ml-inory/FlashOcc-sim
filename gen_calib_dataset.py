"""
Generate calibration dataset for Axera
"""
import os, argparse
import json
import numpy as np
import glob
import random
from dataloader import Dataloader
from model import Model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate calibration dataset for Axera")
    parser.add_argument("--config", help="test config file path", default="config.json")
    parser.add_argument("--onnx", help="onnx file", default="bevdet_ax.onnx")
    parser.add_argument("--data_root", help="data root", default="data/nuscenes")
    parser.add_argument("--num", help="num of data", type=int, default=20)
    parser.add_argument("--no_shuffle", help="whether to shuffle, default shuffle", action="store_true")
    parser.add_argument("--output_dir", help="output path", default="calib_dataset")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.config) as f:
        model_config = json.load(f)

    dataloader = Dataloader(args.data_root, model_config)
    model = Model(args.onnx, model_config)

    img_paths = []
    for cam_type in ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]:
        img_paths.extend(glob.glob(os.path.join("data", "nuscenes", "samples", cam_type) + "/*.jpg"))

    if not args.no_shuffle:
        random.shuffle(img_paths)

    img_paths = img_paths[:args.num]
    os.makedirs(args.output_dir, exist_ok=True)
    for i, img in enumerate(img_paths):
        inputs, _ = dataloader.load(img)
        bev_inputs = model.get_bev_pool_input(inputs)

        onnx_inputs = {
            "img": inputs[0][0],
            "ranks_depth": bev_inputs[0],
            "ranks_feat": bev_inputs[1],
            "ranks_bev": bev_inputs[2],
            "n_points": bev_inputs[3],
        }

        np.save(os.path.join(args.output_dir, str(i) + ".npy"), onnx_inputs)
        print(f"Save data to {args.output_dir}. ({i + 1}/{len(img_paths)})")


if __name__ == "__main__":
    main()