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
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])


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


def save_onnx_outputs(onnx_outputs, name="cls_occ_label.bin"):
    os.makedirs("outputs", exist_ok=True)
    onnx_outputs.tofile(os.path.join("outputs", name))


def AxQuantizedBevPool(depth, feat, ranks_depth, ranks_feat, ranks_bev, n_points):
    # output: 1x200x200x64
    # return np.zeros((1, 200, 200, 64), dtype=np.float32)

    depth = torch.from_numpy(depth.astype(np.float32))
    feat = torch.from_numpy(feat.astype(np.float32))

    depth_scale = 0.003921568859368563
    feat_scale = 0.4500899910926819
    r_scale = 0.005210042465478182
    depth_zp = 0
    feat_zp = 88
    r_zp = 32003

    ranks_depth = torch.from_numpy(ranks_depth.astype(np.int64))
    ranks_feat = torch.from_numpy(ranks_feat.astype(np.int64))
    ranks_bev = torch.from_numpy(ranks_bev.astype(np.int64))
    n_points = torch.from_numpy(n_points)
    ranks_depth = ranks_depth[:n_points]
    ranks_feat = ranks_feat[:n_points]
    ranks_bev = ranks_bev[:n_points]

    bev_feat_shape = (1, 1, 200, 200, 64)
    output_dtype = np.uint16

    B, N, _, iH, iW = depth.shape
    
    C = feat.shape[-1]
    _, oD, oH, oW, _ = bev_feat_shape

    # flatten inputs
    depth_1d = depth.flatten()
    feat_2d = feat.reshape(B * N * iH * iW, C)

    # gather depth and feat
    gathered_depth_1d = torch.gather(input=depth_1d, dim=0, index=ranks_depth)
    ranks_feat = ranks_feat.reshape(ranks_feat.shape[0], 1).repeat(1, C)
    gathered_feat = torch.gather(input=feat_2d, dim=0, index=ranks_feat)

    # subtract zp and mul
    gathered_depth_2d = gathered_depth_1d.reshape(gathered_depth_1d.shape[0], 1)
    r_mul = (gathered_depth_2d - depth_zp) * (gathered_feat - feat_zp)

    # init with zeros
    r_scatter = torch.full(fill_value=0, size=(B * oD * oH * oW, C), dtype=torch.float32)

    # scatter_add
    ranks_bev = ranks_bev.reshape(ranks_bev.shape[0], 1).repeat(1, C)
    r_scatter = torch.scatter_add(input=r_scatter, dim=0, index=ranks_bev, src=r_mul.float())

    # quant
    r_quant = r_scatter * depth_scale * feat_scale / r_scale + r_zp

    # reshape
    r = r_quant.reshape(B, oD, oH, oW, C).numpy()
    r = np.round(r).clip(np.iinfo(output_dtype).min, np.iinfo(output_dtype).max).astype(output_dtype)
    return r

def AxBevPool(depth, feat, ranks_depth, ranks_feat, ranks_bev, n_points):
    # output: 1x200x200x64
    # return np.zeros((1, 200, 200, 64), dtype=np.float32)

    depth = torch.from_numpy(depth.astype(np.float32))
    feat = torch.from_numpy(feat.astype(np.float32))

    ranks_depth = torch.from_numpy(ranks_depth.astype(np.int64))
    ranks_feat = torch.from_numpy(ranks_feat.astype(np.int64))
    ranks_bev = torch.from_numpy(ranks_bev.astype(np.int64))
    n_points = torch.from_numpy(n_points)
    ranks_depth = ranks_depth[:n_points]
    ranks_feat = ranks_feat[:n_points]
    ranks_bev = ranks_bev[:n_points]

    bev_feat_shape = (1, 1, 200, 200, 64)
    output_dtype = np.float32

    N, _, iH, iW = depth.shape
    B = 1
    C = feat.shape[-1]
    _, oD, oH, oW, _ = bev_feat_shape

    # flatten inputs
    depth_1d = depth.flatten()
    feat_2d = feat.reshape(B * N * iH * iW, C)

    # gather depth and feat
    gathered_depth_1d = torch.gather(input=depth_1d, dim=0, index=ranks_depth.long())
    ranks_feat = ranks_feat.reshape(ranks_feat.shape[0], 1).repeat(1, C)
    gathered_feat = torch.gather(input=feat_2d, dim=0, index=ranks_feat.long())

    # subtract zp and mul
    gathered_depth_2d = gathered_depth_1d.reshape(gathered_depth_1d.shape[0], 1)
    r_mul = gathered_depth_2d * gathered_feat

    # init with zeros
    r_scatter = torch.full(fill_value=0, size=(B * oD * oW * oH, C), dtype=torch.float32, device=r_mul.device)

    # scatter_add
    ranks_bev = ranks_bev.reshape(ranks_bev.shape[0], 1).repeat(1, C)
    r_scatter = torch.scatter_add(input=r_scatter, dim=0, index=ranks_bev.long(), src=r_mul)

    # reshape
    r = r_scatter.reshape(B, oD, oW, oH, C)
    return r.numpy()


def vis_occ(semantics):
    print(semantics.shape)
    # simple visualization of result in BEV
    semantics_valid = np.logical_not(semantics == 17)
    d = np.arange(16).reshape(1, 1, 16)
    d = np.repeat(d, 200, axis=0)
    d = np.repeat(d, 200, axis=1).astype(np.float32)
    d = d * semantics_valid
    selected = np.argmax(d, axis=2)

    selected_torch = torch.from_numpy(selected)
    semantics_torch = torch.from_numpy(semantics)

    occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                    index=selected_torch.unsqueeze(-1))
    occ_bev = occ_bev_torch.numpy()

    occ_bev = occ_bev.flatten().astype(np.int32)
    occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
    occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
    occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
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
        "n_points": bev_inputs[3]
    }
    save_onnx_inputs(onnx_inputs)

    # print(f"ranks_depth: {bev_inputs[0]}")
    # print(f"ranks_feat: {bev_inputs[1]}")
    # print(f"ranks_bev: {bev_inputs[2]}")
    # print(f"n_points: {bev_inputs[3]}")

    # onnx_inputs = {
    #     "img": np.fromfile("flashocc/input/0/img.bin", dtype=np.float32).reshape((6, 3, 256, 704)),
    #     "ranks_depth": np.fromfile("flashocc/input/0/ranks_depth.bin", dtype=np.int32),
    #     "ranks_feat": np.fromfile("flashocc/input/0/ranks_feat.bin", dtype=np.int32),
    #     "ranks_bev": np.fromfile("flashocc/input/0/ranks_bev.bin", dtype=np.int32),
    #     "n_points": np.fromfile("flashocc/input/0/n_points.bin", dtype=np.int32),
    # }

    inputs_data = np.load("bev_inputs_data.npy", allow_pickle=True).tolist()
    # print(f"ranks_depth: {inputs_data['ranks_depth']}")
    # print(f"ranks_feat: {inputs_data['ranks_feat']}")
    # print(f"ranks_bev: {inputs_data['ranks_bev']}")
    # print(f"n_points: {inputs_data['n_points']}")
    print(inputs_data['depth'])
    # r = AxBevPool(**inputs_data)
    outputs_data = np.load("bev_outputs_data.npy", allow_pickle=True).tolist()["r"]

    depth_scale = 0.003921568859368563
    feat_scale = 0.4500899910926819
    r_scale = 0.005210042465478182
    depth_zp = 0
    feat_zp = 88
    r_zp = 32003
    inputs_data["depth"] = (inputs_data["depth"][0].astype(np.float32) - depth_zp) * depth_scale
    inputs_data["feat"] = (inputs_data["feat"][0].astype(np.float32) - feat_zp) * feat_scale
    # inputs_data["depth"] = inputs_data["depth"][0].astype(np.float32)
    # inputs_data["feat"] = inputs_data["feat"][0].astype(np.float32)
    outputs_data = (outputs_data.astype(np.float32) - r_zp) * r_scale
    
    # r = AxBevPool(**inputs_data)

    onnx_outputs = model.forward({"317": outputs_data[0]})

    # onnx_outputs = model.forward(inputs_data)
    # save_onnx_outputs(onnx_outputs, "nobev_result.bin")
    # print(f"cls_occ_label: {onnx_outputs}")

    result = vis_occ(onnx_outputs)
    cv2.imshow("result", result)
    cv2.waitKey(0)

    # gt_outputs = np.fromfile("flashocc/output/0/cls_occ_label.bin", dtype=np.int32).reshape((1, 200, 200, 16))
    # visualize(onnx_outputs, info)

    # print(np.sum(onnx_outputs == gt_outputs))


if __name__ == '__main__':
    main()