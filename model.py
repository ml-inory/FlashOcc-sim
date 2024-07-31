import numpy as np
import json
import torch
from PIL import Image
import onnxruntime, onnx
from onnxruntime_extensions import PyCustomOpDef, onnx_op, get_library_path


@onnx_op(op_type="AxBevPool",
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_int32, PyCustomOpDef.dt_int32, PyCustomOpDef.dt_int32, PyCustomOpDef.dt_int32],
         outputs=[PyCustomOpDef.dt_float],
        #  attrs={"output_width": PyCustomOpDef.dt_int32, "output_height": PyCustomOpDef.dt_int32, "output_z": PyCustomOpDef.dt_int32}
         )
def AxBevPool(depth, feat, ranks_depth, ranks_feat, ranks_bev, n_points):
    # output: 1x200x200x64
    # return np.zeros((1, 1, 200, 200, 64), dtype=np.float32)

    if len(depth.shape) < 5:
        depth = depth.unsqueeze(0)
        feat = feat.unsqueeze(0)

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

    B, N, _, iH, iW = depth.shape
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
    r = r_scatter.reshape(B, oD, oW, oH, C).numpy()

    return r


class LSSViewTransformer:
    def __init__(self, 
            config, 
            downsample=16):
        self.config = config
        grid_config = config['grid_config']

        self.downsample = downsample
        self.sid = False
        self.create_grid_infos(**grid_config)
        self.frustum = self.create_frustum(grid_config['depth'],
                                           config['input_size'], downsample)      # (D, fH, fW, 3)  3:(u, v, d)


    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])     # (min_x, min_y, min_z)
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])        # (dx, dy, dz)
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])                   # (Dx, Dy, Dz)


    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        Returns:
            frustum: (D, fH, fW, 3)  3:(u, v, d)
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)      # (D, fH, fW)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
                              torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)

        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)      # (D, fH, fW)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)      # (D, fH, fW)

        return torch.stack((x, y, d), -1)    # (D, fH, fW, 3)  3:(u, v, d)


    def get_lidar_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                       bda):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points
    
    def voxel_pooling_prepare_ax(self, coor):
        """Data preparation for voxel pooling.
        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).
        Returns:
            tuple[torch.tensor]:
                ranks_bev: Rank of the voxel that a point is belong to in shape (N_points, ),
                    rank介于(0, B*Dx*Dy*Dz-1).
                ranks_depth: Reserved index of points in the depth space in shape (N_Points),
                    rank介于(0, B*N*D*fH*fW-1).
                ranks_feat: Reserved index of points in the feature space in shape (N_Points),
                    rank介于(0, B*N*fH*fW-1).
                interval_starts: (N_pillar, )
                interval_lengths: (N_pillar, )
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.arange(
            0, num_points, dtype=torch.int, device=coor.device)    # (B*N*D*H*W, ), [0, 1, ..., B*N*D*fH*fW-1]
        ranks_feat = torch.arange(
            0, num_points // D, dtype=torch.int, device=coor.device)   # [0, 1, ...,B*N*fH*fW-1]
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()     # (B*N*D*fH*fW, )

        # convert coordinate into the voxel space
        # ((B, N, D, fH, fW, 3) - (3, )) / (3, ) --> (B, N, D, fH, fW, 3)   3:(x, y, z)  grid coords.
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)      # (B, N, D, fH, fW, 3) --> (B*N*D*fH*fW, 3)
        # print(f"coor = {coor}")
        # (B, N*D*fH*fW) --> (B*N*D*fH*fW, 1)
        batch_idx = torch.arange(0, B).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)      # (B*N*D*fH*fW, 4)   4: (x, y, z, batch_id)

        # filter out points that are outside box
        # print(f"self.grid_size = {self.grid_size}")
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None

        # (N_points, 4), (N_points, ), (N_points, )
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        
        # print(f"coor = {coor}")

        # get tensors from the same voxel next to each other
        # print(f"coor[:, 3] = {coor[:, 3]}")
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        # print(f"ranks_bev = {ranks_bev}")
        order = ranks_bev.argsort()
        # (N_points, ), (N_points, ), (N_points, )
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]
        # print(f"sorted ranks_bev = {ranks_bev}")

        n_points = len(ranks_bev)
        ranks_ones = torch.ones(num_points - n_points, dtype=torch.int, device=coor.device)
        ranks_depth = torch.cat((ranks_depth, ranks_ones * num_points), 0)
        ranks_feat = torch.cat((ranks_feat, ranks_ones * (num_points // D)), 0)
        ranks_bev = torch.cat((ranks_bev, ranks_ones * 0), 0)

        return ranks_depth.int().contiguous().numpy(), ranks_feat.int().contiguous().numpy(), ranks_bev.int().contiguous().numpy(), torch.tensor([n_points]).int().numpy()


class Model:
    def __init__(self, onnx_file, config):
        so = onnxruntime.SessionOptions()
        so.register_custom_ops_library(get_library_path())

        self.onnx = onnxruntime.InferenceSession(onnx_file, so)
        self.config = config
        self.img_view_transformer = LSSViewTransformer(config)

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from adj sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)    # (B, 1, 4, 4)
        global2keyego = torch.inverse(keyego2global.double())   # (B, 1, 4, 4)
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_views, 4, 4)
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def get_bev_pool_input(self, inputs):
        inputs = self.prepare_inputs(inputs)
        coor = self.img_view_transformer.get_lidar_coor(*inputs[1:7])
        return self.img_view_transformer.voxel_pooling_prepare_ax(coor)
    
    def forward(self, inputs):
        return self.onnx.run([], inputs)[0].astype(np.int32)[0]
    

if __name__ == "__main__":
    depth_1d = np.fromfile("depth_1d.bin", dtype=np.float32)
    ranks_depth = np.fromfile("ranks_depth.bin", dtype=np.int32)

    depth_1d = torch.from_numpy(depth_1d)
    ranks_depth = torch.from_numpy(ranks_depth.astype(np.int64))

    gathered = torch.gather(input=depth_1d, dim=0, index=ranks_depth)