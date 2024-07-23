import os
import numpy as np
import json
import torch
from PIL import Image
from pyquaternion import Quaternion


class Dataloader:
    def __init__(self, data_root, config):
        self.data_root = data_root
        self.config = config

        json_root = os.path.join(self.data_root, 'v1.0-mini')
        calibrated_sensor_path = os.path.join(json_root, 'calibrated_sensor.json')
        ego_pose_path = os.path.join(json_root, 'ego_pose.json')
        sample_data_path = os.path.join(json_root, 'sample_data.json')
        sensor_path = os.path.join(json_root, 'sensor.json')

        with open(calibrated_sensor_path) as f:
            self.calibrated_sensor = json.load(f)

        with open(ego_pose_path) as f:
            self.ego_pose = json.load(f)

        with open(sample_data_path) as f:
            self.sample_data = json.load(f)

        with open(sensor_path) as f:
            self.sensor = json.load(f)

    def get_sample_token(self, filename):
        basename = os.path.basename(filename)
        for data in self.sample_data:
            if basename in data['filename']:
                return data['sample_token']
        return None

    def get_sample_datas(self, sample_token):
        datas = []
        for data in self.sample_data:
            if data['sample_token'] == sample_token:
                datas.append(data)
        return datas

    def get_sensor_type(self, calibrated_sensor_token):
        for calib_data in self.calibrated_sensor:
            if calib_data['token'] == calibrated_sensor_token:
                token = calib_data['sensor_token']
                for sensor_data in self.sensor:
                    if token == sensor_data['token']:
                        return sensor_data['channel']
        return None

    def get_sensor_data(self, calibrated_sensor_token):
        for calib_data in self.calibrated_sensor:
            if calib_data['token'] == calibrated_sensor_token:
                return {
                    'sensor2ego_translation': calib_data['translation'],
                    'sensor2ego_rotation': calib_data['rotation'],
                    'camera_intrinsic': calib_data['camera_intrinsic']
                }
        return None
    
    def get_ego_data(self, ego_pose_token):
        for data in self.ego_pose:
            if data['token'] == ego_pose_token:
                return {
                    'ego2global_translation': data['translation'],
                    'ego2global_rotation': data['rotation']
                }
        return None

    def load(self, filename):
        basename = os.path.basename(filename)
        sample_token = self.get_sample_token(filename)
        datas = self.get_sample_datas(sample_token)
        info = {}
        for data in datas:
            calibrated_sensor_token = data['calibrated_sensor_token']
            ego_pose_token = data['ego_pose_token']

            sensor_type = self.get_sensor_type(calibrated_sensor_token)
            data['sensor_type'] = sensor_type
            data.update(self.get_sensor_data(calibrated_sensor_token))
            data.update(self.get_ego_data(ego_pose_token))
            data['filename'] = os.path.join(self.data_root, data['filename'])

            info[sensor_type] = data

            if basename in data['filename']:
                info['cur_type'] = sensor_type

        return self.prepare_image_inputs(info), info
    
    def get_sensor_transforms(self, cam_data):
        w, x, y, z = cam_data['sensor2ego_rotation']      # 四元数格式
        # sensor to ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        sensor2ego_tran = torch.Tensor(
            cam_data['sensor2ego_translation'])   # (3, )
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran

        # ego to global
        w, x, y, z = cam_data['ego2global_rotation']      # 四元数格式
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        ego2global_tran = torch.Tensor(
            cam_data['ego2global_translation'])   # (3, )
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global
    
    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.config['input_size']
        
        resize = float(fW) / float(W)
        if scale is not None:
            resize += scale
        else:
            resize += self.config['resize_test']
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(self.config['crop_h'])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False if flip is None else flip
        rotate = 0
        return resize, resize_dims, crop, flip, rotate
    
    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])
    
    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        """
        Args:
            img: PIL.Image
            post_rot: torch.eye(2)
            post_tran: torch.eye(2)
            resize: float, resize的比例.
            resize_dims: Tuple(W, H), resize后的图像尺寸
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: bool
            rotate: float 旋转角度
        Returns:
            img: PIL.Image
            post_rot: Tensor (2, 2)
            post_tran: Tensor (2, )
        """
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        # 将上述变换以矩阵表示.
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran
    
    def normalize_img(self, img):
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img = np.array(img, dtype=np.float32)[..., ::-1]
        for i in range(3):
            img[..., i] = (img[..., i] - mean[i]) / std[i]
        img = torch.tensor(img.copy()).float().permute(2, 0, 1).contiguous()
        return img
    
    def prepare_image_inputs(self, info):
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam_name in self.config['cams']:
            cam_data = info[cam_name]
            img = Image.open(cam_data['filename'])
            
            # 初始化图像增广的旋转和平移矩阵
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            # 当前相机内参
            intrin = torch.Tensor(cam_data['camera_intrinsic'])

            # 获取当前相机的sensor2ego(4x4), ego2global(4x4)矩阵.
            sensor2ego, ego2global = \
                self.get_sensor_transforms(cam_data)

            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width)
            resize, resize_dims, crop, flip, rotate = img_augs

            # img: PIL.Image;  post_rot: Tensor (2, 2);  post_tran: Tensor (2, )
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            # 以3x3矩阵表示图像的增广
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(self.normalize_img(img))
            intrins.append(intrin)      # 相机内参 (3, 3)
            sensor2egos.append(sensor2ego)      # camera2ego变换 (4, 4)
            ego2globals.append(ego2global)      # ego2global变换 (4, 4)
            post_rots.append(post_rot)          # 图像增广旋转 (3, 3)
            post_trans.append(post_tran)        # 图像增广平移 (3, ）

        imgs = torch.stack(imgs).numpy()[np.newaxis, ...]    # (N_views, 3, H, W)        # N_views = 6 * (N_history + 1)

        sensor2egos = torch.stack(sensor2egos)      # (N_views, 4, 4)
        ego2globals = torch.stack(ego2globals)      # (N_views, 4, 4)
        intrins = torch.stack(intrins)              # (N_views, 3, 3)
        post_rots = torch.stack(post_rots)          # (N_views, 3, 3)
        post_trans = torch.stack(post_trans)        # (N_views, 3)
        bda = self.bev_transform()

        return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda
    
    def bev_transform(self, rotate_angle=0, scale_ratio=1, flip_dx=False,
                      flip_dy=False):
        """
        Args:
            gt_boxes: (N, 9)
            rotate_angle:
            scale_ratio:
            flip_dx: bool
            flip_dy: bool

        Returns:
            gt_boxes: (N, 9)
            rot_mat: (3, 3）
        """
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:     # 沿着y轴翻转
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:     # 沿着x轴翻转
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)    # 变换矩阵(3, 3)
        return rot_mat