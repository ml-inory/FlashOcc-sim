import os, sys, argparse
import cv2
import numpy as np
import json


class Dataloader:
    def __init__(self, data_root):
        self.data_root = data_root

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
                    'translation': calib_data['translation'],
                    'rotation': calib_data['rotation'],
                    'camera_intrinsic': calib_data['camera_intrinsic']
                }
        return None

    def load(self, filename):
        basename = os.path.basename(filename)
        sample_token = self.get_sample_token(filename)
        datas = self.get_sample_datas(sample_token)
        result = {}
        for data in datas:
            calibrated_sensor_token = data['calibrated_sensor_token']
            sensor_type = self.get_sensor_type(calibrated_sensor_token)
            data['sensor_type'] = sensor_type
            data.update(self.get_sensor_data(calibrated_sensor_token))
            data['filenmae'] = os.path.join(self.data_root, data['filename'])

            result[sensor_type] = data

            if basename in data['filename']:
                result['cur_type'] = sensor_type
        return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run FlashOcc with onnx')
    parser.add_argument('--config', help='test config file path', default='config.json')
    parser.add_argument('--onnx', help='onnx file', default='bevdet_fp16_fuse_for_c_and_trt.onnx')
    parser.add_argument('--data_root', help='data root', default='data/nuscenes/v1.0-mini')
    parser.add_argument('--img', help='image path', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.config) as f:
        model_config = json.load(f)

    dataloader = Dataloader(args.data_root)
    data = dataloader.load(args.img)
    print(data)


if __name__ == '__main__':
    main()