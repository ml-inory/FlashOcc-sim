# FlashOcc-sim
 
本项目用于验证Axera的FlashOCC推理过程，基于[Axera的FlashOCC Fork](https://github.com/Jackpot233333/FlashOCC/tree/export_for_axera)，不再依赖openmmlab的环境，只用onnx推理。

## 环境准备

`pip install -r requirements.txt`

## 数据集

下载nuscenes的[mini数据集](https://www.nuscenes.org/data/v1.0-mini.tgz)

解压到工程根目录的data/nuscenes下，使最终的目录结构如下：

data  
-- nuscenes  
---- maps
---- samples
---- sweeps
---- v1.0-mini

## 模型

已提前转换好，从[这里]()下载

## 运行

`python main.py --img 图片路径`