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

已提前转换好，从[这里](https://github.com/ml-inory/FlashOcc-sim/releases/download/v1.0/bevdet_ax.onnx)下载

## 运行

`python main.py --img 图片路径`

## 示例

`python main.py --img data\\nuscenes\samples\\CAM_BACK\\n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg`

运行完成后会在根目录生成sementics.jpg和vis_result文件夹下的图片，如下：

![sementics](/fig/sementics.jpg)

![overall](/fig/overall.png)