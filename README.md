# FlashOcc-sim

[English](README_EN.md)
 
本项目旨在简化[FlashOcc的官方repo](https://github.com/Yzichen/FlashOCC)，不再依赖openmmlab的环境，目前只实现了推理和模型转换的流程。


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


## 模型转换(PyTorch->ONNX)

已提前转换好，从[这里](https://github.com/ml-inory/FlashOcc-sim/releases/download/v1.0/bevdet_ax.onnx)下载

如果需要转换自己修改的模型，请参考pth_model.py和export_onnx.py，目前只支持flashocc-r50-M0的配置。
将官方repo的[checkpoint](https://drive.google.com/file/d/14my3jdqiIv6VIrkozQ6-ruEcBOPVlWGJ/view?usp=sharing)下载到根目录，
运行`python export_onnx.py`即可得到bevdet_ax.onnx


## 模型转换(ONNX->AX650)

已提前转换好，从[这里](https://github.com/ml-inory/FlashOcc-sim/releases/download/v1.0/flashocc_argmax.axmodel)下载

flashocc_axera.json 为转换板上模型所用的配置文件，其中的量化校准数据集需要通过脚本生成。  

运行`python gen_calib_dataset.py`可得到calib_dataset文件夹，此为用于转换AX650板上模型的量化数据集，将其压缩成zip档，修改flashocc_axera.json中的calibration_dataset为此zip档的路径。

随后通过pulsar2工具链即可得到板端模型。


## 运行

`python main.py --img 图片路径`


## 示例

`python main.py --img data\\nuscenes\samples\\CAM_BACK\\n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg`

运行完成后会在根目录生成sementics.jpg和vis_result文件夹下的图片，如下：

![sementics](/fig/sementics.jpg)

![overall](/fig/overall.png)


## Acknowledgement

感谢[FlashOcc官方repo](https://github.com/Yzichen/FlashOCC) 和 [Jackpot233333的fork](https://github.com/Jackpot233333/FlashOCC/tree/export_for_axera)