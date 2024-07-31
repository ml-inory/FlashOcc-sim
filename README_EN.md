# FlashOcc-sim
 
This repo aims for simplifying [the official FlashOCC repo](https://github.com/Yzichen/FlashOCC) by removing openmmlab dependencies. Currently only inference and model export is implemented.


## Environment Setup

`pip install -r requirements.txt`


## Dataset

Download nuscenes's [mini dataset](https://www.nuscenes.org/data/v1.0-mini.tgz)

Extract to data/nuscenes so that folders ordered as follows：

data  
-- nuscenes  
---- maps  
---- samples  
---- sweeps  
---- v1.0-mini  


## Model Convert(PyTorch->ONNX)

ONNX model could be downloaded [here](https://github.com/ml-inory/FlashOcc-sim/releases/download/v1.0/bevdet_ax.onnx)

If you need to convert your own flashocc model，please refer to pth_model.py and export_onnx.py，currently only config of flashocc-r50-M0 is supported.
Download [checkpoint](https://drive.google.com/file/d/14my3jdqiIv6VIrkozQ6-ruEcBOPVlWGJ/view?usp=sharing) to root folder，
Run `python export_onnx.py` and you will get bevdet_ax.onnx


## Inference

`python main.py --img 图片路径`


## Example

`python main.py --img data\\nuscenes\samples\\CAM_BACK\\n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg`

After inference is finished, sementics.jpg and images in vis_result folder will be generated, here is an example:

![sementics](/fig/sementics.jpg)

![overall](/fig/overall.png)


## Acknowledgement

Thanks to [FlashOcc official repo](https://github.com/Yzichen/FlashOCC) and [fork from Jackpot233333](https://github.com/Jackpot233333/FlashOCC/tree/export_for_axera)