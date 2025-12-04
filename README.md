# CVMT：Complex-valued mix transformer for SAR ship detection
<p align="center">
<a href="https://doi.org/10.1016/j.isprsjprs.2025.10.018"><img src="https://img.shields.io/badge/Paper-ISPRS%20P%26RS-%23FF4203"></a>

</p>

This an official Pytorch implementation of our paper **"Complex-valued mix transformer for SAR ship detection"**. The specific details of the framework are as follows.

![img](https://github.com/RSIP-NJUPT/CVMT/blob/main/network_base.png)

- Notablely, CVMT is built on top of [MMDetection3.x](https://github.com/open-mmlab/mmdetection) with an added dual-stream data loading and processing mechanism. We hope that our approach and code will be helpful for your related research.

## ⚡News

- `Oct 17, 2025`: 🚀 CVMT is accepted to **ISPRS Journal of Photogrammetry and Remote Sensing**!

## Installation (●'◡'●)ﾉ
Please follow the following steps for installation.

Step 1: Create a conda environment

```shell
conda create --name CVMT python=3.8
conda activate CVMT
```

Step 2: Install PyTorch 1.10.1+CU113
```shell
# CUDA 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Step 3: Install OpenMMLab codebases
```shell
#Install MMEngine and MMCV using MIM
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0rc2,<3.2.0"
#Required packages
pip install timm
```

Step 4: Install `CVMT`
```shell
https://github.com/RSIP-NJUPT/CVMT.git
cd CVMT
pip install -e .
```
## Get Started (ง๑ •̀_•́)ง
Get Started with single GPU

Training CVMT, for example :

```shell
python tools/train.py ./mmdet/configs/CVMT/CVMT_4scale_r50_8xb2_12e_coco.py
```

Testing CVMT, for example :
```
python tools/test.py ./mmdet/configs/CVMT/CVMT_4scale_r50_8xb2_12e_coco.py path/to/your/checkpoint
```
## Data Preparation σ(`∀´)σ
- The [OpenSARShip](https://ieeexplore.ieee.org/document/8067489) & [FAIR-CSAR](https://radars.ac.cn/web/data/getData?dataType=FAIR_CSAR_en&pageType=en) or other complex-valued SAR datasets should be prepared as follows:
```
Dataset_root
├── train
│   ├── IT (Amplitude data)
│   ├── POS (Phase data)   
├── val
│   ├── IT (Amplitude data)
│   ├── POS (Phase data)
├── annotations
│   ├── annotations_train.json
│   ├── annotations_val.json
```
> Note: In this project, annotations from OpenSARShip (VOC format) and FAIR-CSAR (DOTA format) have been converted to the COCO-style `[x, y, w, h]` format.

## Pre-Trained Models ヾ(*´∀ ˋ*)ﾉ
You can leverage the pretrained weights we provide in this work as a starting point for your own fine‑tuning.The pretrained CVMT models are available on the [![models](https://img.shields.io/badge/BaiduNetdisk-Pretrained-blue.svg?logo=baidu&logoColor=white)](https://pan.baidu.com/s/1bfvJrRYLDo9U-pkqZGGWUA) or [![dataset](https://img.shields.io/badge/GoogleDrive-dataset-blue.svg?logo=GoogleDrive&logoColor=white)](https://drive.google.com/drive/folders/1X9jSZ1j9H5SovG5dWnQ0YT3VL9TdeqV2?usp=sharing).

**Link (BaiduNetdisk):** [https://pan.baidu.com/s/1bfvJrRYLDo9U-pkqZGGWUA](https://pan.baidu.com/s/1bfvJrRYLDo9U-pkqZGGWUA)

**Link (GoogleDrive):** [https://drive.google.com/drive/folders/1X9jSZ1j9H5SovG5dWnQ0YT3VL9TdeqV2?usp=sharing](https://drive.google.com/drive/folders/1X9jSZ1j9H5SovG5dWnQ0YT3VL9TdeqV2?usp=sharing)

**Extraction code (BaiduNetdisk):** iq95



## Performance ('ᴗ' )و

Table 1. Training Set: **FAIR-CSAR** trainval set, Testing Set: **FAIR-CSAR** test set, 12 epochs.
|Method | Backbone | mAP | mAP<sub>50</sub> | mAP<sub>75</sub> |mAP<sub>S</sub> | mAP<sub>M</sub>  | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---: |
DINO (Baseline) | R-50 | 28.9 | 67.4 | 20.1 | 29.9 | **31.4** | 
CVMT  | R-50 | **33.0** | **74.1** | **23.9** | **34.1** | 26.9 |

Table 2. Training Set: **OpenSARShip** train set, Testing Set: **OpenSARShip** test set, 36 epochs.
|Method | Backbone | mAP | mAP<sub>50</sub> | mAP<sub>75</sub> |mAP<sub>S</sub> | mAP<sub>M</sub>  | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---: |
DINO (Baseline) | R-50 | 74.6 | 88.2 | 78.6 | 74.8 | 70.0 | 
CVMT  | R-50 | **77.9** | **89.5** | **81.3** | **77.3** | **90.6** |

## 📘Citation
If you find our work helpful, please consider citing our paper and giving us a ⭐! Thank you very much!

```bibtex
@article{NI20261,
title = {Complex-valued mix transformer for SAR ship detection},
author = {Kang Ni and Pengcheng Wang and Zhizhong Zheng and Yanfei Zhong},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
year = {2026},
doi = {https://doi.org/10.1016/j.isprsjprs.2025.10.018}
}
```

