# Flow-Guided Feature Aggregation for Video Object Detection

__This repo provides an up-to-date implementation of FGFA which is accuracy stable and supports batch size >1.__

## Introduction

[Paper](https://arxiv.org/abs/1703.10025) proposes a flow guided feature aggregation method for video object detection, and it has been a vital baseline for subsequent researches. However, the code attached to the paper is based on MXNet, which seldom updates anymore.

[MEGA Repo](https://github.com/Scalsol/mega.pytorch) provides a torch 1.3 and CUDA 9,10 series based implementation, which is too old.

[MMTracking](https://github.com/open-mmlab/mmtracking) provides a nice up-to-date implementation, but we experienced some problems below:

1.We train FGFA on ImageNet VID datasets and our custom datasets, the accuracy is oddly low. We solved this problem by replacing the mmtracking's flow net with  [MEGA Repo](https://github.com/Scalsol/mega.pytorch)'s flow net, and the accuracy becomes right.

2.The batch size is force constrained to 1,  which may lead to not stable training and resource waste for large memory GPU. Also, the train speed is low. We solved this probelm by re-implementing the FGFA completely and  some constrains in the mmtracking framework were eliminated.

__Now, This repo provides an up-to-date implementation of FGFA which is accuracy stable and supports batch size >1.__ This repo is based on MMTracking `0.14.0`. We tested on torch `1.10` and CUDA `11.3`, it works well. Other versions may also work.

## Installation, Train, Inference, etc.

The revised FGFA config files are placed in `configs/vid/fgfa_bm`. Pretrained flow net weights is shown in Releases, download it and put it in `pretrain/` folder. To use this code, you should be familiar with OpenMMLab's [MMTracking](https://github.com/open-mmlab/mmtracking).

For other details, please refer to README_mmtracking.md and [MMTracking](https://github.com/open-mmlab/mmtracking) repo.

