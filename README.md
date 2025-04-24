# Medical Instance Segmentation by Mask R-CNN

A project for medical instance segmentation using a ResNet-50-FPN based Mask R-CNN.

Stu: Ping-Yeh Chou 113550901

## Introduction

This repository implements an instance segmentation pipeline for medical images using Mask R-CNN with a ResNet-50-FPN backbone. The goal is to detect and segment different types of medical objects in images, supporting multi-class instance segmentation. The code is designed for flexibility and reproducibility, supporting configurable training and inference via command-line arguments.

Actually, instance segmentation of medical images is a challenging task due to the complex morphology and subtle boundaries of different cell types. In this work, the goal is to accurately segment four categories of cells from colored microscopy images, providing precise masks for each instance. To address this, I adopt Mask R-CNN as the baseline framework, which is well-suited for instance segmentation tasks. To further enhance the model’s ability to capture discriminative features and improve segmentation accuracy, I integrate a Feature Pyramid Network (FPN) backbone to better handle multi-scale information, and incorporate Convolutional Block Attention Module (CBAM) into the backbone. The CBAM attention mechanism adaptively refines feature representations by focusing on informative regions both channel-wise and spatially, which is particularly beneficial for distinguishing overlapping or visually similar cells. This combination aims to boost the model’s sensitivity to relevant structures in medical images, ultimately leading to more accurate and robust cell segmentation results.

## How to Install

1. **Clone the repository:**
```bash
git clone https://github.com/cloud-zhoubingye/Medocal_Instance_Segmentation.git
cd <repo-folder>
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare the dataset:**
Place the dataset in the folder specified by `--data_root` (default: `hw3-data-release`).

Download it [here](https://drive.google.com/file/d/1B0qWNzQZQmfQP7x7o4FDdgb9GvPDoFzI/view?pli=1).

## How to Run

### Training and Inference

```bash
python main.py
```

## Command-line Arguments

All arguments can be set via the command line. Below is a summary of the available options (see `argsparse.py` for details):

| Argument                   | Type    | Default            | Description                                 |
|----------------------------|---------|--------------------|---------------------------------------------|
| `--data_root`              | str     | hw3-data-release   | Path to the dataset root                    |
| `--output_dir`             | str     | results            | Directory to save results                   |
| `--batch_size`             | int     | 3                  | Batch size for data loaders                 |
| `--num_epochs`             | int     | 1                  | Number of training epochs                   |
| `--lr_max`                 | float   | 2e-4               | Maximum learning rate                       |
| `--lr_min`                 | float   | 5e-6               | Minimum learning rate                       |
| `--weight_decay`           | float   | 1e-4               | Weight decay for optimizer                  |
| `--num_classes`            | int     | 5                  | Number of classes (including background)    |
| `--hidden_layer`           | int     | 256                | Size of the hidden layer                    |
| `--trainable_backbone_layers` | int  | 3                  | Number of trainable backbone layers         |
| `--score_threshold`        | float   | 0.5                | Score threshold for predictions             |
| `--nms_threshold`          | float   | 0.5                | NMS threshold for predictions               |

You can override any of these defaults by specifying them on the command line.


## Results
![alt text](results.png)
