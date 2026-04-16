# Infrared-Guided Smoke-Occluded Image Restoration

## Overview

This repository implements a dual-branch encoder architecture for infrared-guided visible light image restoration. The model is designed to restore smoke-occluded visible light images using pixel-aligned infrared images as structural guidance.

### Task Description

**Input:**
- `trainA`: Smoke-occluded visible light images (masked RGB images)
- `trainA_guide`: Pixel-aligned infrared images (structural guidance)

**Output:**
- `trainB`: Restored complete visible light images

**Model:** Guided Pix2Pix with Dual-Branch Architecture

### Key Features

- **Dual-Branch Encoder**: Separate encoders for visible light and infrared structure extraction
- **Cross-Modal Guidance Module**: Spatial adaptive gating fusion with structure-aware attention
- **Structure Extraction**: Depth-wise separable convolutions with asymmetric receptive fields
- **Multi-Scale Feature Fusion**: Skip connections with guided feature integration

## Citation

If you use this code for your research, please cite:

```
[Your paper citation will go here]
```

This work is based on the pix2pix framework:

```
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```


## Prerequisites

- Linux or Windows
- Python 3.7+
- PyTorch 1.4+
- NVIDIA GPU + CUDA CuDNN (recommended)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/quanyushi/Infrared-Guided-Smoke-Occluded-Image-Restoration.git
cd Infrared-Guided-Smoke-Occluded-Image-Restoration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda env create -f environment.yml
conda activate infrared_guided_restoration
```

## Dataset

### Download Dataset

We provide an open-source dataset for infrared-guided smoke-occluded image restoration:

**Download Link (Baidu Netdisk):**
- Link: https://pan.baidu.com/s/1y7qn8xzWqT5pCuVfkUJiqQ
- Extraction Code: `ifqk`
- File: `opensource_dataset.zip`

After downloading, extract the dataset and place it in the `datasets/` folder.

### Dataset Preparation

Organize your dataset in the following structure:

```
datasets/your_dataset/
├── train/
│   ├── trainA/          # Smoke-occluded visible light images
│   ├── trainA_guide/    # Pixel-aligned infrared images
│   └── trainB/          # Ground truth complete visible light images
└── test/
    ├── testA/           # Test smoke-occluded images
    ├── testA_guide/     # Test infrared images
    └── testB/           # Test ground truth images
```

**Important:** Images in `trainA`, `trainA_guide`, and `trainB` must be pixel-aligned and have the same filename.

## Training

Train the model with default settings:

```bash
python train.py \
    --dataroot ./datasets/your_dataset \
    --name experiment_name \
    --model guided_pix2pix \
    --netG dual_branch_unet \
    --dataset_mode guided \
    --batch_size 4 \
    --gpu_ids 0
```

Key training parameters:

- `--lambda_L1`: Weight for L1 reconstruction loss (default: 50.0)
- `--lambda_structure`: Weight for structure guidance loss (default: 5.0)
- `--visible_grad_weight`: Gradient weight for visible branch (default: 1.0)
- `--infrared_grad_weight`: Gradient weight for infrared branch (default: 0.5)
- `--n_epochs`: Number of epochs with initial learning rate (default: 100)
- `--n_epochs_decay`: Number of epochs to linearly decay learning rate (default: 100)

Monitor training progress:
```bash
python -m visdom.server
```
Then navigate to `http://localhost:8097` in your browser.

## Testing

Test the trained model:

```bash
python test.py \
    --dataroot ./datasets/your_dataset \
    --name experiment_name \
    --model guided_pix2pix \
    --netG dual_branch_unet \
    --dataset_mode guided \
    --phase test
```

Results will be saved to `./results/experiment_name/test_latest/`.

## Architecture Validation

Verify the dual-branch architecture:

```bash
python test_dual_branch.py
```

This will test:
- Individual encoder components
- Cross-modal guidance module
- Complete generator forward/backward pass
- Loss computation

## Code Structure

```
├── data/
│   ├── guided_dataset.py      # Dataset loader for guided reconstruction
│   ├── base_dataset.py        # Base dataset class
│   └── image_folder.py        # Image loading utilities
├── models/
│   ├── guided_pix2pix_model.py  # Main model implementation
│   ├── networks.py              # Network architectures
│   │   ├── DualBranchUnetGenerator
│   │   ├── VisibleLightEncoder
│   │   ├── StructureExtractionEncoder
│   │   ├── CrossModalGuidanceModule
│   │   └── FeatureFusionDecoder
│   └── base_model.py          # Base model class
├── options/
│   ├── base_options.py        # Base options
│   ├── train_options.py       # Training options
│   └── test_options.py        # Testing options
├── util/
│   ├── visualizer.py          # Visualization utilities
│   └── util.py                # Helper functions
├── train.py                   # Training script
├── test.py                    # Testing script
└── debug_training.py          # Training debugging utilities
```

## Tips

- Start with a small learning rate (0.0002) and adjust based on training stability
- Monitor the balance between GAN loss, L1 loss, and structure loss
- Use gradient clipping to prevent training instability
- Adjust `lambda_structure` based on how much structural guidance you need
- For high-resolution images, reduce batch size to fit in GPU memory

## Troubleshooting

**Training instability:**
- Reduce learning rate
- Adjust gradient weights
- Check for NaN values in losses

**Out of memory:**
- Reduce batch size
- Reduce image resolution
- Use gradient checkpointing

**Poor reconstruction quality:**
- Increase `lambda_L1`
- Adjust `lambda_structure`
- Check dataset alignment

## Acknowledgments

This code is built upon the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) framework by Jun-Yan Zhu and Taesung Park.
