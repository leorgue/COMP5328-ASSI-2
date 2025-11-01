# Training Scripts for Noisy Label Learning

This directory contains 4 separate training scripts for handling noisy labels in CIFAR dataset using different approaches.

## Scripts Overview

1. **train_resnet_baseline.py** - Standard ResNet18 without noise correction
2. **train_forward_correction.py** - Forward loss correction using transition matrix
3. **train_backward_correction.py** - Backward loss correction using inverse transition matrix
4. **train_coteaching.py** - Co-Teaching approach with two networks

## Requirements

```bash
pip install numpy torch torchvision pandas tqdm
```

## Usage

### 1. Baseline ResNet18

```bash
python train_resnet_baseline.py --data_path data/CIFAR.npz --epochs 15 --batch_size 64
```

**Outputs:**
- Model: `resnet_baseline.pth`
- Results: `resnet_baseline_results.csv`

### 2. Forward Loss Correction

```bash
python train_forward_correction.py --data_path data/CIFAR.npz --epochs 15 --batch_size 64
```

**Outputs:**
- Model: `resnet_forward.pth`
- Results: `resnet_forward_results.csv`

### 3. Backward Loss Correction

```bash
python train_backward_correction.py --data_path data/CIFAR.npz --epochs 15 --batch_size 64
```

**Outputs:**
- Model: `resnet_backward.pth`
- Results: `resnet_backward_results.csv`

### 4. Co-Teaching

```bash
python train_coteaching.py --data_path data/CIFAR.npz --epochs 15 --batch_size 64 --noise_rate 0.3
```

**Outputs:**
- Model: `resnet_coteaching.pth`
- Results: `resnet_coteaching_results.csv`

## Command Line Arguments

All scripts support the following arguments:

- `--data_path`: Path to CIFAR dataset (default: `data/CIFAR.npz`)
- `--epochs`: Number of training epochs (default: `15`)
- `--batch_size`: Batch size for training (default: `64`)
- `--lr`: Learning rate (default: `0.001`)
- `--model_save_path`: Path to save the best model (`.pth` file)
- `--results_save_path`: Path to save training results (`.csv` file)

Co-Teaching additionally has:
- `--noise_rate`: Noise rate for forget rate schedule (default: `0.3`)

## CSV Output Format

Each CSV file contains the following columns:
- `epoch`: Epoch number
- `train_loss`: Training loss
- `train_acc`: Training accuracy (%)
- `test_loss`: Test loss
- `test_acc`: Test accuracy (%)
- `best_test_acc`: Best test accuracy so far

## Running All Scripts

You can run all scripts sequentially:

```bash
# Baseline
python train_resnet_baseline.py

# Forward Correction
python train_forward_correction.py

# Backward Correction
python train_backward_correction.py

# Co-Teaching
python train_coteaching.py
```

## Transition Matrix

The CIFAR dataset has class-dependent label noise with the following transition matrix:

```
T = [[0.7, 0.3, 0.0],
     [0.0, 0.7, 0.3],
     [0.3, 0.0, 0.7]]
```

Where `T[i,j] = P(observed label = j | true label = i)`

This means:
- Class 0: 70% correct, 30% → Class 1
- Class 1: 70% correct, 30% → Class 2
- Class 2: 70% correct, 30% → Class 0

## Model Saving Strategy

All scripts save the model with the **highest test accuracy** achieved during training, not just the final epoch. This ensures you get the best performing model.

## Example Output

```
Using device: cuda

Loading CIFAR dataset from data/CIFAR.npz...
Training set: (15000, 32, 32, 3)
Test set: (3000, 32, 32, 3)

Model: ResNet18 Baseline
Parameters: 11,170,371

Training for 15 epochs...
================================================================================

Epoch [1/15]
  Train Loss: 1.0234, Train Acc: 45.23%
  Test Loss:  0.9876, Test Acc:  52.10%
  ✓ New best model saved! (Test Acc: 52.10%)
--------------------------------------------------------------------------------
...
```

