# Quick Start Guide

This guide helps you quickly run all 4 noisy label learning approaches on CIFAR dataset.

## 📁 Files Created

### Training Scripts (Python)
1. `train_resnet_baseline.py` - Standard ResNet18 (no noise handling)
2. `train_forward_correction.py` - Forward loss correction with transition matrix
3. `train_backward_correction.py` - Backward loss correction with inverse transition matrix
4. `train_coteaching.py` - Co-Teaching with two networks

### Helper Scripts
- `run_all_experiments.py` - Python script to run all experiments and compare results
- `run_all.bat` - Windows batch script
- `run_all.sh` - Linux/Mac shell script

### Documentation
- `README_training_scripts.md` - Detailed documentation
- `QUICK_START.md` - This file

## 🚀 Quick Start

### Option 1: Run All Experiments at Once (Recommended)

**Windows:**
```cmd
run_all.bat
```

**Linux/Mac:**
```bash
chmod +x run_all.sh
./run_all.sh
```

**Or use Python script (cross-platform):**
```bash
python run_all_experiments.py
```

### Option 2: Run Individual Experiments

```bash
# 1. Baseline
python train_resnet_baseline.py

# 2. Forward Correction
python train_forward_correction.py

# 3. Backward Correction
python train_backward_correction.py

# 4. Co-Teaching
python train_coteaching.py
```

## 📊 Output Files

After running, you'll get:

### Model Files (.pth)
- `resnet_baseline.pth` - Best baseline model
- `resnet_forward.pth` - Best forward correction model
- `resnet_backward.pth` - Best backward correction model
- `resnet_coteaching.pth` - Best co-teaching model

### Results Files (.csv)
- `resnet_baseline_results.csv` - Training history for baseline
- `resnet_forward_results.csv` - Training history for forward correction
- `resnet_backward_results.csv` - Training history for backward correction
- `resnet_coteaching_results.csv` - Training history for co-teaching

### Comparison File
- `all_methods_comparison.csv` - Side-by-side comparison (created by `run_all_experiments.py`)

## 📈 CSV File Structure

Each results CSV contains:

| Column | Description |
|--------|-------------|
| `epoch` | Epoch number (1 to 15) |
| `train_loss` | Training loss |
| `train_acc` | Training accuracy (%) |
| `test_loss` | Test loss |
| `test_acc` | Test accuracy (%) |
| `best_test_acc` | Best test accuracy achieved so far |

## ⚙️ Customization

All scripts support command-line arguments:

```bash
python train_resnet_baseline.py \
    --data_path data/CIFAR.npz \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.0001 \
    --model_save_path my_model.pth \
    --results_save_path my_results.csv
```

For co-teaching, you can also set the noise rate:
```bash
python train_coteaching.py --noise_rate 0.3
```

## 🔍 Understanding the Methods

### 1. Baseline ResNet18
- **Approach**: Standard supervised learning
- **Loss**: CrossEntropy
- **Expected**: Will overfit to noisy labels

### 2. Forward Loss Correction
- **Approach**: Adjust predictions using transition matrix before computing loss
- **Formula**: `loss = CE(softmax(logits) @ T, noisy_labels)`
- **Expected**: Better than baseline, handles known noise patterns

### 3. Backward Loss Correction
- **Approach**: Use inverse transition matrix to correct loss
- **Formula**: `loss = CE(logits, corrected_labels)` where `corrected_labels = one_hot @ T^-1`
- **Expected**: Similar to forward correction, theoretically sound

### 4. Co-Teaching
- **Approach**: Two networks teach each other, selecting small-loss samples
- **Strategy**: Each network picks clean samples for the other to learn from
- **Expected**: Robust to high noise rates, no transition matrix needed

## 📊 Example Results

After running all experiments, you can compare results:

```bash
python -c "import pandas as pd; print(pd.read_csv('all_methods_comparison.csv'))"
```

Expected output format:
```
                 Method  Best Test Acc (%)  Best Epoch  Final Test Acc (%)  Total Epochs
              Baseline              68.50           8               66.20            15
    Forward Correction              72.30          10               71.50            15
   Backward Correction              71.80           9               70.90            15
          Co-Teaching              73.10          11               72.40            15
```

## 🐛 Troubleshooting

**Issue**: CUDA out of memory
```bash
# Reduce batch size
python train_resnet_baseline.py --batch_size 32
```

**Issue**: Module not found
```bash
# Install requirements
pip install torch torchvision numpy pandas tqdm
```

**Issue**: Can't find data file
```bash
# Specify correct path
python train_resnet_baseline.py --data_path path/to/CIFAR.npz
```

## 💡 Tips

1. **GPU Recommended**: Training will be much faster with CUDA
2. **Batch Size**: Adjust based on your GPU memory (default: 64)
3. **Learning Rate**: Default 0.01 works well, but you can experiment
4. **Epochs**: 15 is usually sufficient for CIFAR with pre-trained ResNet18

## 📝 Citation

If you use these implementations, please cite the original papers:

**Forward/Backward Correction:**
- Patrini et al. "Making Deep Neural Networks Robust to Label Noise" CVPR 2017

**Co-Teaching:**
- Han et al. "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels" NeurIPS 2018

## 📧 Questions?

See `README_training_scripts.md` for more detailed documentation.

