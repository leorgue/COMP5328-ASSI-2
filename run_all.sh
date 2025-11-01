#!/bin/bash
# Unix/Linux shell script to run all experiments

echo "================================================================================"
echo "RUNNING ALL EXPERIMENTS"
echo "================================================================================"
echo ""

echo "[1/4] Training ResNet18 Baseline..."
python train_resnet_baseline.py
if [ $? -ne 0 ]; then
    echo "ERROR: Baseline training failed"
    exit 1
fi
echo ""

echo "[2/4] Training ResNet18 + Forward Loss Correction..."
python train_forward_correction.py
if [ $? -ne 0 ]; then
    echo "ERROR: Forward correction training failed"
    exit 1
fi
echo ""

echo "[3/4] Training ResNet18 + Backward Loss Correction..."
python train_backward_correction.py
if [ $? -ne 0 ]; then
    echo "ERROR: Backward correction training failed"
    exit 1
fi
echo ""

echo "[4/4] Training ResNet18 + Co-Teaching..."
python train_coteaching.py
if [ $? -ne 0 ]; then
    echo "ERROR: Co-Teaching training failed"
    exit 1
fi
echo ""

echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "================================================================================"
echo ""
echo "Output files:"
echo "  - resnet_baseline.pth & resnet_baseline_results.csv"
echo "  - resnet_forward.pth & resnet_forward_results.csv"
echo "  - resnet_backward.pth & resnet_backward_results.csv"
echo "  - resnet_coteaching.pth & resnet_coteaching_results.csv"
echo ""

