"""
Example script to load a trained model and test it
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import CIFARDataset
import torchvision.models as models
import argparse

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pre-trained ResNet18 adapted for CIFAR
def get_model(num_classes=3):
    model = models.resnet18(pretrained=False)  # Don't load ImageNet weights
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def evaluate_model(model, dataloader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, np.array(all_labels), np.array(all_preds)

def print_per_class_accuracy(labels, preds, num_classes=3):
    """Print accuracy for each class"""
    print("\nPer-Class Accuracy:")
    print("-" * 40)
    for i in range(num_classes):
        mask = (labels == i)
        if mask.sum() > 0:
            class_acc = 100 * (preds[mask] == labels[mask]).sum() / mask.sum()
            print(f"  Class {i}: {class_acc:.2f}% ({mask.sum()} samples)")

def main(args):
    # Load test data
    print(f"Loading CIFAR test data from {args.data_path}...")
    cifar_data = np.load(args.data_path)
    X_test = cifar_data['Xts']
    y_test = cifar_data['Yts']
    
    print(f"Test set: {X_test.shape}")
    
    # Create dataset and dataloader
    test_dataset = CIFARDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    num_classes = len(np.unique(y_test))
    model = get_model(num_classes=num_classes).to(device)
    
    # Load trained weights
    print(f"\nLoading model from {args.model_path}...")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Evaluate
    print("\nEvaluating model on test set...")
    accuracy, labels, preds = evaluate_model(model, test_loader, device)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Per-class accuracy
    print_per_class_accuracy(labels, preds, num_classes)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    
    print("\nConfusion Matrix:")
    print("-" * 40)
    print("       Predicted")
    print("       ", end="")
    for i in range(num_classes):
        print(f"{i:6}", end="")
    print()
    print("True")
    for i in range(num_classes):
        print(f"  {i}  ", end="")
        for j in range(num_classes):
            print(f"{cm[i, j]:6}", end="")
        print()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and test a trained model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model (.pth file)')
    parser.add_argument('--data_path', type=str, default='data/CIFAR.npz',
                       help='Path to CIFAR dataset')
    
    args = parser.parse_args()
    main(args)

