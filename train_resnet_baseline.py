"""
Baseline ResNet18 Training on CIFAR with Noisy Labels
No noise correction - Standard CrossEntropy Loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import argparse
import os
from datetime import datetime

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom Dataset class
class CIFARDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images) / 255.0
        self.images = self.images.permute(0, 3, 1, 2)
        self.labels = torch.LongTensor(labels)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.normalize(self.images[idx])
        return image, self.labels[idx]

# Pre-trained ResNet18 adapted for CIFAR
def get_pretrained_model(num_classes=3):
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    return running_loss / len(dataloader), 100 * correct / total

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(dataloader), 100 * correct / total

def main(args):
    # Load data
    print(f"\nLoading CIFAR dataset from {args.data_path}...")
    cifar_data = np.load(args.data_path)
    X_train = cifar_data['Xtr']
    y_train = cifar_data['Str']
    X_test = cifar_data['Xts']
    y_test = cifar_data['Yts']
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create datasets and dataloaders
    train_dataset = CIFARDataset(X_train, y_train)
    test_dataset = CIFARDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    num_classes = len(np.unique(y_train))
    model = get_pretrained_model(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print(f"\nModel: ResNet18 Baseline")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    results = []
    best_test_acc = 0.0
    best_epoch = 0
    
    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.model_save_path)
            print(f"  ✓ New best model saved! (Test Acc: {best_test_acc:.2f}%)")
        
        # Store results
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'best_test_acc': best_test_acc
        })
        
        print("-" * 80)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.results_save_path, index=False)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Best Model: Epoch {best_epoch}, Test Acc: {best_test_acc:.2f}%")
    print(f"Results saved to: {args.results_save_path}")
    print(f"Model saved to: {args.model_save_path}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet18 Baseline on CIFAR')
    parser.add_argument('--data_path', type=str, default='data/CIFAR.npz',
                       help='Path to CIFAR dataset')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model_save_path', type=str, default='resnet_baseline.pth',
                       help='Path to save best model')
    parser.add_argument('--results_save_path', type=str, default='resnet_baseline_results.csv',
                       help='Path to save training results')
    
    args = parser.parse_args()
    main(args)

