"""
Backward Loss Correction + ResNet18 on CIFAR with Noisy Labels
Uses inverse transition matrix to correct the loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import argparse

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

# Custom Dataset class for FashionMNIST
class FashionMNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images) / 255.0
        # Handle grayscale images - add channel dimension if needed
        if len(self.images.shape) == 3:  # (N, H, W)
            self.images = self.images.unsqueeze(1)  # (N, 1, H, W)
        elif len(self.images.shape) == 4 and self.images.shape[-1] == 1:  # (N, H, W, 1)
            self.images = self.images.permute(0, 3, 1, 2)  # (N, 1, H, W)
        self.labels = torch.LongTensor(labels)
        # FashionMNIST normalization (mean and std for grayscale)
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.normalize(self.images[idx])
        return image, self.labels[idx]

# Pre-trained ResNet18 adapted for CIFAR and FashionMNIST
def get_pretrained_model(num_classes=3, input_channels=3):
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Backward Correction Loss
class BackwardCorrectionLoss(nn.Module):
    def __init__(self, transition_matrix):
        super(BackwardCorrectionLoss, self).__init__()
        # Use inverse of transition matrix
        self.transition_matrix_inv = torch.inverse(transition_matrix)
        
    def forward(self, logits, noisy_labels):
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(logits, noisy_labels, reduction='none')
        
        # Get one-hot encoded noisy labels
        batch_size = noisy_labels.size(0)
        num_classes = logits.size(1)
        one_hot = torch.zeros(batch_size, num_classes).to(logits.device)
        one_hot.scatter_(1, noisy_labels.unsqueeze(1), 1)
        
        # Apply inverse transition matrix: T^-1 @ one_hot
        # This gives us the corrected label distribution
        corrected_labels = torch.matmul(one_hot, self.transition_matrix_inv)
        corrected_labels = torch.clamp(corrected_labels, min=0.0, max=1.0)
        
        # Normalize to ensure it sums to 1
        corrected_labels = corrected_labels / corrected_labels.sum(dim=1, keepdim=True)
        
        # Compute loss with corrected labels
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(corrected_labels * log_probs).sum(dim=1).mean()
        
        return loss

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training (Backward)', leave=False)
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

def get_transition_matrix(dataset_name, device):
    """
    Returns the transition matrix for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'CIFAR', 'FashionMNIST0.3', 'FashionMNIST0.6')
        device: torch device
    
    Returns:
        Transition matrix as a torch tensor
    """
    transition_matrices = {
        'CIFAR': torch.FloatTensor([
            [0.7, 0.3, 0.0],
            [0.0, 0.7, 0.3],
            [0.3, 0.0, 0.7]
        ]),
        'FashionMNIST0.3': torch.FloatTensor([
            [0.7, 0.3, 0.0],
            [0.0, 0.7, 0.3],
            [0.3, 0.0, 0.7]
        ]),
        'FashionMNIST0.6': torch.FloatTensor([
            [0.4, 0.3, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.3, 0.4]
        ])
    }
    
    if dataset_name not in transition_matrices:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(transition_matrices.keys())}")
    
    return transition_matrices[dataset_name].to(device)

def main(args):
    # Load data
    print(f"\nLoading dataset from {args.data_path}...")
    cifar_data = np.load(args.data_path)
    X_train = cifar_data['Xtr']
    y_train = cifar_data['Str']
    X_test = cifar_data['Xts']
    y_test = cifar_data['Yts']
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Determine if we're using FashionMNIST or CIFAR
    is_fashion_mnist = 'FashionMNIST' in args.dataset_type
    
    # Reshape FashionMNIST data if needed (from flat 784 to 28x28)
    if is_fashion_mnist and len(X_train.shape) == 2:
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)
        print(f"Reshaped training set: {X_train.shape}")
        print(f"Reshaped test set: {X_test.shape}")
    
    # Get transition matrix based on dataset type
    transition_matrix = get_transition_matrix(args.dataset_type, device)
    
    print("\nTransition Matrix:")
    print(transition_matrix.cpu().numpy())
    
    # Check if invertible
    try:
        T_inv = torch.inverse(transition_matrix)
        print("\nInverse Transition Matrix:")
        print(T_inv.cpu().numpy())
    except:
        print("\nWarning: Transition matrix is not invertible!")
        return
    
    # Create datasets and dataloaders using appropriate dataset class
    if is_fashion_mnist:
        train_dataset = FashionMNISTDataset(X_train, y_train)
        test_dataset = FashionMNISTDataset(X_test, y_test)
        input_channels = 1
    else:
        train_dataset = CIFARDataset(X_train, y_train)
        test_dataset = CIFARDataset(X_test, y_test)
        input_channels = 3
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    num_classes = len(np.unique(y_train))
    model = get_pretrained_model(num_classes=num_classes, input_channels=input_channels).to(device)
    
    # Backward correction loss for training, standard loss for evaluation
    criterion_train = BackwardCorrectionLoss(transition_matrix)
    criterion_eval = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print(f"\nModel: ResNet18 + Backward Loss Correction")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    results = []
    best_test_acc = 0.0
    best_epoch = 0
    
    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion_train, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion_eval, device)
        
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
    parser = argparse.ArgumentParser(description='Train ResNet18 with Backward Loss Correction')
    parser.add_argument('--data_path', type=str, default='data/CIFAR.npz',
                       help='Path to dataset (.npz file)')
    parser.add_argument('--dataset_type', type=str, default='CIFAR',
                       choices=['CIFAR', 'FashionMNIST0.3', 'FashionMNIST0.6'],
                       help='Type of dataset (determines transition matrix)')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model_save_path', type=str, default='resnet_backward.pth',
                       help='Path to save best model')
    parser.add_argument('--results_save_path', type=str, default='resnet_backward_results.csv',
                       help='Path to save training results')
    
    args = parser.parse_args()
    main(args)
