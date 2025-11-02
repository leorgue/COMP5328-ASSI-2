"""
Forward Loss Correction + ResNet18 on CIFAR with Noisy Labels
Uses transition matrix to correct predictions before computing loss
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

# Forward Correction Loss
class ForwardCorrectionLoss(nn.Module):
    def __init__(self, transition_matrix):
        super(ForwardCorrectionLoss, self).__init__()
        self.transition_matrix = transition_matrix
    
    def forward(self, logits, noisy_labels):
        probs = F.softmax(logits, dim=1)
        corrected_probs = torch.matmul(probs, self.transition_matrix)
        corrected_probs = torch.clamp(corrected_probs, min=1e-7, max=1.0)
        loss = F.nll_loss(torch.log(corrected_probs), noisy_labels)
        return loss

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training (Forward)', leave=False)
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

def run_single_trial(args, trial_num, X_train, y_train, X_test, y_test, 
                      is_fashion_mnist, input_channels, num_classes, 
                      transition_matrix, criterion_eval):
    """Run a single training trial"""
    
    print(f"\n{'='*80}")
    print(f"TRIAL {trial_num}/{args.n_trials}")
    print(f"{'='*80}")
    
    # Create datasets and dataloaders
    if is_fashion_mnist:
        train_dataset = FashionMNISTDataset(X_train, y_train)
        test_dataset = FashionMNISTDataset(X_test, y_test)
    else:
        train_dataset = CIFARDataset(X_train, y_train)
        test_dataset = CIFARDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = get_pretrained_model(num_classes=num_classes, input_channels=input_channels).to(device)
    
    # Forward correction loss for training
    criterion_train = ForwardCorrectionLoss(transition_matrix)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Training loop
    results = []
    best_test_acc = 0.0
    best_epoch = 0
    
    print(f"Training for {args.epochs} epochs...")
    print("-" * 80)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion_train, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion_eval, device)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            if args.n_trials == 1:
                torch.save(model.state_dict(), args.model_save_path)
            else:
                # Save with trial number
                model_path = args.model_save_path.replace('.pth', f'_trial{trial_num}.pth')
                torch.save(model.state_dict(), model_path)
        
        # Store results
        results.append({
            'trial': trial_num,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'best_test_acc': best_test_acc
        })
    
    print("-" * 80)
    print(f"TRIAL {trial_num} COMPLETED | Best Epoch: {best_epoch} | Best Test Acc: {best_test_acc:.2f}%")
    print("=" * 80)
    
    return results, best_test_acc, best_epoch

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
    
    # Determine input channels and number of classes
    input_channels = 1 if is_fashion_mnist else 3
    num_classes = len(np.unique(y_train))
    criterion_eval = nn.CrossEntropyLoss()
    
    print(f"\nModel: ResNet18 + Forward Loss Correction")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Number of Trials: {args.n_trials}")
    print(f"Epochs per Trial: {args.epochs}")
    
    # Run multiple trials
    all_results = []
    trial_best_accs = []
    trial_best_epochs = []
    
    for trial in range(1, args.n_trials + 1):
        trial_results, best_acc, best_epoch = run_single_trial(
            args, trial, X_train, y_train, X_test, y_test,
            is_fashion_mnist, input_channels, num_classes,
            transition_matrix, criterion_eval
        )
        all_results.extend(trial_results)
        trial_best_accs.append(best_acc)
        trial_best_epochs.append(best_epoch)
    
    # Save all results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(args.results_save_path, index=False)
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total Trials: {args.n_trials}")
    print(f"\nBest Test Accuracy per Trial:")
    for i, acc in enumerate(trial_best_accs, 1):
        print(f"  Trial {i}: {acc:.2f}% (Best Epoch: {trial_best_epochs[i-1]})")
    
    if args.n_trials > 1:
        mean_acc = np.mean(trial_best_accs)
        std_acc = np.std(trial_best_accs)
        print(f"\nAggregate Statistics:")
        print(f"  Mean Test Acc: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"  Min Test Acc:  {np.min(trial_best_accs):.2f}%")
        print(f"  Max Test Acc:  {np.max(trial_best_accs):.2f}%")
    
    print(f"\nResults saved to: {args.results_save_path}")
    if args.n_trials == 1:
        print(f"Model saved to: {args.model_save_path}")
    else:
        print(f"Models saved with trial numbers: {args.model_save_path.replace('.pth', '_trial*.pth')}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet18 with Forward Loss Correction')
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
    parser.add_argument('--n_trials', type=int, default=1,
                       help='Number of training trials to run')
    parser.add_argument('--model_save_path', type=str, default='resnet_forward.pth',
                       help='Path to save best model')
    parser.add_argument('--results_save_path', type=str, default='resnet_forward_results.csv',
                       help='Path to save training results')
    
    args = parser.parse_args()
    main(args)

