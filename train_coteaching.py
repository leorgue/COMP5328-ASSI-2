"""
Co-Teaching + ResNet18 on CIFAR with Noisy Labels
Two networks teach each other by selecting small-loss samples
Reference: Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels (NeurIPS 2018)
"""

import numpy as np
from get_model import get_model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import CIFARDataset, FashionMNISTDataset
from tqdm import tqdm
import pandas as pd
import argparse

from seed_everything import seed_everything

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Pre-trained ResNet18 adapted for CIFAR
# Co-Teaching Loss
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    """
    Co-Teaching loss function
    y_1, y_2: logits from two networks
    t: labels
    forget_rate: percentage of data to discard
    """
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = torch.argsort(loss_1)
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = torch.argsort(loss_2)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    # Select small loss samples from each network
    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    # Exchange and update
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return loss_1_update, loss_2_update

# Training function for Co-Teaching
def train_epoch(model1, model2, dataloader, optimizer1, optimizer2, epoch, num_epochs, noise_rate, device):
    model1.train()
    model2.train()
    
    running_loss1 = 0.0
    running_loss2 = 0.0
    correct1 = 0
    correct2 = 0
    total = 0
    
    # Define forget rate schedule
    forget_rate = noise_rate
    exponent = 1
    # Adjust forget rate
    forget_rate = min(forget_rate, (epoch / num_epochs) * noise_rate)
    
    pbar = tqdm(dataloader, desc=f'Training (forget_rate={forget_rate:.3f})', leave=False)
    
    for images, labels, indices in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass for both networks
        logits1 = model1(images)
        logits2 = model2(images)
        
        # Co-teaching loss
        loss_1, loss_2 = loss_coteaching(logits1, logits2, labels, forget_rate, 
                                         indices, None)
        
        # Update network 1
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        
        # Update network 2
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        
        # Statistics
        running_loss1 += loss_1.item()
        running_loss2 += loss_2.item()
        
        _, predicted1 = torch.max(logits1.data, 1)
        _, predicted2 = torch.max(logits2.data, 1)
        
        total += labels.size(0)
        correct1 += (predicted1 == labels).sum().item()
        correct2 += (predicted2 == labels).sum().item()
        
        pbar.set_postfix({
            'loss1': f'{loss_1.item():.4f}',
            'loss2': f'{loss_2.item():.4f}',
            'acc': f'{100*correct1/total:.2f}%'
        })
    
    avg_loss = (running_loss1 + running_loss2) / (2 * len(dataloader))
    avg_acc = (100 * correct1 / total + 100 * correct2 / total) / 2
    
    return avg_loss, avg_acc

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(dataloader), 100 * correct / total

def main(args):
    # Set random seed
    seed_everything(args.seed, cuda_deterministic=False)
    # Load data
    print(f"\nLoading CIFAR dataset from {args.data_path}...")
    dataset = np.load(args.data_path)
    X_train = dataset['Xtr']
    y_train = dataset['Str']
    X_test = dataset['Xts']
    y_test = dataset['Yts']
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create datasets and dataloaders
    train_dataset = CIFARDataset(X_train, y_train)
    test_dataset = CIFARDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize two models for co-teaching
    num_classes = len(np.unique(y_train))
    model1 = get_model(num_classes=num_classes).to(device)
    model2 = get_model(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print(f"\nModel: Co-Teaching with 2 ResNet18 networks")
    print(f"Parameters per network: {sum(p.numel() for p in model1.parameters()):,}")
    print(f"Noise rate: {args.noise_rate}")
    
    # Training loop
    results = []
    best_test_acc = 0.0
    best_epoch = 0
    
    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model1, model2, train_loader, 
            optimizer1, optimizer2, 
            epoch, args.epochs, args.noise_rate, device
        )
        
        # Evaluate both models and take the better one
        test_loss1, test_acc1 = evaluate(model1, test_loader, criterion, device)
        test_loss2, test_acc2 = evaluate(model2, test_loader, criterion, device)
        
        # Use average of both models for reporting
        test_loss = (test_loss1 + test_loss2) / 2
        test_acc = (test_acc1 + test_acc2) / 2
        
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        print(f"  Model 1 Test Acc: {test_acc1:.2f}%, Model 2 Test Acc: {test_acc2:.2f}%")
        
        # Save best model (use the better of the two models)
        current_best_acc = max(test_acc1, test_acc2)
        if current_best_acc > best_test_acc:
            best_test_acc = current_best_acc
            best_epoch = epoch + 1
            # Save the better model
            if test_acc1 >= test_acc2:
                torch.save(model1.state_dict(), args.model_save_path)
                print(f"  ✓ Model 1 saved! (Test Acc: {test_acc1:.2f}%)")
            else:
                torch.save(model2.state_dict(), args.model_save_path)
                print(f"  ✓ Model 2 saved! (Test Acc: {test_acc2:.2f}%)")
        
        # Store results
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_acc1': test_acc1,
            'test_acc2': test_acc2,
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
    parser = argparse.ArgumentParser(description='Train ResNet18 with Co-Teaching')
    parser.add_argument('--data_path', type=str, default='data/CIFAR.npz',
                       help='Path to CIFAR dataset')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--noise_rate', type=float, default=0.3,
                       help='Noise rate for forget rate schedule')
    parser.add_argument('--model_save_path', type=str, default='resnet_coteaching.pth',
                       help='Path to save best model')
    parser.add_argument('--results_save_path', type=str, default='resnet_coteaching_results.csv',
                       help='Path to save training results')
    
    args = parser.parse_args()
    main(args)

