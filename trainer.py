# Training function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CIFARDataset, FashionMNISTDataset
from get_model import get_model
from losses import BackwardCorrectionLoss, ForwardCorrectionLoss, loss_coteaching
from estimator import estimate_transition_matrix

def pretrain_baseline(model, train_loader, val_loader, device, epochs=3, lr=0.01):
    """
    Fast baseline pretrain with CE to get a decent predictor for T estimation.
    """
    criterion = nn.CrossEntropyLoss()
    optimz = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimz.zero_grad()
            loss.backward()
            optimz.step()
        # one pass val just to keep consistency
        evaluate(model, val_loader, criterion, device)
    return model


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


def train_epoch_coteaching(model1, model2, dataloader, optimizer1, optimizer2, epoch, num_epochs, noise_rate, device):
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
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), 100 * correct / total


def run_single_trial(args, trial_num, X_train, y_train, X_test, y_test,
                      is_fashion_mnist, input_channels, num_classes,
                      transition_matrix, device):
    """Run a single training trial"""

    print(f"\n{'='*80}")
    print(f"TRIAL {trial_num}/{args.n_trials}")
    print(f"{'='*80}")

    # Split 80-20 randomly for training and validation
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Create datasets and dataloaders
    if is_fashion_mnist:
        train_dataset = FashionMNISTDataset(X_train[train_indices], y_train[train_indices])
        val_dataset = FashionMNISTDataset(X_train[val_indices], y_train[val_indices])
        test_dataset = FashionMNISTDataset(X_test, y_test)
    else:
        train_dataset = CIFARDataset(X_train[train_indices], y_train[train_indices])
        val_dataset = CIFARDataset(X_train[val_indices], y_train[val_indices])
        test_dataset = CIFARDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = get_model(num_classes=num_classes, input_channels=input_channels).to(device)

    if args.method == 'forward':
        criterion_train = ForwardCorrectionLoss(transition_matrix)
    elif args.method == 'backward':
        criterion_train = BackwardCorrectionLoss(transition_matrix)
    elif args.method == 'baseline':
        criterion_train = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown method: {args.method}. Choose from 'forward', 'backward', 'baseline'.")
    criterion_eval = nn.CrossEntropyLoss()    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    if args.estimate_T_only:
        # quick baseline pretrain to make probabilities useful
        from trainer import pretrain_baseline
        pretrain_baseline(model, train_loader, val_loader, device,
                          epochs=args.pretrain_epochs, lr=args.lr)

        # estimate T using t revision
        T_est = estimate_transition_matrix(model, train_dataset,
                                           batch_size=args.batch_size, num_workers=2,
                                           num_classes=num_classes, device=device,
                                           top_k=args.t_top_k,
                                           revise_epochs=args.t_revise_epochs,
                                           revise_lr=args.t_revise_lr)
        np.save(args.transition_matrix_path, T_est.detach().cpu().numpy())
        print("\nEstimated transition matrix saved at:", args.transition_matrix_path)
        print(T_est.detach().cpu().numpy())
        return [], 0.0, 0.0, 0
    # Training loop
    results = []
    best_test_acc = 0.0
    best_val_acc = 0.0
    best_epoch = 0

    print(f"Training for {args.epochs} epochs...")
    print("-" * 80)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion_train, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion_eval, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion_train, device)

        print(f"Epoch [{epoch+1}/{args.epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch + 1
            # if args.n_trials == 1:
            #     torch.save(model.state_dict(), args.model_save_path)
            # else:
            #     # Save with trial number
            #     model_path = args.model_save_path.replace('.pth', f'_trial{trial_num}.pth')
            #     torch.save(model.state_dict(), model_path)

        # Store results
        results.append({
            'trial': trial_num,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
        })

    print("-" * 80)
    print(f"TRIAL {trial_num} COMPLETED | Best Epoch: {best_epoch} | Best Val Acc: {best_val_acc:.2f}% | Best Test Acc: {best_test_acc:.2f}%")
    print("=" * 80)

    return results, best_val_acc, best_test_acc, best_epoch


def run_single_trial_coteaching(args, trial_num, X_train, y_train, X_test, y_test,
                      is_fashion_mnist, input_channels, num_classes,
                      transition_matrix, device):
    """Run a single training trial"""

    print(f"\n{'='*80}")
    print(f"TRIAL {trial_num}/{args.n_trials}")
    print(f"{'='*80}")
    # Split 80-20 randomly for training and validation
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    # Create datasets and dataloaders
    if is_fashion_mnist:
        train_dataset = FashionMNISTDataset(X_train[train_indices], y_train[train_indices])
        val_dataset = FashionMNISTDataset(X_train[val_indices], y_train[val_indices])
        test_dataset = FashionMNISTDataset(X_test, y_test)
    else:
        train_dataset = CIFARDataset(X_train[train_indices], y_train[train_indices])
        val_dataset = CIFARDataset(X_train[val_indices], y_train[val_indices])
        test_dataset = CIFARDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = get_model(num_classes=num_classes, input_channels=input_channels).to(device)
    model2 = get_model(num_classes=num_classes, input_channels=input_channels).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    criterion_train = nn.CrossEntropyLoss()  # Will handle inside training loop
    criterion_eval = nn.CrossEntropyLoss()

    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Training loop

    print(f"\nModel: Co-Teaching with 2 ResNet18 networks")
    print(f"Parameters per network: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Noise rate: {args.noise_rate}")

    results = []
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0

    print(f"Training for {args.epochs} epochs...")
    print("-" * 80)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch_coteaching(
            model, model2, train_loader,
            optimizer, optimizer2,
            epoch, args.epochs, args.noise_rate, device
        )
        val_loss1, val_acc1 = evaluate(model, val_loader, criterion_train, device)
        val_loss2, val_acc2 = evaluate(model2, val_loader, criterion_train, device)

        # Evaluate both models and take the better one
        test_loss1, test_acc1 = evaluate(model, test_loader, criterion_eval, device)
        test_loss2, test_acc2 = evaluate(model2, test_loader, criterion_eval, device)

        # Use average of both models for reporting
        val_loss = (val_loss1 + val_loss2) / 2
        val_acc = (val_acc1 + val_acc2) / 2
        test_loss = (test_loss1 + test_loss2) / 2
        test_acc = (test_acc1 + test_acc2) / 2

        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss:  {val_loss:.4f},  Val  Acc:  {val_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        print(f"  Model 1 Test Acc: {test_acc1:.2f}%, Model 2 Test Acc: {test_acc2:.2f}%")

        # Save best model (use the better of the two models)
        current_best_acc = max(val_acc1, val_acc2)

        if current_best_acc > best_val_acc:
            best_val_acc = current_best_acc
            best_test_acc = test_acc1 if val_acc1 >= val_acc2 else test_acc2
            best_epoch = epoch + 1
            # Save the better model
            # if val_acc1 >= val_acc2:
            #     torch.save(model.state_dict(), args.model_save_path)
            #     print(f"  ✓ Model 1 saved! (Val Acc: {val_acc1:.2f}%)")
            # else:
            #     torch.save(model2.state_dict(), args.model_save_path)
            #     print(f"  ✓ Model 2 saved! (Val Acc: {val_acc2:.2f}%)")
        # Store results
        results.append({
            'trial': trial_num,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_acc1': test_acc1,
            'test_acc2': test_acc2,
            'val_acc1': val_acc1,
            'val_acc2': val_acc2,
            'best_test_acc': best_test_acc,
            'best_val_acc': best_val_acc
        })

        print("-" * 80)


    print("-" * 80)
    print(f"TRIAL {trial_num} COMPLETED | Best Epoch: {best_epoch} | Best Val Acc: {best_val_acc:.2f}% | Best Test Acc: {best_test_acc:.2f}%")
    print("=" * 80)

    return results, best_val_acc, best_test_acc, best_epoch