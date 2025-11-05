"""
Forward Loss Correction + ResNet18 on CIFAR with Noisy Labels
Uses transition matrix to correct predictions before computing loss
"""

import numpy as np
import torch
import pandas as pd
import argparse

from seed_everything import seed_everything
from trainer import run_single_trial, run_single_trial_coteaching
# from estimator import estimate_transition_matrix

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == "__main__":
    print(f"Using device: {device}")

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
            [1, 0.0, 0.0],
            [0.0, 1, 0.0],
            [0.0, 0.0, 1]
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
    # Set random seed
    seed_everything(args.seed, cuda_deterministic=False)
    # Load data
    print(f"\nLoading dataset from {args.data_path}...")
    dataset = np.load(args.data_path)
    X_train = dataset['Xtr']
    y_train = dataset['Str']
    X_test = dataset['Xts']
    y_test = dataset['Yts']
    
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
    # transition_matrix = get_transition_matrix(args.dataset_type, device)
    if args.estimate_T_only:
        # create a placeholder T; it will not be used because trainer returns early in estimate mode
        num_classes = len(np.unique(y_train))
        transition_matrix = torch.eye(num_classes, device=device)
    else:
        use_estimated = (args.transition_matrix_path is not None)
        transition_matrix = None
        if use_estimated:
            try:
                T_loaded = np.load(args.transition_matrix_path)
                transition_matrix = torch.tensor(T_loaded, dtype=torch.float32, device=device)
                print("\nLoaded estimated transition matrix from file:")
                print(transition_matrix.detach().cpu().numpy())
            except Exception as e:
                print("\nCould not load estimated T. Falling back to configured matrix. Reason:", e)

        if transition_matrix is None:
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
    
    # Determine input channels and number of classes
    input_channels = 1 if is_fashion_mnist else 3
    num_classes = len(np.unique(y_train))
    
    print(f"\nModel: ResNet18 + Forward Loss Correction")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Number of Trials: {args.n_trials}")
    print(f"Epochs per Trial: {args.epochs}")
    
    # Run multiple trials
    all_results = []
    trial_best_accs = []
    trial_best_epochs = []
    
    if args.estimate_T_only:
        # one call just to trigger the estimator branch in trainer and exit
        run_single_trial(
            args, 1, X_train, y_train, X_test, y_test,
            is_fashion_mnist, input_channels, num_classes,
            transition_matrix, device
        )
        return
    
    for trial in range(1, args.n_trials + 1):
        if args.method == 'coteaching':
            trial_results, _, best_acc, best_epoch = run_single_trial_coteaching(
                args, trial, X_train, y_train, X_test, y_test,
                is_fashion_mnist, input_channels, num_classes,
                transition_matrix, device
            )
        else:
            trial_results, _, best_acc, best_epoch = run_single_trial(
                args, trial, X_train, y_train, X_test, y_test,
                is_fashion_mnist, input_channels, num_classes,
                transition_matrix, device
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
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--noise_rate', type=float, default=0.3,
                       help='Noise rate for forget rate schedule')
    parser.add_argument('--n_trials', type=int, default=1,
                       help='Number of training trials to run')
    parser.add_argument('--model_save_path', type=str, default='resnet_forward.pth',
                       help='Path to save best model')
    parser.add_argument('--results_save_path', type=str, default='resnet_forward_results.csv',
                       help='Path to save training results')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')
    parser.add_argument('--method', type=str, default='baseline',
                        choices=['forward', 'backward', 'baseline', 'coteaching'],
                        help='Loss correction method to use')
    parser.add_argument('--estimate_T_only', action='store_true',
                        help='Pretrain a baseline model and output an estimated transition matrix then exit')
    parser.add_argument('--pretrain_epochs', type=int, default=3,
                        help='Baseline epochs before estimating T')
    parser.add_argument('--t_top_k', type=int, default=1,
                        help='Top k high confidence samples per class for initial T')
    parser.add_argument('--t_revise_epochs', type=int, default=5,
                        help='Epochs to refine T with slack')
    parser.add_argument('--t_revise_lr', type=float, default=1e-3,
                        help='Learning rate to refine T')
    parser.add_argument('--transition_matrix_path', type=str, default=None,
                        help='Path to .npy file containing transition matrix (overrides default)')

    args = parser.parse_args()
    main(args)

