"""
Run all training experiments and compare results
"""

import subprocess
import pandas as pd
import os

def run_experiment(script_name, description):
    """Run a training script"""
    print("\n" + "=" * 80)
    print(f"Running: {description}")
    print("=" * 80)
    
    try:
        subprocess.run(['python', script_name], check=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error: {e}")
        return False

def compare_results():
    """Load and compare all results"""
    print("\n" + "=" * 80)
    print("COMPARING RESULTS")
    print("=" * 80)
    
    results_files = {
        'Baseline': 'resnet_baseline_results.csv',
        'Forward Correction': 'resnet_forward_results.csv',
        'Backward Correction': 'resnet_backward_results.csv',
        'Co-Teaching': 'resnet_coteaching_results.csv'
    }
    
    comparison = []
    
    for method, file_path in results_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            best_test_acc = df['best_test_acc'].max()
            final_test_acc = df['test_acc'].iloc[-1]
            best_epoch = df.loc[df['test_acc'].idxmax(), 'epoch']
            
            comparison.append({
                'Method': method,
                'Best Test Acc (%)': f"{best_test_acc:.2f}",
                'Best Epoch': int(best_epoch),
                'Final Test Acc (%)': f"{final_test_acc:.2f}",
                'Total Epochs': len(df)
            })
        else:
            print(f"Warning: {file_path} not found")
    
    if comparison:
        comparison_df = pd.DataFrame(comparison)
        print("\n" + comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv('all_methods_comparison.csv', index=False)
        print(f"\n✓ Comparison saved to: all_methods_comparison.csv")
    
    print("\n" + "=" * 80)

def main():
    print("=" * 80)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 80)
    print("\nThis will train 4 models:")
    print("1. ResNet18 Baseline")
    print("2. ResNet18 + Forward Loss Correction")
    print("3. ResNet18 + Backward Loss Correction")
    print("4. ResNet18 + Co-Teaching")
    print("\nEach will run for 15 epochs.")
    
    input("\nPress Enter to start...")
    
    experiments = [
        ('train_resnet_baseline.py', 'ResNet18 Baseline'),
        ('train_forward_correction.py', 'ResNet18 + Forward Loss Correction'),
        ('train_backward_correction.py', 'ResNet18 + Backward Loss Correction'),
        ('train_coteaching.py', 'ResNet18 + Co-Teaching'),
    ]
    
    results = {}
    for script, description in experiments:
        success = run_experiment(script, description)
        results[description] = success
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    for experiment, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {experiment}")
    
    # Compare results
    if all(results.values()):
        compare_results()
    else:
        print("\nSome experiments failed. Skipping comparison.")
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print("\nOutput files:")
    print("  Models: resnet_baseline.pth, resnet_forward.pth, resnet_backward.pth, resnet_coteaching.pth")
    print("  Results: *_results.csv files")
    print("  Comparison: all_methods_comparison.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()

