data_path="data/CIFAR.npz"
dataset_type="CIFAR"
epochs=15
batch_size=128
lr=0.01
noise_rate=0.3
n_trials=1
model_save_path="resnet_baseline.pth"
results_save_path="resnet_baseline_results.csv"
seed=0
method="baseline"


cmd="python train.py \
    --data_path $data_path \
    --dataset_type $dataset_type \
    --epochs $epochs \
    --batch_size $batch_size \
    --lr $lr \
    --noise_rate $noise_rate \
    --n_trials $n_trials \
    --model_save_path $model_save_path \
    --results_save_path $results_save_path \
    --seed $seed \
    --method $method"

echo "Running baseline method..."
echo $cmd
eval $cmd