declare -a data_paths=("data/CIFAR.npz" "data/FashionMNIST0.3.npz" "data/FashionMNIST0.6.npz")
declare -a dataset_types=("CIFAR" "FashionMNIST0.3" "FashionMNIST0.6")
epochs=15
batch_size=1280
lr=0.01
noise_rate=0.3
n_trials=10
model_save_path="resnet_forward.pth"
results_save_path="resnet_forward_results.csv"
seed=0
method="forward"

for i in "${!data_paths[@]}"; do
    data_path=${data_paths[$i]}
    dataset_type=${dataset_types[$i]}
    results_save_path="resnet_forward_${dataset_type}_results.csv"

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

    echo "Running forward method on $dataset_type..."
    echo $cmd
    eval $cmd
done

# cmd="python train.py \
#     --data_path $data_path \
#     --dataset_type $dataset_type \
#     --epochs $epochs \
#     --batch_size $batch_size \
#     --lr $lr \
#     --noise_rate $noise_rate \
#     --n_trials $n_trials \
#     --model_save_path $model_save_path \
#     --results_save_path $results_save_path \
#     --seed $seed \
#     --method $method"

# echo "Running forward method..."
# echo $cmd
# eval $cmd