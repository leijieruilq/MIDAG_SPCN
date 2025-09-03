#!/bin/bash

# --- 配置区 ---
GPU=1
root_path=./data
seed=2025 # 将seed固定为一个值

all_models=("MIDAG_SPCN")
datasets=("Traffic")
pred_lengths=(12)
batch_sizes=(8) # 在这里设置您想测试的不同batch size
seq_lengths=(8)    # 在这里设置您想测试的不同seq_len
# datasets=("Traffic" "SocialGood" "Security" "Health" "Environment" "Energy" "Economy" "Climate" "Agriculture")

# --- 主循环区 ---
current_dir=$(pwd)
prior_weight=0.5
text_emb=12

for model_name in "${all_models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        data_path=${dataset}.csv
        model_id_prefix=$(basename ${root_path})

        for batch_size in "${batch_sizes[@]}"
        do
            for seq_len in "${seq_lengths[@]}"
            do
                for pred_len in "${pred_lengths[@]}"
                do
                    echo "Running model $model_name on dataset $dataset with seq_len $seq_len, pred_len $pred_len and batch_size $batch_size"
                    
                    CUDA_VISIBLE_DEVICES=${GPU} python -u run.py \
                        --task_name long_term_forecast \
                        --is_training 1 \
                        --root_path $root_path \
                        --data_path $data_path \
                        --model_id ${model_id_prefix}_${seed}_s${seq_len}_p${pred_len}_b${batch_size}_${dataset} \
                        --model $model_name \
                        --data custom \
                        --seq_len $seq_len \
                        --label_len 0 \
                        --pred_len $pred_len \
                        --text_emb $text_emb \
                        --batch_size $batch_size \
                        --des Exp \
                        --seed $seed \
                        --prior_weight $prior_weight \
                        --save_name result_${model_name}_${dataset}_bert \
                        --llm_model BERT \
                        --huggingface_token NA \
                        --train_epochs 10 \
                        --patience 5
                done
            done
        done
    done
done