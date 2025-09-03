export CUDA_VISIBLE_DEVICES=1

model_name=MIDAG_SPCN

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/QPS/ \
  --data_path QPS.csv \
  --model_id QPS_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --layers 1 \
  --num_clusters 64 \
  --id_dim 8 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 512 \
  --enc_in 10 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --des 'Exp' \
  --train_epochs 10 \
  --patience 10 \
  --itr 1 \
  --gpu 0 \
  --lradj cosine \
  --loss 'MAE' \
  --target "y9" \
  # --weight_decay 0.0003 \
  # --use_last 1 \
  # --use_revin 1 \

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/QPS/ \
  --data_path QPS.csv \
  --model_id QPS_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --layers 1 \
  --num_clusters 64 \
  --id_dim 8 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 512 \
  --enc_in 10 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --des 'Exp' \
  --train_epochs 50 \
  --patience 10 \
  --itr 1 \
  --gpu 0 \
  --lradj cosine \
  --loss 'MAE' \
  --target "y9" \
  # --weight_decay 0.0003 \
  # --use_last 1 \
  # --use_revin 1 \

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/QPS/ \
  --data_path QPS.csv \
  --model_id QPS_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --layers 1 \
  --num_clusters 64 \
  --id_dim 8 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 512 \
  --enc_in 10 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --des 'Exp' \
  --train_epochs 50 \
  --patience 10 \
  --itr 1 \
  --gpu 0 \
  --lradj cosine \
  --loss 'MAE' \
  --target "y9" \
  # --weight_decay 0.0003 \
  # --use_last 1 \
  # --use_revin 1 \

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/QPS/ \
  --data_path QPS.csv \
  --model_id QPS_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --layers 3 \
  --num_clusters 64 \
  --id_dim 8 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 512 \
  --enc_in 10 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --des 'Exp' \
  --train_epochs 10 \
  --patience 3 \
  --itr 1 \
  --gpu 0 \
  --lradj cosine \
  --loss 'MAE' \
  --target "y9" \
  --weight_decay 0.0003 \
  --use_last 1 \
  --use_revin 1 \