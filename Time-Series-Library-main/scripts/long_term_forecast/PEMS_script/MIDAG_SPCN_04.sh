export CUDA_VISIBLE_DEVICES=0

model_name=MIDAG_SPCN

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/PEMS04/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --layers 1 \
  --num_clusters 64 \
  --id_dim 8 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 512 \
  --enc_in 307 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --des 'Exp' \
  --train_epochs 50 \
  --patience 10 \
  --itr 1 \
  --gpu 0 \
  --lradj cosine \
  --loss 'MAE' \
  # --weight_decay 0.0003 \
  # --use_last 1 \
  # --use_revin 1 \

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/PEMS04/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_24 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --layers 1 \
  --num_clusters 64 \
  --id_dim 8 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 512 \
  --enc_in 307 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --des 'Exp' \
  --train_epochs 50 \
  --patience 10 \
  --itr 1 \
  --gpu 0 \
  --lradj cosine \
  --loss 'MAE' \
  # --weight_decay 0.0003 \
  # --use_last 1 \
  # --use_revin 1 \

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/PEMS04/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_48 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --layers 1 \
  --num_clusters 64 \
  --id_dim 8 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 512 \
  --enc_in 307 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --des 'Exp' \
  --train_epochs 50 \
  --patience 10 \
  --itr 1 \
  --gpu 0 \
  --lradj cosine \
  --loss 'MAE' \
  # --weight_decay 0.0003 \
  # --use_last 1 \
  # --use_revin 1 \

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/PEMS04/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_96 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --layers 1 \
  --num_clusters 64 \
  --id_dim 8 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 512 \
  --enc_in 307 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --des 'Exp' \
  --train_epochs 50 \
  --patience 10 \
  --itr 1 \
  --gpu 0 \
  --lradj cosine \
  --loss 'MAE' \
  # --weight_decay 0.0003 \
  # --use_last 1 \
  # --use_revin 1 \