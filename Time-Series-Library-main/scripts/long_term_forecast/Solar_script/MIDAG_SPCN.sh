export CUDA_VISIBLE_DEVICES=0

model_name=MIDAG_SPCN

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/SOLAR/ \
  --data_path solar_Alabama.csv  \
  --model_id solar_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --layers 1 \
  --num_clusters 64 \
  --id_dim 32 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 256 \
  --enc_in 137 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --des 'Exp' \
  --train_epochs 10 \
  --patience 10 \
  --itr 1 \
  --gpu 0 \
  --lradj cosine \
  --loss 'MAE' \
  --target "PV137"
  # --weight_decay 0.0005 \

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/SOLAR/ \
  --data_path solar_Alabama.csv  \
  --model_id solar_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --layers 1 \
  --num_clusters 64 \
  --id_dim 32 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 256 \
  --enc_in 137 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --des 'Exp' \
  --train_epochs 10\
  --patience 10 \
  --itr 1 \
  --gpu 0 \
  --lradj cosine \
  --loss 'MAE' \
  --target "PV137"
  # --weight_decay 0.0005 \

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/SOLAR/ \
  --data_path solar_Alabama.csv  \
  --model_id solar_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --layers 1 \
  --num_clusters 64 \
  --id_dim 32 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 256 \
  --enc_in 137 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --des 'Exp' \
  --train_epochs 10 \
  --patience 10 \
  --itr 1 \
  --gpu 0 \
  --lradj cosine \
  --loss 'MAE' \
  --target "PV137"
  # --weight_decay 0.0005 \

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/SOLAR/ \
  --data_path solar_Alabama.csv  \
  --model_id solar_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --layers 3 \
  --num_clusters 64 \
  --id_dim 32 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 256 \
  --enc_in 137 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --des 'Exp' \
  --train_epochs 50 \
  --patience 50 \
  --itr 1 \
  --gpu 0 \
  --lradj cosine \
  --loss 'MAE' \
  --target "PV137"
  # --weight_decay 0.0005 \