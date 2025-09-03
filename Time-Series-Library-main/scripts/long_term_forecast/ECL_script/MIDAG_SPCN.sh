export CUDA_VISIBLE_DEVICES=0

model_name=MIDAG_SPCN

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --layers 1 \
  --num_clusters 32 \
  --id_dim 8 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 512 \
  --enc_in 321 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --train_epochs 10 \
  --patience 10 \
  --itr 1 \
  --gpu 0 \
  --lradj "type1" \
  --loss 'MAE' \
#   # --weight_decay 0.0003 \

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --layers 1 \
  --num_clusters 32 \
  --id_dim 8 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 512 \
  --enc_in 321 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --train_epochs 10 \
  --patience 10 \
  --itr 1 \
  --gpu 0 \
  --lradj "type1" \
  --loss 'MAE' \
# #   # --weight_decay 0.0003 \

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --layers 1 \
  --num_clusters 32 \
  --id_dim 8 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 512 \
  --enc_in 321 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --des 'Exp' \
  --train_epochs 20 \
  --patience 10 \
  --itr 1 \
  --gpu 0 \
  --lradj "type1" \
  --loss 'MAE' \
  # --weight_decay 0.0003 \

python -u run_midag_spcn.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/ljr/raw_files/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --layers 1 \
  --num_clusters 32 \
  --id_dim 8 \
  --cluster_dim 8 \
  --graph_dim 32 \
  --d_model 512 \
  --enc_in 321 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --des 'Exp' \
  --train_epochs 20 \
  --patience 10 \
  --itr 1 \
  --gpu 0 \
  --lradj "type1" \
  --loss 'MAE' \