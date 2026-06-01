# CUDA_VISIBLE_DEVICES=2 python train.py --name=Visual_map_try1 --data_root_path=Features/AV1M-Trimmed_with_visual_map --metadata_root_path=av1m_metadata > logs/Visual_map_try1.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 python train_sup.py --name=AVH_A_V_simple_mlp_train_val_splits_video_wise --data_root_path=Features/AV1M-Trimmed --metadata_root_path=av1m_metadata > logs/AVH_A_V_M_simple_mlp_train_val_splits_video_wise.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python train_sup.py --name=AVH_A_M_simple_mlp_real_synth_framewise_logsum_exp_std_loss --data_root_path=Features/AV1M --data_val_root_path=Features/AV1M-Trimmed --metadata_root_path=av1m_metadata > logs/Trains/AVH_A_M_simple_mlp_real_synth_framewise_logsum_exp_std_loss.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python train_sup_new.py --config Configs/train_sup_new.yaml > logs/Trains/Input_hard_synth_supervised_A_M_SimpleTemporalFusion_weight_20_0.log 2>&1 &


# CUDA_VISIBLE_DEVICES=1 python train_auvire.py \
#     --config Configs/train_auvire_synth.yaml\
#     --print_config > logs/Trains/AUVIRE_synth_trained.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 python train_auvire.py \
#   --config Configs/train_auvire_hard_synth.yaml \
#   --set data.train_metadata_file=train_metadata.csv \
#   --synthesize_at_feature \
#   --synthesize_prob 0.5 \
#   --synthstyle hard \
#   --hard_min_start_deviation 10 \
#   --hard_max_start_deviation 30 > logs/Trains/AUVIRE_hard_synth_trained.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 python train_auvire.py \
#   --config Configs/train_auvire_hard_synth.yaml > logs/Trains/AUVIRE_input_hard_synth_trained.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python train_sup_new.py \
#   --config Configs/train_hard_synth.yaml \
#   --synthesize_at_feature \
#   --synthesize_prob 0.5 \
#   --synthstyle hard \
#   --hard_min_start_deviation 10 \
#   --hard_max_start_deviation 30 \
#   --synth_modalities random \
#   > logs/Trains/train_sup_new_real_hard_synth.log 2>&1 & 

# CUDA_VISIBLE_DEVICES=0 python train_sup_new.py \
#   --config Configs/train_hard_synth.yaml \
#   --synthesize_at_feature \
#   --synthesize_prob 0.5 \
#   --synthstyle hard \
#   --hard_min_start_deviation 10 \
#   --hard_max_start_deviation 30 \
#   --synth_modalities random \
#   > logs/Trains/train_sup_new_real_hard_synth.log 2>&1 & 

CUDA_VISIBLE_DEVICES=3 python train_sup_new.py \
  --config Configs/train_sup_new_auvire_avdeepfake1m.yaml \
  > logs/Trains/auvire_avdeepfake1m_hard_synth_supervised.log 2>&1 &