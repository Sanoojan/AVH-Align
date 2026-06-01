CUDA_VISIBLE_DEVICES=1 python eval_new_auvire_time.py \
  --features_path Features/AV1M-Base-Trimmed/test \
  --metadata av1m_metadata/test_metadata_cleaned.csv \
  --dataset avdeepfake1m \
  --trained_on lavdf \
  --feature_dim 768 \
  --plot_framewise \
  --checkpoint_path auvire/ckpt/auvire-lavdf/model.safetensors \
  --config_path auvire/ckpt/lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json \
  --test_name auvire_AV1M_trained_on_lavdf > logs/Evals_TIME/AUVIRE/auvire_AV1M_trained_on_lavdf.log 2>&1 &

  # CUDA_VISIBLE_DEVICES=3 python eval_new_auvire_time.py \
  # --features_path Features/AV1M-Trimmed/test \
  # --metadata av1m_metadata/test_metadata_cleaned.csv \
  # --dataset avdeepfake1m \
  # --trained_on avdeepfake1m \
  # --feature_dim 1024 \
  # --plot_framewise \
  # --checkpoint_path ckpt/auvire/hard_synth_trained/avdeepfake1m_b_avhubert_t_cnn_cnn_h_8_d_128_l_r1d1_w_15_o_subtraction_rl_r2d1u1s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.pth \
  # --config_path ckpt/auvire/hard_synth_trained/avdeepfake1m_b_avhubert_t_cnn_cnn_h_8_d_128_l_r1d1_w_15_o_subtraction_rl_r2d1u1s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json \
  # --test_name AUVIRE_hard_synth_trained > logs/Evals_TIME/AUVIRE/hard_synth_trained.log 2>&1 &

  #   CUDA_VISIBLE_DEVICES=1 python eval_new_auvire_time.py \
  # --features_path Features/LavDF-Base-Trimmed/test \
  # --metadata LavDF_metadata/test_metadata.csv \
  # --dataset lavdf \
  # --trained_on lavdf \
  # --feature_dim 768 \
  # --plot_framewise \
  # --checkpoint_path auvire/ckpt/auvire-lavdf/model.safetensors \
  # --config_path /egr/research-sprintai/baliahsa/projects/AVH-Align/auvire/ckpt/lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json \
  # --test_name AUVIRE_Original_LavDF_ckpt_test_on_LavDF_cvpr_test_set > logs/Evals_TIME/AUVIRE/Original_LavDF_ckpt_test_on_LavDF_cvpr_test_set_time.log 2>&1 &


  