CUDA_VISIBLE_DEVICES=5 python analysis_quad_final_comp.py \
  --dataset FakeAVCeleb \
  --features_root Features/FakeAVCeleb-Trimmed \
  --metadata data/DeepfakeDatasets/FakeAVCeleb/video/meta_data.csv \
  --checkpoint_path checkpoints/AVH-Align_AV1M.pt \
  --checkpoint_path_ours checkpoints/AVH_A_V_simple_mlp_train_val_splits_train_from_real_and_synth_with_framewise.pt