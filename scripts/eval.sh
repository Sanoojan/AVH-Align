# CUDA_VISIBLE_DEVICES=3 python eval.py \
#     --checkpoint_path checkpoints/Baseline-Trimmed-random.pt \
#     --features_path Features/AV1M-Trimmed_with_visual_map/test \
#     --metadata av1m_metadata/test_metadata_cleaned.csv \
#     --dataset AV1M > logs/Baseline_test_2_zero_shot_using_visual_map.log 2>&1 &


# python eval.py \
#     --checkpoint_path checkpoints/Baseline.pt \
#     --features_path Features/AV1M-Trimmed/test \
#     --metadata av1m_metadata/test_metadata_cleaned.csv \
#     --dataset AV1M > logs/Baseline-trained_test_on_trimmed.log 2>&1 &


# python eval.py \
#     --checkpoint_path checkpoints/Baseline-Trimmed.pt \
#     --features_path Features/AV1M-Trimmed/test \
#     --metadata av1m_metadata/test_metadata_cleaned.csv \
#     --dataset AV1M > logs/Baseline-trained-Trimmed_test_on_trimmed.log 2>&1 &

# python eval.py \
#     --checkpoint_path checkpoints/Baseline.pt \
#     --features_path Features/AV1M/test \
#     --metadata av1m_metadata/test_metadata_cleaned.csv \
#     --dataset AV1M > logs/Baseline-trained_test_on_untrimmed.log 2>&1 &


# python eval.py \
#     --checkpoint_path checkpoints/Baseline-Trimmed.pt \
#     --features_path Features/AV1M/test \
#     --metadata av1m_metadata/test_metadata_cleaned.csv \
#     --dataset AV1M > logs/Baseline-trained-Trimmed_test_on_untrimmed.log 2>&1 &

    

#  CUDA_VISIBLE_DEVICES=2 python eval_new.py \
#     --checkpoint_path checkpoints/AVH_A_V_simple_mlp_train_val_splits_train_from_real_and_synth_with_framewise_logsum_exp.pt \
#     --features_path Features/FakeAVCeleb-Trimmed \
#     --metadata /egr/research-sprintai/baliahsa/projects/AVH-Align/data/DeepfakeDatasets/FakeAVCeleb/video/meta_data.csv \
#     --model_name SimpleTemporalFusionAV_only \
#     --dataset FakeAVCeleb > logs/Evals/SimpleTemporalFusionAV_A_V_only_test_fakeavceleb_real_and_synth_with_framewise2.log 2>&1 &

#  CUDA_VISIBLE_DEVICES=5 python eval_new.py \
#     --checkpoint_path checkpoints/AVH_A_V_simple_mlp_train_val_splits_train_from_real_and_synth_with_framewise.pt \
#     --features_path Features/AV1M-Trimmed/test \
#     --metadata av1m_metadata/test_metadata_cleaned.csv \
#     --model_name SimpleTemporalFusionAV_only \
#     --dataset AV1M > logs/Evals/SimpleTemporalFusionAV_only_test_av1m_trimm_train_from_real_and_synth_with_framewise.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python eval_new.py \
#     --checkpoint_path checkpoints/AVH_A_M_simple_mlp_train_val_splits_train_from_real_and_synth_with_framewise_logsum_exp.pt \
#     --features_path Features/LavDF-Trimmed/test \
#     --metadata LavDF_metadata/test_metadata.csv \
#     --model_name SimpleTemporalFusionAV_only \
#     --dataset LavDF > logs/Evals/SimpleTemporalFusionAV_A_M_only_test_lavdf_real_and_synth_with_corrected_framewise_AP_0.375thresh.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python eval_new.py \
#     --checkpoint_path checkpoints/AVH_A_M_simple_mlp_train_val_splits_train_from_real_and_synth_with_framewise_logsum_exp.pt \
#     --features_path Features/AV1M-Trimmed/test \
#     --metadata av1m_metadata/test_metadata_cleaned.csv \
#     --model_name SimpleTemporalFusionAV_only \
#     --dataset AV1M > logs/Evals/SimpleTemporalFusionAV_A_M_only_test_AV1M_real_and_synth_with_corrected_framewise_AP_0.375thresh.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python eval_new_time.py \
    --checkpoint_path checkpoints/auvire_avdeepfake1m_supervised.pt \
    --features_path Features/AV1M-Trimmed/test \
    --metadata av1m_metadata/test_metadata_cleaned.csv \
    --model_name AuvireAVDeepfake1MLocalizer \
    --number_of_plots 20 \
    --frame_score_type sigmoid \
    --test_name fully_supervised_AuvireAVDeepfake1MLocalizer_50_prop_wo_real_boundary \
    --plot_framewise \
    --max_proposed_segments 50 \
    --dataset AV1M > logs/Evals_TIME/Ours/fully_supervised_AuvireAVDeepfake1MLocalizer_50_prop_wo_real_boundary2.log 2>&1 &


    # --plot_framewise \

# CUDA_VISIBLE_DEVICES=1 python eval_new.py \
#     --checkpoint_path checkpoints/AVH-Align_AV1M.pt \
#     --features_path Features/AV1M-Trimmed/test \
#     --metadata av1m_metadata/test_metadata_cleaned.csv \
#     --model_name FusionModel \
#     --dataset AV1M > logs/Evals/FusionModel_test_av1m_corrected_framewise.log 2>&1 &

