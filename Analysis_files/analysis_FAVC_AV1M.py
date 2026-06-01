import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from model import FusionModel, SimpleTemporalFusionAV_only

# Eg:python analysis_FAVC_AV1M.py --datasets FakeAVCeleb AV1M --video_types real fake_video_real_audio

VIDEO_TYPES = [
    "real",
    "fake_video_real_audio",
    "fake_video_fake_audio",
    "real_video_fake_audio",
]

FAVC_TYPE_MAP = {
    "RealVideo-RealAudio": "real",
    "FakeVideo-RealAudio": "fake_video_real_audio",
    "FakeVideo-FakeAudio": "fake_video_fake_audio",
    "RealVideo-FakeAudio": "real_video_fake_audio",
}

TYPE_LABELS = {
    "real": "Real",
    "fake_video_real_audio": "FVRA",
    "fake_video_fake_audio": "FVFA",
    "real_video_fake_audio": "RVFA",
}

TYPE_COLORS = {
    "real": "black",
    "fake_video_real_audio": "red",
    "fake_video_fake_audio": "blue",
    "real_video_fake_audio": "gray",
}

DATASET_STYLES = {
    "FakeAVCeleb": "-",
    "AV1M": "--",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 4 FakeAVCeleb and 4 AVDeepfake1M videos in the same graph."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["FakeAVCeleb", "AV1M"],
        choices=["FakeAVCeleb", "AV1M"],
        help="Dataset(s) to include in each plot.",
    )
    parser.add_argument(
        "--video_types",
        nargs="+",
        default=VIDEO_TYPES,
        choices=VIDEO_TYPES,
        help="Video type(s) to include in each plot.",
    )
    parser.add_argument("--favc_metadata", type=str, default="data/DeepfakeDatasets/FakeAVCeleb/video/meta_data.csv")
    parser.add_argument("--av1m_metadata", type=str, default="av1m_metadata/test_metadata.csv")
    parser.add_argument("--favc_features_root", type=str, default="Features/FakeAVCeleb-Trimmed")
    parser.add_argument("--av1m_features_root", type=str, default="Features/AV1M-Trimmed/test")
    parser.add_argument("--av1m_labels_root", type=str, default="Features/Labels/test")
    parser.add_argument("--output_dir", type=str, default="Plots/New_comparison_Vis_Aud_favc")
    parser.add_argument("--num_groups", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/AVH-Align_AV1M.pt")
    parser.add_argument(
        "--checkpoint_path_ours",
        type=str,
        default="checkpoints/AVH_A_V_simple_mlp_train_val_splits_train_from_real_and_synth_with_framewise_logsum_exp.pt",
    )
    return parser.parse_args()


def load_metadata(favc_metadata, av1m_metadata):
    favc_df = pd.read_csv(favc_metadata)
    favc_df["plot_type"] = favc_df["type"].map(FAVC_TYPE_MAP)
    favc_df = favc_df[favc_df["plot_type"].notna()].copy()

    av1m_df = pd.read_csv(av1m_metadata)
    av1m_df["plot_type"] = av1m_df["path"].apply(lambda x: x.split("/")[-1].replace(".mp4", ""))

    return {"FakeAVCeleb": favc_df, "AV1M": av1m_df}


def print_counts(dataset_dfs, selected_datasets, selected_video_types):
    for dataset_name in selected_datasets:
        df = dataset_dfs[dataset_name]
        print(f"\n{dataset_name} counts:")
        for video_type in selected_video_types:
            print(f"{TYPE_LABELS[video_type]}: {len(df[df['plot_type'] == video_type])}")


def validate_groups(dataset_dfs, selected_datasets, selected_video_types):
    missing = []
    for dataset_name in selected_datasets:
        df = dataset_dfs[dataset_name]
        for video_type in selected_video_types:
            if len(df[df["plot_type"] == video_type]) == 0:
                missing.append(f"{dataset_name}:{video_type}")

    if missing:
        raise ValueError(f"Metadata is missing required groups: {missing}")


def build_feature_path(dataset_name, row, args):
    if dataset_name == "AV1M":
        base = row["path"].replace(".mp4", "")
        return os.path.join(args.av1m_features_root, f"{base}.npz")

    base = row["path"].replace("FakeAVCeleb/", "", 1)
    filename = row["filename"].replace(".mp4", ".npz")
    return os.path.join(args.favc_features_root, base, filename)


def build_label_path(dataset_name, row, args):
    if dataset_name != "AV1M":
        return None

    base = row["path"].replace(".mp4", "")
    return os.path.join(args.av1m_labels_root, f"{base}_labels.npz")


def load_labels(label_path):
    if label_path is None or not os.path.exists(label_path):
        return None

    data = np.load(label_path, allow_pickle=True)
    if "framewise_labels" in data:
        labels = data["framewise_labels"]
    else:
        labels = list(data.values())[0]

    return labels.astype(int)


def build_labels(dataset_name, row, label_path, num_frames):
    if dataset_name == "FakeAVCeleb":
        label = 0 if row["plot_type"] == "real" else 1
        return np.full(num_frames, label, dtype=int)

    labels = load_labels(label_path)
    if labels is not None:
        labels = labels[:num_frames]
    return labels


def process_video_ours(model, device, visual_tensor, audio_tensor):
    visual_tensor = visual_tensor.to(device)
    audio_tensor = audio_tensor.to(device)

    features = torch.stack([visual_tensor, audio_tensor], dim=1)
    features = features.transpose(0, 1).unsqueeze(0)

    output = model(features)
    ours_logits = output.detach().cpu().squeeze(1).view(-1)
    score_ours = torch.logsumexp(ours_logits, dim=0)

    return score_ours.item(), ours_logits.numpy()


def process_video_baseline(model, device, visual_tensor, audio_tensor):
    visual_tensor = visual_tensor.to(device)
    audio_tensor = audio_tensor.to(device)

    output = model(visual_tensor, audio_tensor)
    baseline_logits = output.detach().cpu().squeeze(1)
    score_baseline = torch.logsumexp(-baseline_logits, dim=0)

    return score_baseline.item(), baseline_logits.numpy()


def compute_metrics(feature_path, fusion_model, simple_fusion_model, device):
    data = np.load(feature_path, allow_pickle=True)

    multimodal = torch.tensor(data["multimodal"]).float()
    visual = torch.tensor(data["visual"]).float()
    audio = torch.tensor(data["audio"]).float()

    multimodal_norm = F.normalize(multimodal, dim=-1)
    visual_norm = F.normalize(visual, dim=-1)
    audio_norm = F.normalize(audio, dim=-1)

    audio_multimodal = torch.sum(audio_norm * multimodal_norm, dim=-1)
    visual_multimodal = torch.sum(visual_norm * multimodal_norm, dim=-1)
    audio_vs_visual = torch.sum(audio_norm * visual_norm, dim=-1)

    score_baseline, baseline_logits = process_video_baseline(fusion_model, device, visual_norm, audio_norm)
    score_ours, logits_ours = process_video_ours(simple_fusion_model, device, visual_norm, audio_norm)

    return {
        "audio_multimodal": audio_multimodal.numpy(),
        "visual_multimodal": visual_multimodal.numpy(),
        "audio_vs_visual": audio_vs_visual.numpy(),
        "baseline": torch.tensor(baseline_logits).numpy(),
        "ours": -torch.tensor(logits_ours).numpy(),
        "score_ours": score_ours,
        "score_baseline": score_baseline,
        "num_frames": len(multimodal_norm),
    }


def sample_pack(dataset_dfs, selected_datasets, selected_video_types):
    pack = {}
    for dataset_name in selected_datasets:
        df = dataset_dfs[dataset_name]
        for video_type in selected_video_types:
            row = df[df["plot_type"] == video_type].sample(1).iloc[0].to_dict()
            pack[(dataset_name, video_type)] = row
    return pack


def plot_metric(ax, metric_name, ylabel, metrics_dict, labels_dict):
    for (dataset_name, video_type), metrics in metrics_dict.items():
        
        values = np.asarray(metrics[metric_name])
        frames = np.arange(len(values))
        label = f"{dataset_name} {TYPE_LABELS[video_type]}"
        ax.plot(
            frames,
            values,
            label=label,
            color=TYPE_COLORS[video_type],
            linestyle=DATASET_STYLES[dataset_name],
            linewidth=2,
        )

        labels = labels_dict.get((dataset_name, video_type))
        if labels is not None and len(labels) == len(values):
            fake_idx = np.where(labels == 1)[0]
            if len(fake_idx) > 0:
                ax.scatter(
                    fake_idx,
                    values[fake_idx],
                    color=TYPE_COLORS[video_type],
                    edgecolors="yellow",
                    marker="o",
                    s=35,
                    zorder=3,
                    alpha=0.8,
                )

    ax.set_title(metric_name.capitalize())
    ax.set_xlabel("Frame Index (trimmed)")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(fontsize=7, ncol=2)


def plot_auc(output_dir, all_labels, all_scores_ours, all_scores_baseline):
    all_labels = np.array(all_labels)
    all_scores_ours = np.array(all_scores_ours)
    all_scores_baseline = np.array(all_scores_baseline)

    if len(np.unique(all_labels)) < 2:
        print("Skipping AUC: labels contain only one class.")
        return

    if roc_auc_score(all_labels, all_scores_ours) < 0.5:
        all_scores_ours = -all_scores_ours

    if roc_auc_score(all_labels, all_scores_baseline) < 0.5:
        all_scores_baseline = -all_scores_baseline

    auc_ours = roc_auc_score(all_labels, all_scores_ours)
    auc_baseline = roc_auc_score(all_labels, all_scores_baseline)

    for name, scores in [("Ours", all_scores_ours), ("Baseline", all_scores_baseline)]:
        fpr, tpr, _ = roc_curve(all_labels, scores)
        for target in [0.01, 0.02, 0.05, 0.10]:
            idx = np.argmin(np.abs(fpr - target))
            print(f"{name} TPR @ {target * 100:.1f}% FPR: {tpr[idx]:.4f}")

    print("\n==============================")
    print("VIDEO LEVEL AUC COMPARISON")
    print("==============================")
    print(f"Our Score AUC     : {auc_ours:.4f}")
    print(f"Baseline AUC      : {auc_baseline:.4f}")
    print("==============================")

    fpr_ours, tpr_ours, _ = roc_curve(all_labels, all_scores_ours)
    fpr_base, tpr_base, _ = roc_curve(all_labels, all_scores_baseline)

    plt.figure()
    plt.plot(fpr_ours, tpr_ours, label=f"Ours (AUC={auc_ours:.3f})")
    plt.plot(fpr_base, tpr_base, label=f"Baseline (AUC={auc_baseline:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid()

    roc_path = os.path.join(output_dir, "auc_comparison.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print("ROC curve saved to:", roc_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fusion_model_weights = torch.load(args.checkpoint_path, weights_only=False)
    simple_fusion_weights = torch.load(args.checkpoint_path_ours, weights_only=False)

    fusion_model = FusionModel().to(device)
    fusion_model.load_state_dict(fusion_model_weights["state_dict"])
    fusion_model.eval()

    simple_fusion_model = SimpleTemporalFusionAV_only().to(device)
    simple_fusion_model.load_state_dict(simple_fusion_weights["state_dict"])
    simple_fusion_model.eval()

    selected_datasets = list(dict.fromkeys(args.datasets))
    selected_video_types = list(dict.fromkeys(args.video_types))
    expected_num_videos = len(selected_datasets) * len(selected_video_types)

    dataset_dfs = load_metadata(args.favc_metadata, args.av1m_metadata)
    print("Selected datasets:", ", ".join(selected_datasets))
    print("Selected video types:", ", ".join(selected_video_types))
    print_counts(dataset_dfs, selected_datasets, selected_video_types)
    validate_groups(dataset_dfs, selected_datasets, selected_video_types)

    all_labels = []
    all_scores_ours = []
    all_scores_baseline = []

    for idx in range(args.num_groups):
        video_pack = sample_pack(dataset_dfs, selected_datasets, selected_video_types)
        # breakpoint()
        metrics_dict = {}
        labels_dict = {}

        for key, row in video_pack.items():
            dataset_name, video_type = key
            feature_path = build_feature_path(dataset_name, row, args)
            label_path = build_label_path(dataset_name, row, args)

            if not os.path.exists(feature_path):
                print("Missing feature:", feature_path)
                continue

            try:
                metrics = compute_metrics(feature_path, fusion_model, simple_fusion_model, device)
                labels = build_labels(dataset_name, row, label_path, metrics["num_frames"])

                metrics_dict[key] = metrics
                labels_dict[key] = labels

                label = 0 if video_type == "real" else 1
                all_labels.append(label)
                all_scores_ours.append(metrics["score_ours"])
                all_scores_baseline.append(metrics["score_baseline"])
            except Exception as exc:
                print("Error:", feature_path, exc)
        # breakpoint()
        if len(metrics_dict) != expected_num_videos:
            continue

        plt.figure(figsize=(18, 7))

        plt.subplot(1, 2, 1)
        plot_metric(plt.gca(), "baseline", "Logits", metrics_dict, labels_dict)

        plt.subplot(1, 2, 2)
        plot_metric(plt.gca(), "ours", "Negated Logits", metrics_dict, labels_dict)

        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f"favc_av1m_compare_{idx}.jpeg")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved {save_path}")

    plot_auc(args.output_dir, all_labels, all_scores_ours, all_scores_baseline)
    print("Done")


if __name__ == "__main__":
    main()
