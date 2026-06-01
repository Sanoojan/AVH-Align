import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import roc_curve
from tqdm import tqdm

from utils import seed_run

AUVIRE_ROOT = Path(__file__).resolve().parent / "auvire"
if str(AUVIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(AUVIRE_ROOT))

from src.models import Model  # noqa: E402
from src.post_process import soft_nms_torch_parallel  # noqa: E402

try:
    torch.backends.mha.set_fastpath_enabled(False)
except AttributeError:
    pass

OUT_FEATURES_PATH = "Out_Features_trimmed_with_visual_map"
os.makedirs(OUT_FEATURES_PATH, exist_ok=True)



# change the ckpt path here
CKPT_ROOT = Path(__file__).resolve().parent / "auvire"
CKPT_ROOT = Path(__file__).resolve().parent 
DEFAULT_AUVIRE_CONFIGS = {
    "avdeepfake1m": CKPT_ROOT
    / "ckpt"
    / "avdeepfake1m_b_avhubert_t_cnn_cnn_h_8_d_128_l_r1d1_w_15_o_subtraction_rl_r2d1u1s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json",
    "lavdf": CKPT_ROOT
    / "ckpt"
    / "lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json",
}

DEFAULT_AUVIRE_CHECKPOINTS = {
    # "avdeepfake1m": CKPT_ROOT / "ckpt" / "auvire-avdeepfake1m" / "model.safetensors",
    "avdeepfake1m": CKPT_ROOT / "ckpt" / "avdeepfake1m_b_avhubert_t_cnn_cnn_h_8_d_128_l_r1d1_w_15_o_subtraction_rl_r2d1u1s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.pth",
    "lavdf": CKPT_ROOT / "ckpt" / "auvire-lavdf" / "model.safetensors",
}


class MissingFeatureError(RuntimeError):
    pass


class AuVireFeatureAdapter(nn.Module):
    def __init__(self, auvire_model, input_dim, auvire_dim=768):
        super().__init__()
        if input_dim != 1024:
            raise ValueError(f"Projection adapter is only supported for 1024-dim features, got {input_dim}.")
        self.auvire_model = auvire_model
        self.video_projection = nn.Linear(input_dim, auvire_dim)
        self.audio_projection = nn.Linear(input_dim, auvire_dim)

    def forward(self, video_features, audio_features):
        video_features = self.video_projection(video_features)
        audio_features = self.audio_projection(audio_features)
        return self.auvire_model([video_features, audio_features])


def to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def compute_thresholded_metrics(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    if len(np.unique(labels)) < 2:
        threshold = np.nan
        preds = np.zeros_like(labels)
        auc = np.nan
    else:
        auc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        finite = np.isfinite(thresholds)
        if finite.any():
            best_idx = np.argmax(tpr[finite] - fpr[finite])
            threshold = thresholds[finite][best_idx]
        else:
            threshold = 0.5
        preds = (scores >= threshold).astype(int)

    return {
        "auc": auc,
        "threshold": threshold,
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }


def binary_segments(values):
    values = np.asarray(values).astype(int).reshape(-1)
    segments = []
    start = None

    for idx, value in enumerate(values):
        if value == 1 and start is None:
            start = idx
        elif value == 0 and start is not None:
            segments.append((start, idx))
            start = None

    if start is not None:
        segments.append((start, len(values)))

    return segments


def segment_iou(segment_a, segment_b):
    start = max(segment_a[0], segment_b[0])
    end = min(segment_a[1], segment_b[1])
    intersection = max(0.0, end - start)
    union = max(segment_a[1], segment_b[1]) - min(segment_a[0], segment_b[0])
    return intersection / union if union > 0 else 0.0


def count_iou_matches(pred_segments, gt_segments, iou_threshold):
    matched_gt = set()
    true_positives = 0

    for pred_segment in pred_segments:
        best_gt_idx = None
        best_iou = 0.0
        for gt_idx, gt_segment in enumerate(gt_segments):
            if gt_idx in matched_gt:
                continue
            iou = segment_iou(pred_segment, gt_segment)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            true_positives += 1

    return true_positives


def rasterize_proposals(proposals, length):
    scores = np.zeros(length, dtype=np.float32)
    for score, start, end in proposals:
        start_idx = max(0, int(np.floor(start)))
        end_idx = min(length, int(np.ceil(end)))
        if end_idx > start_idx:
            scores[start_idx:end_idx] = np.maximum(scores[start_idx:end_idx], float(score))
    return scores


def flatten_framewise(framewise_scores, framewise_labels):
    flat_scores = []
    flat_labels = []

    for scores, labels in zip(framewise_scores, framewise_labels):
        scores = np.asarray(scores).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        if len(scores) != len(labels):
            print(f"Warning: Inconsistent lengths in framewise data: {len(scores)} vs {len(labels)}")
        keep = min(len(scores), len(labels))
        flat_scores.append(scores[:keep])
        flat_labels.append(labels[:keep])

    if not flat_scores:
        return np.array([]), np.array([])
    return np.concatenate(flat_scores), np.concatenate(flat_labels)


def prepare_segment_proposals(proposals, label_length=None, confidence_threshold=None):
    proposals = np.asarray(proposals, dtype=float)
    if proposals.size == 0:
        return np.empty((0, 3), dtype=float)
    proposals = proposals.reshape(-1, 3)
    proposals = proposals[np.isfinite(proposals).all(axis=1)]
    if label_length is not None:
        proposals[:, 1] = np.clip(proposals[:, 1], 0, label_length)
        proposals[:, 2] = np.clip(proposals[:, 2], 0, label_length)
    proposals = proposals[proposals[:, 2] > proposals[:, 1]]
    if confidence_threshold is not None:
        proposals = proposals[proposals[:, 0] >= confidence_threshold]
    order = np.argsort(-proposals[:, 0], kind="mergesort")
    return proposals[order]


def compute_segment_ap_at_iou_thresholds(
    framewise_labels,
    proposal_segments,
    confidence_threshold,
    iou_thresholds=(0.5, 0.75, 0.9, 0.95),
):
    metrics = {}

    for iou_threshold in iou_thresholds:
        true_positives = 0
        pred_count = 0

        for labels, proposals in zip(framewise_labels, proposal_segments):
            labels = np.asarray(labels).astype(int).reshape(-1)
            gt_segments = binary_segments(labels)
            proposals = prepare_segment_proposals(
                proposals,
                label_length=len(labels),
                confidence_threshold=confidence_threshold,
            )
            pred_segments = [(start, end) for _, start, end in proposals]

            true_positives += count_iou_matches(pred_segments, gt_segments, iou_threshold)
            pred_count += len(pred_segments)

        metrics[iou_threshold] = true_positives / pred_count if pred_count > 0 else 0.0

    return metrics


def compute_segment_ar_at_proposal_counts(
    framewise_labels,
    proposal_segments,
    proposal_counts=(100, 50, 30, 20, 10, 5),
    iou_thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
):
    metrics = {}

    for proposal_count in proposal_counts:
        recalls = []
        for iou_threshold in iou_thresholds:
            true_positives = 0
            gt_count = 0

            for labels, proposals in zip(framewise_labels, proposal_segments):
                labels = np.asarray(labels).astype(int).reshape(-1)
                gt_segments = binary_segments(labels)
                proposals = prepare_segment_proposals(proposals, label_length=len(labels))[:proposal_count]
                pred_segments = [(start, end) for _, start, end in proposals]

                true_positives += count_iou_matches(pred_segments, gt_segments, iou_threshold)
                gt_count += len(gt_segments)

            recalls.append(true_positives / gt_count if gt_count > 0 else 0.0)

        metrics[int(proposal_count)] = float(np.mean(recalls)) if recalls else 0.0

    return metrics


def sanitize_filename(value):
    value = str(value)
    value = value.replace(os.sep, "_")
    if os.altsep:
        value = value.replace(os.altsep, "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._") or "video"


def framewise_plot_performance(scores, labels, ranking_threshold):
    scores = np.asarray(scores).astype(float).reshape(-1)
    labels = np.asarray(labels).astype(int).reshape(-1)
    keep = min(len(scores), len(labels))
    scores = scores[:keep]
    labels = labels[:keep]

    if keep == 0:
        return -np.inf
    if len(np.unique(labels)) > 1:
        return average_precision_score(labels, scores)

    predictions = (scores >= ranking_threshold).astype(int)
    return accuracy_score(labels, predictions)


def select_framewise_plot_items(framewise_scores, framewise_labels, number_of_plots, ranking_threshold):
    performances = [
        (idx, framewise_plot_performance(scores, labels, ranking_threshold))
        for idx, (scores, labels) in enumerate(zip(framewise_scores, framewise_labels))
    ]
    plot_count = min(max(number_of_plots, 0), len(performances))

    worst = sorted(performances, key=lambda item: (item[1], item[0]))[:plot_count]
    best = sorted(performances, key=lambda item: (-item[1], item[0]))[:plot_count]

    rng = np.random.default_rng(0)
    random_indices = rng.choice(len(performances), size=plot_count, replace=False) if plot_count else []
    random_items = [(int(idx), performances[int(idx)][1]) for idx in random_indices]

    return (("W", worst), ("B", best), ("R", random_items))


def plot_framewise_predictions(
    framewise_scores,
    framewise_labels,
    path_names,
    dataset,
    checkpoint_path,
    number_of_plots,
    thresholds=(0.2, 0.5, 0.85),
    ranking_threshold=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checkpoint_name = Path(checkpoint_path).stem
    plot_dir = os.path.join(
        "Plots",
        "framewise_predictions",
        f"{sanitize_filename(dataset)}_{sanitize_filename(checkpoint_name)}",
    )
    os.makedirs(plot_dir, exist_ok=True)

    if ranking_threshold is None:
        ranking_threshold = thresholds[1] if len(thresholds) > 1 else 0.5

    saved_count = 0
    for prefix, plot_items in select_framewise_plot_items(
        framewise_scores,
        framewise_labels,
        number_of_plots,
        ranking_threshold,
    ):
        for rank, (idx, performance) in enumerate(plot_items):
            scores = np.asarray(framewise_scores[idx]).astype(float).reshape(-1)
            labels = np.asarray(framewise_labels[idx]).astype(int).reshape(-1)
            keep = min(len(scores), len(labels))
            scores = scores[:keep]
            labels = labels[:keep]
            frames = np.arange(keep)

            fig, axes = plt.subplots(
                len(thresholds) + 2,
                1,
                figsize=(16, 2.2 * (len(thresholds) + 2)),
                sharex=True,
            )

            video_name = path_names[idx] if idx < len(path_names) else f"video_{idx}"
            fig.suptitle(f"{video_name} | score={performance:.4f}", fontsize=12)

            axes[0].step(frames, labels, where="post", color="black", linewidth=1.5)
            axes[0].set_ylabel("GT")
            axes[0].set_ylim(-0.1, 1.1)
            axes[0].grid(True, alpha=0.25)

            axes[1].plot(frames, scores, color="#1f77b4", linewidth=1.2)
            for threshold in thresholds:
                axes[1].axhline(threshold, linestyle="--", linewidth=1.0, label=f"{threshold:.2f}")
            axes[1].set_ylabel("Score")
            axes[1].legend(title="Threshold", loc="upper right")
            axes[1].grid(True, alpha=0.25)

            for axis, threshold in zip(axes[2:], thresholds):
                predictions = (scores >= threshold).astype(int)
                axis.step(frames, labels, where="post", color="black", linewidth=1.2, label="GT")
                axis.step(frames, predictions, where="post", color="#d62728", linewidth=1.0, alpha=0.85, label="Pred")
                axis.set_ylabel(f"@{threshold:.2f}")
                axis.set_ylim(-0.1, 1.1)
                axis.grid(True, alpha=0.25)
                axis.legend(loc="upper right")

            axes[-1].set_xlabel("Frame")
            plt.tight_layout(rect=(0, 0, 1, 0.97))

            safe_video_name = sanitize_filename(Path(str(video_name)).stem)
            save_path = os.path.join(plot_dir, f"{prefix}_{rank:04d}_{safe_video_name}.png")
            plt.savefig(save_path, dpi=200)
            plt.close(fig)
            saved_count += 1

    print(f"Saved {saved_count} framewise prediction plots to {plot_dir}")


def print_metric_block(title, metrics):
    print(f"\n{title}")
    print("=" * len(title))
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Threshold: {metrics['threshold']:.6f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")


def print_ap_ar_block(title, ap_metrics, ar_metrics):
    print(f"\n{title}")
    print("=" * len(title))
    for iou_threshold, value in ap_metrics.items():
        print(f"AP@{iou_threshold:g}: {value:.4f}")
    for proposal_count, value in ar_metrics.items():
        print(f"AR@{proposal_count}: {value:.4f}")


def load_auvire_config(config_path):
    with open(config_path, "r") as handle:
        payload = json.load(handle)
    return payload["config"] if "config" in payload else payload


def build_factor(configuration):
    nlayers = configuration["model"]["encoder"]["nlayers"]
    return [1] * nlayers["retain"] + [2 ** (i + 1) for i in range(nlayers["downsample"])]


def build_auvire_model(configuration, device, input_dim=768):
    factor = build_factor(configuration)
    model_cfg = configuration["model"]
    model = Model(
        max_length=configuration["dataset"]["params"]["max_length"],
        d_model=model_cfg["d_model"],
        win_size=model_cfg["win_size"],
        num_heads=model_cfg["num_heads"],
        operation=model_cfg["operation"],
        reconstruction=model_cfg["reconstruction"],
        encoder=model_cfg["encoder"],
        dropout=model_cfg["dropout"],
        use_ln=model_cfg["conv"]["use_ln"],
        use_rl=model_cfg["conv"]["use_rl"],
        use_do=model_cfg["conv"]["use_do"],
        model_type=model_cfg["type"],
        factor=factor,
        device=device,
    )
    if input_dim == 768:
        return model.to(device), factor
    if input_dim == 1024:
        return AuVireFeatureAdapter(model, input_dim=input_dim).to(device), factor
    raise ValueError(f"AUVIRE supports 768-dim features directly or 1024-dim features with projection, got {input_dim}.")


def normalize_state_dict_keys(state_dict):
    candidates = [state_dict]
    for prefix in ("model.", "module."):
        if any(key.startswith(prefix) for key in state_dict):
            candidates.append({key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in state_dict.items()})
    return candidates


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading AUVIRE .safetensors checkpoints requires safetensors. "
                "Install it or pass a .pth checkpoint with --checkpoint_path."
            ) from exc
        state_dict = load_file(checkpoint_path, device=str(device))
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    last_error = None
    for candidate in normalize_state_dict_keys(state_dict):
        try:
            model.load_state_dict(candidate)
            return
        except RuntimeError as exc:
            last_error = exc
    raise last_error


def resolve_feature_path(args, row):
    candidates = []
    feature_root = Path(args.features_path)
    # breakpoint()
    if "feature_path" in row and pd.notna(row["feature_path"]):
        candidates.append(Path(row["feature_path"]))

    if "path" in row and pd.notna(row["path"]):
        rel = str(row["path"])
        candidates.extend(
            [
                feature_root / rel.replace(".mp4", ".npz"),
                feature_root / rel.replace(".mp4", "") / "features.npz",
                feature_root / rel / "features.npz",
            ]
        )

    if "filename" in row and pd.notna(row["filename"]):
        filename = str(row["filename"])
        base = Path(str(row.get("path", ""))).as_posix().replace("FakeAVCeleb/", "")
        candidates.extend(
            [
                feature_root / base / filename.replace(".mp4", ".npz"),
                feature_root / base / filename.replace(".mp4", "") / "features.npz",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not resolve feature file. Tried: " + ", ".join(str(x) for x in candidates))


def get_npz_array(data, keys, path, expected_dim=768):
    available = []
    fallback = None

    for key in keys:
        if key not in data:
            continue
        array = np.asarray(data[key])
        available.append(f"{key}{array.shape}")
        if array.ndim == 2 and array.shape[-1] == expected_dim:
            return array
        if fallback is None:
            fallback = (key, array)

    if fallback is not None:
        key, array = fallback
        raise MissingFeatureError(
            f"{path} has key '{key}' with shape {array.shape}, but AUVIRE expects "
            f"{expected_dim}-dimensional video/audio features. "
            "Use --feature_dim 1024 for AVH 1024-dimensional features trained with the projection adapter, "
            "or --feature_dim 768 for original AUVIRE/AV-HuBERT features. "
            f"Available matching keys: {available}"
        )

    raise MissingFeatureError(f"{path} is missing all expected keys: {keys}. Available keys: {list(data.keys())}")


def load_features(path, feature_dim=768):
    data = np.load(path, allow_pickle=True)
    video = get_npz_array(data, ["visual", "video_features", "visual_features", "multimodal", "video"], path, expected_dim=feature_dim)
    audio = get_npz_array(data, ["audio_features", "audio"], path, expected_dim=feature_dim)
    return np.asarray(video, dtype=np.float32), np.asarray(audio, dtype=np.float32)


def infer_feature_dim(feature_path):
    data = np.load(feature_path, allow_pickle=True)
    for key in ["video_features", "visual_features", "multimodal", "visual", "video", "audio_features", "audio"]:
        if key in data:
            array = np.asarray(data[key])
            if array.ndim == 2:
                return int(array.shape[-1])
    raise MissingFeatureError(f"{feature_path} does not contain a feature array with at least 2 dimensions.")


def pad_or_trim_features(video_features, audio_features, max_length):
    t = min(video_features.shape[0], audio_features.shape[0], max_length)
    video = torch.zeros((max_length, video_features.shape[1]), dtype=torch.float32)
    audio = torch.zeros((max_length, audio_features.shape[1]), dtype=torch.float32)
    video[:t] = torch.from_numpy(video_features[:t])
    audio[:t] = torch.from_numpy(audio_features[:t])
    return video.unsqueeze(0), audio.unsqueeze(0), t


def resolve_label_path(feature_path):
    feature_path = Path(feature_path)
    candidates = [
        feature_path.with_name(feature_path.stem + "_labels.npz"),
        feature_path.parent / "labels.npz",
        feature_path.parent / "features_labels.npz",
    ]
    if feature_path.name == "features.npz":
        candidates.append(feature_path.parent / "framewise_labels.npz")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def get_row_label(row, dataset):
    if "label" in row and pd.notna(row["label"]):
        return int(row["label"])
    if dataset == "FakeAVCeleb" and "type" in row:
        return 0 if row["type"] == "RealVideo-RealAudio" else 1
    raise KeyError("Metadata must contain a label column, or FakeAVCeleb type column.")


def load_framewise_labels(feature_path, row, output_length, dataset):
    label_path = resolve_label_path(feature_path)
    if label_path is not None:
        label_data = np.load(label_path, allow_pickle=True)
        if "framewise_labels" in label_data:
            labels = np.asarray(label_data["framewise_labels"]).astype(int)
        elif "labels" in label_data:
            labels = np.asarray(label_data["labels"]).astype(int)
        else:
            raise KeyError(f"{label_path} does not contain framewise_labels or labels")
    else:
        video_label = get_row_label(row, dataset)
        labels = np.full(output_length, video_label, dtype=int)

    labels = labels.reshape(-1)
    if len(labels) > output_length:
        labels = labels[:output_length]
    elif len(labels) < output_length:
        labels = np.pad(labels, (0, output_length - len(labels)), constant_values=0)
    return labels


def transform_auvire_predictions(predictions, max_length, factor, metrics_device):
    sigma, t1, t2, fps = 0.7234, 0.1968, 0.4123, 25
    idx = torch.arange(0, max_length).to(metrics_device)

    if isinstance(predictions, list):
        idx = torch.cat([idx[:: factor[i]] for i, _ in enumerate(predictions)])
        predictions = torch.cat(predictions, dim=1)

    predictions = predictions.to(metrics_device)
    predictions[:, :, 0] = torch.sigmoid(predictions[:, :, 0])
    predictions[:, :, 1] = torch.clamp(idx - predictions[:, :, 1], min=0.0)
    predictions[:, :, 2] = torch.clamp(idx + predictions[:, :, 2], max=max_length)
    _, indexes = torch.sort(predictions[:, :, 0], dim=1, descending=True)
    first_indices = torch.arange(predictions.shape[0])[:, None]
    predictions = predictions[first_indices, indexes]
    return soft_nms_torch_parallel(predictions, sigma, t1, t2, fps, metrics_device)


def process_video(feature_path, model, max_length, factor, device, metrics_device, feature_dim):
    video_features, audio_features = load_features(feature_path, feature_dim=feature_dim)
    video_tensor, audio_tensor, valid_length = pad_or_trim_features(video_features, audio_features, max_length)
    video_tensor = video_tensor.to(device)
    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        if isinstance(model, AuVireFeatureAdapter):
            predictions, _ = model(video_tensor, audio_tensor)
        else:
            predictions, _ = model([video_tensor, audio_tensor])

    proposals = transform_auvire_predictions(
        [prediction.detach().cpu() for prediction in predictions],
        max_length=max_length,
        factor=factor,
        metrics_device=metrics_device,
    )[0].detach().cpu().numpy()

    proposals = proposals[np.isfinite(proposals).all(axis=1)]
    proposals = proposals[proposals[:, 2] > proposals[:, 1]]
    video_score = float(proposals[:, 0].max()) if len(proposals) else 0.0
    return video_score, proposals, valid_length


def main(args):
    seed_run()

    trained_on = args.trained_on.lower()
    config_path = Path(args.config_path) if args.config_path else DEFAULT_AUVIRE_CONFIGS[trained_on]
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else DEFAULT_AUVIRE_CHECKPOINTS[trained_on]

    print(f"Evaluating AUVIRE on {args.dataset} with weights at {checkpoint_path} ...")
    configuration = load_auvire_config(config_path)
    max_length = configuration["dataset"]["params"]["max_length"]

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    metrics_device = torch.device(args.metrics_device)
    metadata = pd.read_csv(args.metadata)
    first_feature_path = resolve_feature_path(args, metadata.iloc[0])
    feature_dim = args.feature_dim or infer_feature_dim(first_feature_path)
    print(f"Using feature_dim={feature_dim} ({'1024 with projection' if feature_dim == 1024 else 'original AUVIRE' if feature_dim == 768 else 'unsupported'}).")
    model, factor = build_auvire_model(configuration, device, input_dim=feature_dim)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    outputs = []
    ground_truths = []
    framewise_scores = []
    framewise_labels = []
    proposal_segments = []
    path_names = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        feature_path = resolve_feature_path(args, row)
        label = get_row_label(row, args.dataset)
        score, proposals, valid_length = process_video(feature_path, model, max_length, factor, device, metrics_device, feature_dim)
        labels = load_framewise_labels(feature_path, row, valid_length, args.dataset)
        proposals = proposals.copy()
        proposals[:, 1] = np.clip(proposals[:, 1], 0, len(labels))
        proposals[:, 2] = np.clip(proposals[:, 2], 0, len(labels))
        raster_scores = rasterize_proposals(proposals, len(labels))

        outputs.append(score)
        ground_truths.append(label)
        framewise_scores.append(raster_scores)
        framewise_labels.append(labels)
        proposal_segments.append(proposals)
        path_names.append(str(feature_path))

    outputs = np.asarray(outputs, dtype=float)
    ground_truths = np.asarray(ground_truths, dtype=int)

    if args.plot_framewise:
        plot_framewise_predictions(
            framewise_scores,
            framewise_labels,
            path_names,
            args.dataset,
            checkpoint_path,
            args.number_of_plots,
            ranking_threshold=args.proposal_confidence_threshold,
        )

    flat_framewise_scores, flat_framewise_labels = flatten_framewise(framewise_scores, framewise_labels)

    video_ap = average_precision_score(ground_truths, outputs) if len(np.unique(ground_truths)) > 1 else np.nan
    framewise_ap = (
        average_precision_score(flat_framewise_labels, flat_framewise_scores)
        if len(np.unique(flat_framewise_labels)) > 1
        else np.nan
    )
    video_metrics = compute_thresholded_metrics(ground_truths, outputs)
    framewise_metrics = compute_thresholded_metrics(flat_framewise_labels, flat_framewise_scores)
    segment_ap = compute_segment_ap_at_iou_thresholds(
        framewise_labels,
        proposal_segments,
        confidence_threshold=args.proposal_confidence_threshold,
        iou_thresholds=tuple(args.iou_thresholds),
    )
    segment_ar = compute_segment_ar_at_proposal_counts(
        framewise_labels,
        proposal_segments,
        proposal_counts=tuple(args.proposal_counts),
        iou_thresholds=tuple(args.ar_iou_thresholds),
    )

    print(f"Video-wise AP: {video_ap:.4f}")
    print(f"Framewise AP: {framewise_ap:.4f}")
    print_metric_block("VIDEO-WISE METRICS", video_metrics)
    print_metric_block("FRAMEWISE METRICS", framewise_metrics)
    print(f"Proposal confidence threshold: {args.proposal_confidence_threshold:.2f}")
    print_ap_ar_block("FRAMEWISE SEGMENT AP/AR METRICS", segment_ap, segment_ar)

    if len(np.unique(ground_truths)) > 1:
        fpr, tpr, _ = roc_curve(ground_truths, outputs)
        for target in [0.05, 0.10]:
            idx = np.argmin(np.abs(fpr - target))
            print(f"TPR @ {target * 100:.1f}% FPR: {tpr[idx]:.4f}")

    np.save(os.path.join(OUT_FEATURES_PATH, f"{args.test_name}_auvire_outputs.npy"), outputs)
    np.save(os.path.join(OUT_FEATURES_PATH, f"{args.test_name}_auvire_ground_truths.npy"), ground_truths)
    np.save(
        os.path.join(OUT_FEATURES_PATH, f"{args.test_name}_auvire_framewise_scores.npy"),
        np.asarray(framewise_scores, dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(OUT_FEATURES_PATH, f"{args.test_name}_auvire_framewise_labels.npy"),
        np.asarray(framewise_labels, dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(OUT_FEATURES_PATH, f"{args.test_name}_auvire_proposals.npy"),
        np.asarray(proposal_segments, dtype=object),
        allow_pickle=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AUVIRE on a deepfake dataset")
    parser.add_argument("--features_path", type=str, default="data/AV-Deepfake1M_emb_avhubert/val")
    parser.add_argument("--metadata", type=str, default="av1m_metadata/test_metadata.csv")
    parser.add_argument("--dataset", type=str, default="AV1M")
    parser.add_argument("--trained_on", choices=("avdeepfake1m", "lavdf"), default="avdeepfake1m")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--test_name", type=str, default="debug")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--metrics_device", type=str, default="cpu")
    parser.add_argument("--feature_dim", type=int, choices=[768, 1024], default=1024, help="Feature dimension: 768 uses original AUVIRE, 1024 adds projection layers. Defaults to auto-detect from the first feature file.")
    parser.add_argument("--proposal_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--iou_thresholds", type=float, nargs="+", default=[0.5, 0.75, 0.9, 0.95])
    parser.add_argument("--proposal_counts", type=int, nargs="+", default=[100, 50, 30, 20, 10, 5])
    parser.add_argument("--ar_iou_thresholds", type=float, nargs="+", default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    parser.add_argument("--plot_framewise", action="store_true", help="Save framewise ground-truth and prediction plots.")
    parser.add_argument(
        "--number_of_plots",
        type=int,
        default=20,
        help="Number of framewise prediction plots to save when --plot_framewise is set.",
    )
    args = parser.parse_args()
    main(args)
