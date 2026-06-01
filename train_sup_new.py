import argparse
import math
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
except ModuleNotFoundError:
    def accuracy_score(labels, preds):
        labels = np.asarray(labels).astype(int)
        preds = np.asarray(preds).astype(int)
        return float(np.mean(labels == preds)) if labels.size else 0.0

    def precision_score(labels, preds, zero_division=0):
        labels = np.asarray(labels).astype(int)
        preds = np.asarray(preds).astype(int)
        tp = np.sum((labels == 1) & (preds == 1))
        fp = np.sum((labels == 0) & (preds == 1))
        denom = tp + fp
        return float(tp / denom) if denom else float(zero_division)

    def recall_score(labels, preds, zero_division=0):
        labels = np.asarray(labels).astype(int)
        preds = np.asarray(preds).astype(int)
        tp = np.sum((labels == 1) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))
        denom = tp + fn
        return float(tp / denom) if denom else float(zero_division)

    def f1_score(labels, preds, zero_division=0):
        precision = precision_score(labels, preds, zero_division=zero_division)
        recall = recall_score(labels, preds, zero_division=zero_division)
        denom = precision + recall
        return float(2.0 * precision * recall / denom) if denom else float(zero_division)

    def roc_auc_score(labels, scores):
        labels = np.asarray(labels).astype(int)
        scores = np.asarray(scores).astype(float)
        positives = labels == 1
        negatives = labels == 0
        n_pos = int(np.sum(positives))
        n_neg = int(np.sum(negatives))
        if n_pos == 0 or n_neg == 0:
            return np.nan
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        ranks = np.empty(len(scores), dtype=float)
        start = 0
        while start < len(scores):
            end = start + 1
            while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
            start = end
        pos_rank_sum = float(np.sum(ranks[positives]))
        return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    def average_precision_score(labels, scores):
        labels = np.asarray(labels).astype(int)
        scores = np.asarray(scores).astype(float)
        n_pos = int(np.sum(labels == 1))
        if n_pos == 0:
            return np.nan
        order = np.argsort(-scores, kind="mergesort")
        sorted_labels = labels[order]
        true_positives = np.cumsum(sorted_labels == 1)
        ranks = np.arange(1, len(sorted_labels) + 1)
        precision = true_positives / ranks
        return float(np.sum(precision[sorted_labels == 1]) / n_pos)

    def roc_curve(labels, scores):
        labels = np.asarray(labels).astype(int).reshape(-1)
        scores = np.asarray(scores).astype(float).reshape(-1)
        order = np.argsort(-scores, kind="mergesort")
        sorted_labels = labels[order]
        sorted_scores = scores[order]
        distinct = np.where(np.diff(sorted_scores))[0]
        threshold_idxs = np.r_[distinct, sorted_labels.size - 1]
        true_positives = np.cumsum(sorted_labels == 1)[threshold_idxs]
        false_positives = 1 + threshold_idxs - true_positives
        positives = max(int(np.sum(labels == 1)), 1)
        negatives = max(int(np.sum(labels == 0)), 1)
        tpr = np.r_[0.0, true_positives / positives]
        fpr = np.r_[0.0, false_positives / negatives]
        thresholds = np.r_[np.inf, sorted_scores[threshold_idxs]]
        return fpr.astype(float), tpr.astype(float), thresholds.astype(float)
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
AUVIRE_ROOT = PROJECT_ROOT / "auvire"
if str(AUVIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(AUVIRE_ROOT))

try:
    from src.metrics import AP, AR
except ModuleNotFoundError:
    AP = None
    AR = None

from dataset import AVFeatureDataset
try:
    from helpers.model_helpers import make_boundary_labels
except ModuleNotFoundError:
    from helpers.model_helprers import make_boundary_labels
from helpers.loss_helpers import boundary_localization_loss
from model import AuvireAVDeepfake1MLocalizer, ConvBoundaryTemporalFusionAV_only, SimpleTemporalFusion, SimpleTemporalFusionAV_only, TemporalFusionModel, BoundaryAwareAVLocalizer
from utils import seed_run


MODEL_REGISTRY = {
    "TemporalFusionModel": TemporalFusionModel,
    "SimpleTemporalFusion": SimpleTemporalFusion,
    "SimpleTemporalFusionAV_only": SimpleTemporalFusionAV_only,
    "ConvBoundaryTemporalFusionAV_only": ConvBoundaryTemporalFusionAV_only,
    "BoundaryAwareAVLocalizer": BoundaryAwareAVLocalizer,
    "AuvireAVDeepfake1MLocalizer": AuvireAVDeepfake1MLocalizer,
}

SEGMENT_AP_IOU_THRESHOLDS = [0.5, 0.75, 0.9, 0.95]
SEGMENT_AR_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
SEGMENT_AR_PROPOSAL_COUNTS = [100, 50, 30, 20, 10, 5]
SEGMENT_FPS = 25.0


def load_config(config_path):
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def print_config(config):
    print("\nTraining config")
    print("=" * 15)
    print(yaml.safe_dump(config, sort_keys=False).strip())


def save_checkpoint(state, is_best, save_path, model_name):
    if not is_best:
        return None
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{model_name}.pt")
    torch.save(state, save_file)
    return save_file


def class_std_loss(logits, labels):
    fake_mask = labels > 0.5
    real_mask = labels < 0.5
    loss = torch.tensor(0.0, device=logits.device)
    count = 0

    if fake_mask.sum() > 1:
        loss = loss + logits[fake_mask].std(unbiased=False)
        count += 1
    if real_mask.sum() > 1:
        loss = loss + logits[real_mask].std(unbiased=False)
        count += 1

    return loss / count if count else loss


BOUNDARY_AWARE_KEYS = {"fake_logits", "start_logits", "end_logits"}


def is_boundary_aware_output(model_output):
    return isinstance(model_output, dict) and BOUNDARY_AWARE_KEYS.issubset(model_output.keys())


def extract_frame_logits(model_output):
    if isinstance(model_output, dict):
        if "fake_logits" in model_output:
            return model_output["fake_logits"]
        if "frame_logits" not in model_output:
            raise KeyError("Dict model outputs must include 'fake_logits' or 'frame_logits'.")
        return model_output["frame_logits"]
    return model_output


def crop_time_pair(logits, labels):
    keep = min(logits.shape[1], labels.shape[1])
    return logits[:, :keep], labels[:, :keep]


def crop_boundary_aware_output(model_output, keep):
    return {
        "fake_logits": model_output["fake_logits"][:, :keep],
        "start_logits": model_output["start_logits"][:, :keep],
        "end_logits": model_output["end_logits"][:, :keep],
    }


def boundary_and_offset_targets(labels, radius):
    labels = (labels > 0.5).float()
    batch_size, num_frames = labels.shape
    boundary_targets = torch.zeros((batch_size, num_frames, 2), device=labels.device, dtype=labels.dtype)
    offset_targets = torch.zeros((batch_size, num_frames, 2), device=labels.device, dtype=labels.dtype)
    offset_mask = labels.bool()
    radius = max(0, int(radius))

    for batch_idx in range(batch_size):
        positive = labels[batch_idx].bool()
        start = None
        for frame_idx, is_positive in enumerate(positive.tolist() + [False]):
            if is_positive and start is None:
                start = frame_idx
            elif not is_positive and start is not None:
                end = frame_idx - 1
                boundary_start = max(0, start - radius)
                boundary_end = min(num_frames, start + radius + 1)
                boundary_targets[batch_idx, boundary_start:boundary_end, 0] = 1.0
                boundary_start = max(0, end - radius)
                boundary_end = min(num_frames, end + radius + 1)
                boundary_targets[batch_idx, boundary_start:boundary_end, 1] = 1.0

                segment_idx = torch.arange(start, end + 1, device=labels.device, dtype=labels.dtype)
                offset_targets[batch_idx, start : end + 1, 0] = segment_idx - start
                offset_targets[batch_idx, start : end + 1, 1] = end - segment_idx
                start = None

    return boundary_targets, offset_targets, offset_mask


def masked_smooth_l1_loss(predictions, targets, mask):
    if mask.sum() == 0:
        return predictions.new_tensor(0.0)
    expanded_mask = mask.unsqueeze(-1).expand_as(predictions)
    return torch.nn.functional.smooth_l1_loss(
        predictions[expanded_mask],
        targets[expanded_mask],
        reduction="mean",
    )


def compute_thresholded_metrics(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    if len(labels) == 0:
        return empty_binary_metrics()

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


def empty_binary_metrics():
    return {
        "auc": np.nan,
        "threshold": np.nan,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
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
    intersection = max(0, end - start)
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


def compute_segment_ap_ar_at_iou_thresholds(
    framewise_labels,
    framewise_scores,
    confidence_threshold,
    iou_thresholds,
):
    metrics = {}
    for iou_threshold in iou_thresholds:
        true_positives = 0
        pred_count = 0
        gt_count = 0

        for labels, scores in zip(framewise_labels, framewise_scores):
            labels = np.asarray(labels).astype(int).reshape(-1)
            scores = np.asarray(scores).astype(float).reshape(-1)
            keep = min(len(scores), len(labels))
            labels = labels[:keep]
            scores = scores[:keep]

            pred_segments = binary_segments(scores >= confidence_threshold)
            gt_segments = binary_segments(labels)
            true_positives += count_iou_matches(pred_segments, gt_segments, iou_threshold)
            pred_count += len(pred_segments)
            gt_count += len(gt_segments)

        metrics[iou_threshold] = {
            "ap": true_positives / pred_count if pred_count > 0 else 0.0,
            "ar": true_positives / gt_count if gt_count > 0 else 0.0,
        }
    return metrics


def framewise_labels_to_time_segments(labels, fps=SEGMENT_FPS):
    return [(start / fps, end / fps) for start, end in binary_segments(labels)]


def prepare_proposals(proposals, label_length=None):
    proposals = np.asarray(proposals, dtype=float)
    if proposals.size == 0:
        return np.empty((0, 3), dtype=float)

    proposals = proposals.reshape(-1, 3)
    proposals = proposals[np.isfinite(proposals).all(axis=1)]
    if label_length is not None and proposals.size:
        proposals[:, 1] = np.clip(proposals[:, 1], 0, label_length)
        proposals[:, 2] = np.clip(proposals[:, 2], 0, label_length)
    proposals = proposals[proposals[:, 2] > proposals[:, 1]]

    if proposals.size == 0:
        return np.empty((0, 3), dtype=float)

    order = np.argsort(-proposals[:, 0], kind="mergesort")
    return proposals[order]


def frame_scores_to_proposals(scores, label_length=None, max_proposals=None):
    scores = np.asarray(scores).astype(float).reshape(-1)
    if label_length is not None:
        scores = scores[:label_length]
    if scores.size == 0:
        return np.empty((0, 3), dtype=float)

    scores = np.nan_to_num(
        scores,
        nan=-np.inf,
        posinf=np.finfo(np.float32).max,
        neginf=-np.inf,
    )
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        return np.empty((0, 3), dtype=float)

    if max_proposals is None:
        max_proposals = max(SEGMENT_AR_PROPOSAL_COUNTS)

    proposal_by_span = {}
    for threshold in np.unique(finite_scores)[::-1]:
        for start, end in binary_segments(scores >= threshold):
            confidence = float(np.max(scores[start:end]))
            span = (int(start), int(end))
            proposal_by_span[span] = max(confidence, proposal_by_span.get(span, -np.inf))

        if len(proposal_by_span) >= max_proposals:
            break

    proposals = np.asarray(
        [[confidence, start, end] for (start, end), confidence in proposal_by_span.items()],
        dtype=float,
    )
    return prepare_proposals(proposals, label_length=label_length)[:max_proposals]


def resolve_metrics_device(metric_config, fallback_device):
    device_name = metric_config.get("test_style_metrics_device") or metric_config.get("metrics_device")
    if device_name is not None:
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)
    if fallback_device.type == "cuda" and torch.cuda.is_available():
        return fallback_device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_test_style_segment_metrics(framewise_labels, framewise_scores, metrics_device):
    if AP is None or AR is None:
        return None, None

    ground_truth_segments = [framewise_labels_to_time_segments(labels) for labels in framewise_labels]
    max_proposals = max(SEGMENT_AR_PROPOSAL_COUNTS)
    proposal_tensors = []

    for labels, scores in zip(framewise_labels, framewise_scores):
        proposals = frame_scores_to_proposals(scores, label_length=len(labels), max_proposals=max_proposals)
        if len(proposals) < max_proposals:
            padded = np.zeros((max_proposals, 3), dtype=np.float32)
            padded[: len(proposals)] = proposals.astype(np.float32, copy=False)
            proposals = padded
        proposal_tensors.append(torch.as_tensor(proposals, dtype=torch.float32, device=metrics_device))

    if proposal_tensors:
        proposals_tensor = torch.stack(proposal_tensors, dim=0)
    else:
        proposals_tensor = torch.empty((0, max_proposals, 3), dtype=torch.float32, device=metrics_device)

    ap = AP(iou_thresholds=SEGMENT_AP_IOU_THRESHOLDS, device=str(metrics_device))
    ar = AR(
        n_proposals_list=SEGMENT_AR_PROPOSAL_COUNTS,
        iou_thresholds=SEGMENT_AR_IOU_THRESHOLDS,
        device=str(metrics_device),
    )
    return ap(proposals_tensor, ground_truth_segments), ar(proposals_tensor, ground_truth_segments)


def add_test_style_segment_metrics(metrics, framewise_labels, framewise_scores, metric_config, device):
    if not metric_config.get("test_style_segment_metrics", True):
        return

    metrics_device = resolve_metrics_device(metric_config, device)
    ap_metrics, ar_metrics = compute_test_style_segment_metrics(framewise_labels, framewise_scores, metrics_device)
    if ap_metrics is None or ar_metrics is None:
        metrics["test_segment/enabled"] = 0.0
        return

    metrics["test_segment/enabled"] = 1.0

    for iou_threshold in SEGMENT_AP_IOU_THRESHOLDS:
        metrics[f"test_segment/ap@{iou_threshold:g}"] = float(ap_metrics.get(iou_threshold, 0.0))
    for proposal_count in SEGMENT_AR_PROPOSAL_COUNTS:
        metrics[f"test_segment/ar@{proposal_count}"] = float(ar_metrics.get(proposal_count, 0.0))


def flatten_framewise(framewise_scores, framewise_labels):
    flat_scores = []
    flat_labels = []

    for scores, labels in zip(framewise_scores, framewise_labels):
        scores = np.asarray(scores).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        keep = min(len(scores), len(labels))
        flat_scores.append(scores[:keep])
        flat_labels.append(labels[:keep])

    if not flat_scores:
        return np.array([]), np.array([])
    return np.concatenate(flat_scores), np.concatenate(flat_labels)


def safe_average_precision(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return np.nan
    return average_precision_score(labels, scores)


def tpr_at_fpr_targets(labels, scores, target_fprs):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {f"tpr@fpr_{target:.2f}": np.nan for target in target_fprs}

    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        f"tpr@fpr_{target:.2f}": float(tpr[np.argmin(np.abs(fpr - target))])
        for target in target_fprs
    }


def sanitize_for_wandb(metrics):
    clean = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, float)) and not math.isfinite(float(value)):
            clean[key] = None
        else:
            clean[key] = value
    return clean


def add_metric_group(output, prefix, metrics):
    for key, value in metrics.items():
        output[f"{prefix}/{key}"] = value


def build_model(config, device):
    model_config = config["model"]
    model_name = model_config["name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Options: {sorted(MODEL_REGISTRY)}")

    model_kwargs = deepcopy(model_config.get("params", {}))
    return MODEL_REGISTRY[model_name](**model_kwargs).to(device)


SYNTHESIS_DEFAULTS = {
    "synthesize_at_feature": False,
    "synthesize_prob": 0.5,
    "synthstyle": "random",
    "hard_min_start_deviation": 10,
    "hard_max_start_deviation": 30,
    "max_fake_segment_length": 10,
    "num_max_synth_segments_per_video": 3,
    "synth_modalities": "random",
}


def synthesis_kwargs(data_config):
    return {key: data_config.get(key, value) for key, value in SYNTHESIS_DEFAULTS.items()}


def add_synthesis_args(parser):
    parser.add_argument("--synthesize_at_feature", action=argparse.BooleanOptionalAction, default=None, help="Enable train-time feature synthesis for real training rows.")
    parser.add_argument("--synthesize_prob", type=float, default=None, help="Probability that a real training sample is synthesized on each load.")
    parser.add_argument("--synthstyle", choices=["random", "hard"], default=None, help="Replacement-start style for train-time feature synthesis.")
    parser.add_argument("--hard_min_start_deviation", type=int, default=None, help="Minimum hard replacement start offset in frames.")
    parser.add_argument("--hard_max_start_deviation", type=int, default=None, help="Maximum hard replacement start offset in frames.")
    parser.add_argument("--max_fake_segment_length", type=int, default=None, help="Maximum synthetic fake segment length in frames.")
    parser.add_argument("--num_max_synth_segments_per_video", type=int, default=None, help="Maximum number of synthetic fake segments per sample.")
    parser.add_argument("--synth_modalities", choices=["random", "audio", "video", "both"], default=None, help="Feature modality to synthesize.")


def apply_synthesis_args(config, args):
    data_config = config.setdefault("data", {})
    for key in SYNTHESIS_DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            data_config[key] = value


def add_boundary_loss_args(parser):
    parser.add_argument("--lambda_start", type=float, default=None, help="Boundary-aware start loss weight; defaults to config or 1.0.")
    parser.add_argument("--lambda_end", type=float, default=None, help="Boundary-aware end loss weight; defaults to config or 1.0.")
    parser.add_argument("--lambda_dice", type=float, default=None, help="Boundary-aware fake-logit dice loss weight; defaults to config or 1.0.")
    parser.add_argument("--lambda_smooth", type=float, default=None, help="Boundary-aware temporal smoothness weight; defaults to config or 0.03.")
    parser.add_argument("--boundary_radius", type=int, default=None, help="Radius for soft start/end boundary labels; defaults to config or 2.")
    parser.add_argument("--focal_alpha", type=float, default=None, help="Focal BCE alpha for fake logits; defaults to config or 0.75.")
    parser.add_argument("--focal_gamma", type=float, default=None, help="Focal BCE gamma for fake logits; defaults to config or 2.0.")


def apply_boundary_loss_args(config, args):
    loss_config = config.setdefault("loss", {})
    for key in (
        "lambda_start",
        "lambda_end",
        "lambda_dice",
        "lambda_smooth",
        "boundary_radius",
        "focal_alpha",
        "focal_gamma",
    ):
        value = getattr(args, key, None)
        if value is not None:
            loss_config[key] = value


def build_datasets(config):
    data_config = config["data"]
    metadata_root = data_config["metadata_root_path"]
    train_metadata = os.path.join(metadata_root, data_config["train_metadata_file"])
    val_metadata = os.path.join(metadata_root, data_config["val_metadata_file"])

    train_dataset = AVFeatureDataset(
        train_metadata,
        os.path.join(data_config["data_root_path"], data_config["train_split"]),
        T_train=data_config["T_train"],
        debug=data_config.get("debug", False),
        **synthesis_kwargs(data_config),
    )
    val_dataset = AVFeatureDataset(
        val_metadata,
        os.path.join(data_config["data_val_root_path"], data_config["val_split"]),
        T_train=data_config["T_train"],
        debug=data_config.get("debug", False),
    )
    return train_dataset, val_dataset


def build_loaders(train_dataset, val_dataset, config):
    loader_config = config["loader"]
    common_kwargs = {
        "batch_size": loader_config["batch_size"],
        "num_workers": loader_config["num_workers"],
        "pin_memory": loader_config["pin_memory"],
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    return train_loader, val_loader


def run_epoch(
    dataloader,
    model,
    device,
    loss_config,
    metric_config,
    optimizer=None,
    use_tqdm=False,
    compute_test_style_metrics=False,
    logit_save_dir=None,
    split="val",
    epoch=None,
):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    pos_weight = torch.tensor([loss_config["frame_pos_weight"]], device=device)
    boundary_pos_weight = torch.tensor([loss_config.get("boundary_pos_weight", 1.0)] * 2, device=device)
    frame_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    boundary_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=boundary_pos_weight)
    video_criterion = torch.nn.BCEWithLogitsLoss()
    loader = tqdm(dataloader) if use_tqdm else dataloader

    loss_sums = {
        "total": 0.0,
        "loss_total": 0.0,
        "frame_bce": 0.0,
        "coarse_frame_bce": 0.0,
        "video_bce": 0.0,
        "class_std": 0.0,
        "boundary_bce": 0.0,
        "boundary_offset": 0.0,
        "loss_fake": 0.0,
        "loss_start": 0.0,
        "loss_end": 0.0,
        "loss_dice": 0.0,
        "loss_smooth": 0.0,
    }
    num_batches = 0
    framewise_scores = []
    framewise_labels = []
    video_scores = []
    video_labels = []
    saved_boundary_logits = {"fake_logits": [], "start_logits": [], "end_logits": [], "labels": []}

    with torch.set_grad_enabled(is_training):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device).float()

            model_output = model(features)
            logits = extract_frame_logits(model_output)
            logits, labels_for_loss = crop_time_pair(logits, labels)
            keep = logits.shape[1]
            zero_loss = logits.new_tensor(0.0)
            frame_bce = zero_loss
            coarse_frame_bce = zero_loss
            video_bce = zero_loss
            std_loss = zero_loss
            boundary_bce = zero_loss
            boundary_offset = zero_loss
            loss_component_values = {
                "loss_fake": 0.0,
                "loss_start": 0.0,
                "loss_end": 0.0,
                "loss_dice": 0.0,
                "loss_smooth": 0.0,
            }

            if is_boundary_aware_output(model_output):
                boundary_model_output = crop_boundary_aware_output(model_output, keep)
                start_labels, end_labels = make_boundary_labels(
                    labels_for_loss,
                    radius=loss_config.get("boundary_radius", 2),
                )
                loss, loss_component_values = boundary_localization_loss(
                    boundary_model_output,
                    labels_for_loss,
                    start_labels,
                    end_labels,
                    lambda_start=loss_config.get("lambda_start", 1.0),
                    lambda_end=loss_config.get("lambda_end", 1.0),
                    lambda_dice=loss_config.get("lambda_dice", 1.0),
                    lambda_smooth=loss_config.get("lambda_smooth", 0.03),
                    focal_alpha=loss_config.get("focal_alpha", 0.75),
                    focal_gamma=loss_config.get("focal_gamma", 2.0),
                )
                frame_bce = logits.new_tensor(loss_component_values["loss_fake"])
                if (not is_training) and metric_config.get("save_validation_logits", False):
                    for key in ("fake_logits", "start_logits", "end_logits"):
                        saved_boundary_logits[key].extend(
                            boundary_model_output[key].detach().cpu().numpy().astype(np.float32)
                        )
                    saved_boundary_logits["labels"].extend(
                        labels_for_loss.detach().cpu().numpy().astype(np.float32)
                    )
            else:
                frame_bce = frame_criterion(logits, labels_for_loss)
                std_loss = class_std_loss(logits, labels_for_loss)
                video_logits_for_loss = torch.logsumexp(logits, dim=1)
                batch_video_labels_for_loss = labels_for_loss.max(dim=1)[0]
                video_bce = video_criterion(video_logits_for_loss, batch_video_labels_for_loss)

                if isinstance(model_output, dict):
                    if loss_config.get("coarse_frame_bce_weight", 0.0) > 0 and "coarse_logits" in model_output:
                        coarse_logits = model_output["coarse_logits"][:, :keep]
                        coarse_frame_bce = frame_criterion(coarse_logits, labels_for_loss)

                    boundary_weight = loss_config.get("boundary_bce_weight", 0.0)
                    offset_weight = loss_config.get("boundary_offset_weight", 0.0)
                    needs_boundary_targets = (
                        (boundary_weight > 0 and "boundary_logits" in model_output)
                        or (offset_weight > 0 and "offsets" in model_output)
                    )
                    if needs_boundary_targets:
                        boundary_targets, offset_targets, offset_mask = boundary_and_offset_targets(
                            labels_for_loss,
                            loss_config.get("boundary_target_radius", 2),
                        )
                        if boundary_weight > 0 and "boundary_logits" in model_output:
                            boundary_bce = boundary_criterion(model_output["boundary_logits"][:, :keep], boundary_targets)
                        if offset_weight > 0 and "offsets" in model_output:
                            boundary_offset = masked_smooth_l1_loss(model_output["offsets"][:, :keep], offset_targets, offset_mask)

                loss = (
                    loss_config["frame_bce_weight"] * frame_bce
                    + loss_config.get("coarse_frame_bce_weight", 0.0) * coarse_frame_bce
                    + loss_config["video_bce_weight"] * video_bce
                    + loss_config["class_std_weight"] * std_loss
                    + loss_config.get("boundary_bce_weight", 0.0) * boundary_bce
                    + loss_config.get("boundary_offset_weight", 0.0) * boundary_offset
                )
                loss_component_values["loss_fake"] = frame_bce.item()

            video_logits = torch.logsumexp(logits, dim=1)
            batch_video_labels = labels_for_loss.max(dim=1)[0]

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            num_batches += 1
            loss_sums["total"] += loss.item()
            loss_sums["loss_total"] += loss.item()
            loss_sums["frame_bce"] += frame_bce.item()
            loss_sums["coarse_frame_bce"] += coarse_frame_bce.item()
            loss_sums["video_bce"] += video_bce.item()
            loss_sums["class_std"] += std_loss.item()
            loss_sums["boundary_bce"] += boundary_bce.item()
            loss_sums["boundary_offset"] += boundary_offset.item()
            for loss_name in ("loss_fake", "loss_start", "loss_end", "loss_dice", "loss_smooth"):
                loss_sums[loss_name] += float(loss_component_values.get(loss_name, 0.0))

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            batch_labels = labels_for_loss.detach().cpu().numpy()
            for sample_scores, sample_labels in zip(probs, batch_labels):
                framewise_scores.append(sample_scores)
                framewise_labels.append(sample_labels)

            video_scores.extend(video_logits.detach().cpu().numpy().tolist())
            video_labels.extend(batch_video_labels.detach().cpu().numpy().tolist())

    if num_batches == 0:
        raise RuntimeError("Dataloader produced no batches.")

    flat_frame_scores, flat_frame_labels = flatten_framewise(framewise_scores, framewise_labels)
    frame_metrics = compute_thresholded_metrics(flat_frame_labels, flat_frame_scores)
    video_metrics = compute_thresholded_metrics(video_labels, video_scores)
    frame_metrics["ap"] = safe_average_precision(flat_frame_labels, flat_frame_scores)
    video_metrics["ap"] = safe_average_precision(video_labels, video_scores)
    frame_metrics.update(tpr_at_fpr_targets(flat_frame_labels, flat_frame_scores, metric_config["target_fprs"]))
    video_metrics.update(tpr_at_fpr_targets(video_labels, video_scores, metric_config["target_fprs"]))

    segment_metrics = compute_segment_ap_ar_at_iou_thresholds(
        framewise_labels,
        framewise_scores,
        confidence_threshold=metric_config["frame_confidence_threshold"],
        iou_thresholds=metric_config["iou_thresholds"],
    )

    metrics = {f"loss/{name}": value / num_batches for name, value in loss_sums.items()}
    add_metric_group(metrics, "frame", frame_metrics)
    add_metric_group(metrics, "video", video_metrics)
    for iou_threshold, values in segment_metrics.items():
        metrics[f"segment/ap@{iou_threshold:.2f}"] = values["ap"]
        metrics[f"segment/ar@{iou_threshold:.2f}"] = values["ar"]

    if compute_test_style_metrics:
        add_test_style_segment_metrics(metrics, framewise_labels, framewise_scores, metric_config, device)

    if (
        (not is_training)
        and metric_config.get("save_validation_logits", False)
        and logit_save_dir is not None
        and saved_boundary_logits["fake_logits"]
    ):
        os.makedirs(logit_save_dir, exist_ok=True)
        epoch_suffix = f"_epoch_{epoch:04d}" if epoch is not None else ""
        save_path = os.path.join(logit_save_dir, f"{split}{epoch_suffix}_boundary_logits.npz")
        np.savez_compressed(
            save_path,
            fake_logits=np.asarray(saved_boundary_logits["fake_logits"], dtype=object),
            start_logits=np.asarray(saved_boundary_logits["start_logits"], dtype=object),
            end_logits=np.asarray(saved_boundary_logits["end_logits"], dtype=object),
            labels=np.asarray(saved_boundary_logits["labels"], dtype=object),
        )
        metrics["logits/saved"] = 1.0
    else:
        metrics["logits/saved"] = 0.0

    return metrics


def metric_value(metrics, metric_name, default=0.0):
    value = metrics.get(metric_name, default)
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def prefixed_metrics(split, metrics):
    return {f"{split}/{key}": value for key, value in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Train supervised AVH-Align framewise model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config.")
    add_synthesis_args(parser)
    add_boundary_loss_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    apply_synthesis_args(config, args)
    apply_boundary_loss_args(config, args)
    seed_run(config.get("seed", 42))
    print_config(config)

    device_name = config["training"].get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu")

    train_dataset, val_dataset = build_datasets(config)
    train_loader, val_loader = build_loaders(train_dataset, val_dataset, config)
    model = build_model(config, device)

    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name", "AdamW").lower()
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer '{optimizer_config.get('name')}'. Expected Adam or AdamW.")
    optimizer = optimizer_class(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config["scheduler"]["mode"],
        factor=config["scheduler"]["factor"],
        patience=config["scheduler"]["patience"],
    )

    wandb_config = config["wandb"]
    wandb_run = None
    if wandb_config.get("enabled", True):
        wandb_run = wandb.init(
            project=wandb_config["project"],
            entity=wandb_config.get("entity"),
            name=config["run"]["name"],
            group=wandb_config.get("group"),
            tags=wandb_config.get("tags"),
            mode=wandb_config.get("mode", "online"),
            config=config,
        )
        if wandb_config.get("watch_model", True):
            wandb.watch(
                model,
                log=wandb_config.get("watch_log", "gradients"),
                log_freq=wandb_config.get("watch_log_freq", 100),
            )

    best_metric = -float("inf")
    epochs_without_improvement = 0
    monitor = config["checkpoint"]["monitor"]
    use_tqdm = config["training"].get("use_tqdm", False)
    validation_logits_dir = config["metrics"].get("validation_logits_dir")
    if config["metrics"].get("save_validation_logits", False) and validation_logits_dir is None:
        validation_logits_dir = os.path.join(
            config["checkpoint"]["save_path"],
            f"{config['run']['name']}_validation_logits",
        )

    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        train_metrics = run_epoch(
            train_loader,
            model,
            device,
            config["loss"],
            config["metrics"],
            optimizer=optimizer,
            use_tqdm=use_tqdm,
        )
        val_metrics = run_epoch(
            val_loader,
            model,
            device,
            config["loss"],
            config["metrics"],
            optimizer=None,
            use_tqdm=use_tqdm,
            compute_test_style_metrics=True,
            logit_save_dir=validation_logits_dir,
            split="val",
            epoch=epoch + 1,
        )

        current_metric = metric_value(prefixed_metrics("val", val_metrics), monitor)
        scheduler.step(current_metric)

        print(
            f"Train | loss: {train_metrics['loss/total']:.4f} "
            f"| frame AP: {train_metrics['frame/ap']:.4f} "
            f"| frame AUC: {train_metrics['frame/auc']:.4f} "
            f"| video AP: {train_metrics['video/ap']:.4f} "
            f"| video AUC: {train_metrics['video/auc']:.4f}"
        )
        print(
            f"Val   | loss: {val_metrics['loss/total']:.4f} "
            f"| frame AP: {val_metrics['frame/ap']:.4f} "
            f"| frame AUC: {val_metrics['frame/auc']:.4f} "
            f"| video AP: {val_metrics['video/ap']:.4f} "
            f"| video AUC: {val_metrics['video/auc']:.4f}"
        )

        log_payload = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **prefixed_metrics("train", train_metrics),
            **prefixed_metrics("val", val_metrics),
        }
        if wandb_run is not None:
            wandb.log(sanitize_for_wandb(log_payload))

        is_best = current_metric > best_metric
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_metric": best_metric,
            "monitor": monitor,
            "config": config,
        }
        save_path = save_checkpoint(
            checkpoint,
            is_best,
            config["checkpoint"]["save_path"],
            config["run"]["name"],
        )

        if is_best:
            best_metric = current_metric
            epochs_without_improvement = 0
            print(f"New best model saved to {save_path} ({monitor}: {best_metric:.4f}).")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs.")

        if epochs_without_improvement >= config["training"]["early_stopping_patience"]:
            print("Early stopping triggered.")
            break

    if wandb_run is not None:
        wandb.finish()
    print("Training finished.")


if __name__ == "__main__":
    main()
