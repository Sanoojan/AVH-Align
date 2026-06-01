import argparse
import math
import os
from copy import deepcopy

import numpy as np
import torch
import wandb
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AVFeatureDataset
from model import SimpleTemporalFusion, SimpleTemporalFusionAV_only, TemporalFusionModel
from utils import seed_run


MODEL_REGISTRY = {
    "TemporalFusionModel": TemporalFusionModel,
    "SimpleTemporalFusion": SimpleTemporalFusion,
    "SimpleTemporalFusionAV_only": SimpleTemporalFusionAV_only,
}


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


def run_epoch(dataloader, model, device, loss_config, metric_config, optimizer=None, use_tqdm=False):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    pos_weight = torch.tensor([loss_config["frame_pos_weight"]], device=device)
    frame_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    video_criterion = torch.nn.BCEWithLogitsLoss()
    loader = tqdm(dataloader) if use_tqdm else dataloader

    loss_sums = {"total": 0.0, "frame_bce": 0.0, "video_bce": 0.0, "class_std": 0.0}
    num_batches = 0
    framewise_scores = []
    framewise_labels = []
    video_scores = []
    video_labels = []

    with torch.set_grad_enabled(is_training):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device).float()

            logits = model(features)
            frame_bce = frame_criterion(logits, labels)
            std_loss = class_std_loss(logits, labels)
            video_logits = torch.logsumexp(logits, dim=1)
            batch_video_labels = labels.max(dim=1)[0]
            video_bce = video_criterion(video_logits, batch_video_labels)

            loss = (
                loss_config["frame_bce_weight"] * frame_bce
                + loss_config["video_bce_weight"] * video_bce
                + loss_config["class_std_weight"] * std_loss
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            num_batches += 1
            loss_sums["total"] += loss.item()
            loss_sums["frame_bce"] += frame_bce.item()
            loss_sums["video_bce"] += video_bce.item()
            loss_sums["class_std"] += std_loss.item()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            batch_labels = labels.detach().cpu().numpy()
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
    args = parser.parse_args()

    config = load_config(args.config)
    apply_synthesis_args(config, args)
    seed_run(config.get("seed", 42))
    print_config(config)

    device_name = config["training"].get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu")

    train_dataset, val_dataset = build_datasets(config)
    train_loader, val_loader = build_loaders(train_dataset, val_dataset, config)
    model = build_model(config, device)

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
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
