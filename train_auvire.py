import argparse
import copy
import datetime
import json
import math
import os
import sys

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
AUVIRE_ROOT = os.path.join(PROJECT_ROOT, "auvire")
if AUVIRE_ROOT not in sys.path:
    sys.path.insert(0, AUVIRE_ROOT)

from src.config import load_default  # noqa: E402


DEFAULT_DATA_CONFIG = {
    "data_root_path": "Features/AV1M",
    "data_val_root_path": "Features/AV1M-Trimmed",
    "metadata_root_path": "av1m_metadata/",
    "train_split": "train",
    "val_split": "test",
    "train_metadata_file": "train_metadata_with_synth_fake_segments.csv",
    "val_metadata_file": "test_metadata_supervised.csv",
    "T_train": 500,
    "debug": False,
    "synthesize_at_feature": False,
    "synthesize_prob": 0.5,
    "synthstyle": "random",
    "hard_min_start_deviation": 10,
    "hard_max_start_deviation": 30,
    "max_fake_segment_length": 10,
    "num_max_synth_segments_per_video": 3,
    "synth_modalities": "random",
}


DEFAULT_CHECKPOINT_CONFIG = {
    "save_path": "ckpt/auvire",
    "monitor": "val/ap",
}


DEFAULT_METRICS_CONFIG = {
    "frame_confidence_threshold": 0.5,
    "iou_thresholds": [0.5, 0.75],
}


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


def get_filename(cfg):
    return "_".join(
        map(
            str,
            [
                cfg["dataset"]["name"],
                "b",
                cfg["dataset"]["backbone"],
                "t",
                cfg["model"]["type"]["reconstruction"],
                cfg["model"]["type"]["encoder"],
                "h",
                cfg["model"]["num_heads"],
                "d",
                cfg["model"]["d_model"],
                "l",
                f"r{cfg['model']['encoder']['nlayers']['retain']}"
                + f"d{cfg['model']['encoder']['nlayers']['downsample']}",
                "w",
                cfg["model"]["win_size"],
                "o",
                cfg["model"]["operation"],
                "rl",
                f"r{cfg['model']['reconstruction']['nlayers']['pre']}"
                + f"d{cfg['model']['reconstruction']['nlayers']['downsample']}"
                + f"u{cfg['model']['reconstruction']['nlayers']['upsample']}"
                + f"s{cfg['model']['reconstruction']['nlayers']['post']}",
                "rm",
                "_".join(cfg["model"]["reconstruction"]["modality"]),
                "f",
                cfg["model"]["encoder"]["fpn"],
                "conv",
                "".join(
                    [
                        "l" if cfg["model"]["conv"]["use_ln"] else "-",
                        "r" if cfg["model"]["conv"]["use_rl"] else "-",
                        "d" if cfg["model"]["conv"]["use_do"] else "-",
                    ]
                ),
                "c",
                "_".join(cfg["criterion"]["composition"]),
            ],
        )
    )


def load_config(config_path):
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def deep_update(base, updates):
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_value(value):
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_dot_override(config, override):
    if "=" not in override:
        raise ValueError(f"Override must use key=value syntax: {override}")

    key, value = override.split("=", 1)
    keys = key.split(".")
    target = config
    for part in keys[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[keys[-1]] = parse_value(value)


def print_config(config):
    print("\nAuViRe training config")
    print("=" * 22)
    print(yaml.safe_dump(config, sort_keys=False).strip())


def infer_dataset_name(args, file_config):
    if args.dataset_name is not None:
        return args.dataset_name
    if "dataset_name" in file_config:
        return file_config["dataset_name"]
    if isinstance(file_config.get("dataset"), dict) and file_config["dataset"].get("name"):
        return file_config["dataset"]["name"]
    return "avdeepfake1m"


def normalize_config(config):
    config.setdefault("data", copy.deepcopy(DEFAULT_DATA_CONFIG))
    config["data"] = deep_update(DEFAULT_DATA_CONFIG, config["data"])
    config.setdefault("checkpoint", copy.deepcopy(DEFAULT_CHECKPOINT_CONFIG))
    config["checkpoint"] = deep_update(DEFAULT_CHECKPOINT_CONFIG, config["checkpoint"])
    config.setdefault("metrics", copy.deepcopy(DEFAULT_METRICS_CONFIG))
    config["metrics"] = deep_update(DEFAULT_METRICS_CONFIG, config["metrics"])

    if "loader" in config:
        loader = config["loader"]
        config.setdefault("dataloader", {})
        if "batch_size" in loader:
            config["dataloader"]["batch_size"] = loader["batch_size"]
        if "num_workers" in loader:
            config["dataloader"]["workers"] = loader["num_workers"]

    config["dataset"].setdefault("params", {})
    config["dataset"]["params"]["max_length"] = config["data"]["T_train"]
    config.setdefault("disable_tqdm", True)
    config.setdefault("logging", True)
    config.setdefault("delete_ckpt", False)
    return config


def build_config(args):
    file_config = load_config(args.config) if args.config is not None else {}
    if "model" in file_config and "type" not in file_config["model"]:
        file_config = copy.deepcopy(file_config)
        file_config.pop("model")
    if "seed" in file_config and "seeds" not in file_config:
        file_config = copy.deepcopy(file_config)
        file_config["seeds"] = [file_config.pop("seed")]
    dataset_name = infer_dataset_name(args, file_config)
    config = deep_update(load_default(dataset_name), file_config)

    config.pop("dataset_name", None)
    config["dataset"]["name"] = dataset_name
    config = normalize_config(config)
    

    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["seeds"] = [args.seed]
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["dataloader"]["batch_size"] = args.batch_size
    if args.workers is not None:
        config["dataloader"]["workers"] = args.workers
    if args.lr is not None:
        config["optimization"]["lr"] = args.lr
    if args.feature_dim is not None:
        config["feature_dim"] = args.feature_dim
    if args.logging is not None:
        config["logging"] = args.logging
    if args.disable_tqdm is not None:
        config["disable_tqdm"] = args.disable_tqdm
    if args.delete_ckpt is not None:
        config["delete_ckpt"] = args.delete_ckpt

    for override in args.set:
        apply_dot_override(config, override)

    apply_synthesis_args(config, args)

    config = normalize_config(config)
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; using CPU.")
        config["device"] = "cpu"
    return config


def build_datasets(config):
    from dataset import AVFeatureDataset

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
    loader_config = config["dataloader"]
    pin_memory = config.get("loader", {}).get("pin_memory", True)
    common_kwargs = {
        "batch_size": loader_config["batch_size"],
        "num_workers": loader_config["workers"],
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    return train_loader, val_loader


def labels_to_tfl_targets(labels):
    labels = (labels > 0.5).float()
    targets = torch.zeros((*labels.shape, 3), device=labels.device, dtype=labels.dtype)
    targets[:, :, 0] = labels

    for batch_idx in range(labels.shape[0]):
        positive = labels[batch_idx].bool()
        start = None
        for frame_idx, is_positive in enumerate(positive.tolist() + [False]):
            if is_positive and start is None:
                start = frame_idx
            elif not is_positive and start is not None:
                end = frame_idx - 1
                segment_idx = torch.arange(start, end + 1, device=labels.device)
                targets[batch_idx, start : end + 1, 1] = segment_idx - start
                targets[batch_idx, start : end + 1, 2] = end - segment_idx
                start = None
    return targets


def split_av_features(features):
    if features.ndim != 4 or features.shape[1] < 2:
        raise ValueError(f"Expected features with shape [batch, 2, time, dim], got {tuple(features.shape)}")
    video_features = features[:, 0]
    audio_features = features[:, 1]
    return video_features, audio_features


def compute_factor(config):
    encoder = config["model"]["encoder"]
    return [1] * encoder["nlayers"]["retain"] + [2 ** (i + 1) for i in range(encoder["nlayers"]["downsample"])]


def build_model(config, input_dim):
    from src.models import Model

    factor = compute_factor(config)
    model_config = config["model"]
    auvire_model = Model(
        max_length=config["dataset"]["params"]["max_length"],
        d_model=model_config["d_model"],
        win_size=model_config["win_size"],
        num_heads=model_config["num_heads"],
        operation=model_config["operation"],
        reconstruction=model_config["reconstruction"],
        encoder=model_config["encoder"],
        dropout=model_config["dropout"],
        use_ln=model_config["conv"]["use_ln"],
        use_rl=model_config["conv"]["use_rl"],
        use_do=model_config["conv"]["use_do"],
        model_type=model_config["type"],
        factor=factor,
        device=config["device"],
    )
    if input_dim == 768:
        return auvire_model, factor
    if input_dim == 1024:
        return AuVireFeatureAdapter(auvire_model, input_dim=input_dim), factor
    raise ValueError(f"AUVIRE supports 768-dim features directly or 1024-dim features with projection, got {input_dim}.")


def build_optimizer(model, config):
    optimizer_name = config["optimization"]["optimizer"]["name"]
    if optimizer_name != "adam":
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported by AuViRe defaults.")
    return torch.optim.Adam(model.parameters(), lr=config["optimization"]["lr"])


def build_scheduler(optimizer, config):
    name = config["optimization"]["scheduler"]["name"]
    if name == "none":
        return None
    if name == "reduceonplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=config["patience"] - 3)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, config["epochs"] // 5)
    if name == "cosineanealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    raise ValueError(f"Scheduler '{name}' is not supported.")


def build_criterion(config, factor):
    from src.losses import CombinedLoss

    criterion_config = config["criterion"]
    return CombinedLoss(
        alpha=criterion_config["params"]["alpha"],
        gamma=criterion_config["params"]["gamma"],
        composition=criterion_config["composition"],
        factor=factor,
    )


def safe_average_precision(labels, scores):
    from sklearn.metrics import average_precision_score

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return np.nan
    return average_precision_score(labels, scores)


def safe_roc_auc(labels, scores):
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, scores)


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


def compute_segment_ap_ar_at_iou_thresholds(framewise_labels, framewise_scores, confidence_threshold, iou_thresholds):
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


def add_metric_group(output, prefix, metrics):
    for key, value in metrics.items():
        output[f"{prefix}/{key}"] = value


def compute_auvire_ap_from_batches(multiscale_scores, multiscale_labels):
    ap_scores = []
    for scores, labels in zip(multiscale_scores, multiscale_labels):
        flat_scores, flat_labels = flatten_framewise(scores, labels)
        ap_scores.append(safe_average_precision(flat_labels, flat_scores))
    finite_scores = [score for score in ap_scores if math.isfinite(float(score))]
    return float(np.mean(finite_scores)) if finite_scores else 0.0


def collect_auvire_metrics(predictions, targets, factor, multiscale_scores, multiscale_labels, framewise_scores, framewise_labels):
    labels = targets.detach().cpu().numpy()
    finest_idx = min(range(len(factor)), key=lambda idx: factor[idx])

    for i, prediction in enumerate(predictions):
        scores = torch.sigmoid(prediction[:, :, 0]).detach().cpu().numpy()
        downsampled_labels = labels[:, :: factor[i], 0]
        keep = min(scores.shape[1], downsampled_labels.shape[1])
        batch_scores = scores[:, :keep]
        batch_labels = downsampled_labels[:, :keep]
        for sample_scores, sample_labels in zip(batch_scores, batch_labels):
            multiscale_scores[i].append(sample_scores)
            multiscale_labels[i].append(sample_labels)
            if i == finest_idx:
                framewise_scores.append(sample_scores)
                framewise_labels.append(sample_labels)


def run_epoch(loader, model, criterion, optimizer, device, factor, metric_config, disable_tqdm):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    total_loss = 0.0
    num_batches = 0
    multiscale_scores = [[] for _ in factor]
    multiscale_labels = [[] for _ in factor]
    framewise_scores = []
    framewise_labels = []
    progress = tqdm(loader, unit="batch", dynamic_ncols=True, disable=disable_tqdm)

    with torch.set_grad_enabled(is_training):
        for features, labels in progress:
            features = features.to(device)
            labels = labels.to(device).float()
            video_features, audio_features = split_av_features(features)
            targets = labels_to_tfl_targets(labels)

            if isinstance(model, AuVireFeatureAdapter):
                predictions, dissimilarity = model(video_features, audio_features)
            else:
                predictions, dissimilarity = model([video_features, audio_features])
            loss = criterion(predictions, targets, dissimilarity)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            collect_auvire_metrics(
                predictions,
                targets,
                factor,
                multiscale_scores,
                multiscale_labels,
                framewise_scores,
                framewise_labels,
            )
            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"loss": total_loss / num_batches})

    if num_batches == 0:
        raise RuntimeError("Dataloader produced no batches.")

    flat_frame_scores, flat_frame_labels = flatten_framewise(framewise_scores, framewise_labels)
    frame_metrics = {
        "ap": safe_average_precision(flat_frame_labels, flat_frame_scores),
        "auc": safe_roc_auc(flat_frame_labels, flat_frame_scores),
    }
    segment_metrics = compute_segment_ap_ar_at_iou_thresholds(
        framewise_labels,
        framewise_scores,
        confidence_threshold=metric_config["frame_confidence_threshold"],
        iou_thresholds=metric_config["iou_thresholds"],
    )

    metrics = {
        "loss": total_loss / num_batches,
        "ap": compute_auvire_ap_from_batches(multiscale_scores, multiscale_labels),
    }
    add_metric_group(metrics, "frame", frame_metrics)
    for iou_threshold, values in segment_metrics.items():
        metrics[f"segment/ap@{iou_threshold:.2f}"] = values["ap"]
        metrics[f"segment/ar@{iou_threshold:.2f}"] = values["ar"]
    return metrics


def metric_value(metrics, name):
    value = metrics.get(name, 0.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def prefixed_metrics(split, metrics):
    return {f"{split}/{key}": value for key, value in metrics.items()}


def save_checkpoint(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def run_training(config):
    device = torch.device(config["device"])
    train_dataset, val_dataset = build_datasets(config)
    train_loader, val_loader = build_loaders(train_dataset, val_dataset, config)

    sample_features, _ = train_dataset[0]
    input_dim = config.get("feature_dim") or sample_features.shape[-1]
    if input_dim != sample_features.shape[-1]:
        print(f"Using feature_dim={input_dim}; first sample has dim {sample_features.shape[-1]}.")
    model, factor = build_model(config, input_dim=input_dim)
    model.to(device)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = build_criterion(config, factor)

    filename = get_filename(config)
    ckpt_path = os.path.join(config["checkpoint"]["save_path"], f"{filename}.pth")
    log_path = os.path.join(config["checkpoint"]["save_path"], f"{filename}.json")
    best_score = -float("inf")
    best_epoch = 0
    history = []

    print(f"[{datetime.datetime.now()}] Training starts")
    print(f"checkpoint: {ckpt_path}")

    for epoch in range(config["epochs"]):
        start = datetime.datetime.now()
        train_metrics = run_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            device,
            factor,
            config["metrics"],
            config["disable_tqdm"],
        )
        val_metrics = run_epoch(
            val_loader,
            model,
            criterion,
            None,
            device,
            factor,
            config["metrics"],
            config["disable_tqdm"],
        )

        current_score = metric_value(prefixed_metrics("val", val_metrics), config["checkpoint"]["monitor"])
        if scheduler is not None:
            if config["optimization"]["scheduler"]["name"] == "reduceonplateau":
                scheduler.step(current_score)
            else:
                scheduler.step()

        duration = str(datetime.datetime.now() - start)
        epoch_metrics = {
            "epoch": epoch + 1,
            "duration": duration,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **prefixed_metrics("train", train_metrics),
            **prefixed_metrics("val", val_metrics),
        }
        history.append(epoch_metrics)

        print(
            f"[{datetime.datetime.now()}: Epoch {epoch + 1}/{config['epochs']} {duration}] "
            f"train_loss={train_metrics['loss']:.4f} train_ap={100 * train_metrics['ap']:.2f} "
            f"train_frame_auc={train_metrics['frame/auc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_ap={100 * val_metrics['ap']:.2f} "
            f"val_frame_auc={val_metrics['frame/auc']:.4f} "
            f"val_ap@0.50={val_metrics.get('segment/ap@0.50', 0.0):.4f} "
            f"val_ar@0.50={val_metrics.get('segment/ar@0.50', 0.0):.4f}"
        )

        if config.get("logging", True):
            os.makedirs(config["checkpoint"]["save_path"], exist_ok=True)
            with open(log_path, "w") as handle:
                json.dump({"config": config, "results": history}, handle, indent=2)

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            save_checkpoint(
                ckpt_path,
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_score": best_score,
                    "factor": factor,
                    "config": config,
                },
            )
            print(f"[{datetime.datetime.now()}] Saved checkpoint at {ckpt_path}")
        elif epoch - best_epoch >= config["patience"]:
            print(f"[{datetime.datetime.now()}] Early stopped training at epoch {epoch + 1}")
            break

    if config.get("delete_ckpt", False) and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"[{datetime.datetime.now()}] Training ends")
    return {"checkpoint": ckpt_path, "log": log_path if config.get("logging", True) else None, "best_score": best_score}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


SYNTHESIS_ARG_KEYS = [
    "synthesize_at_feature",
    "synthesize_prob",
    "synthstyle",
    "hard_min_start_deviation",
    "hard_max_start_deviation",
    "max_fake_segment_length",
    "num_max_synth_segments_per_video",
    "synth_modalities",
]


def synthesis_kwargs(data_config):
    return {key: data_config[key] for key in SYNTHESIS_ARG_KEYS}


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
    for key in SYNTHESIS_ARG_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            config["data"][key] = value


def main():
    parser = argparse.ArgumentParser(description="Train AuViRe architecture on AVH-Align feature data.")
    parser.add_argument(
        "-d",
        "--dataset_name",
        choices=["lavdf", "avdeepfake1m"],
        default=None,
        help="AUVIRE default experiment/model settings to start from. Defaults to avdeepfake1m.",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config/override file.")
    parser.add_argument("--device", type=str, default=None, help="Override config device.")
    parser.add_argument("--seed", type=int, default=None, help="Run a single seed instead of config seeds.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override dataloader batch size.")
    parser.add_argument("--workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--lr", type=float, default=None, help="Override optimization learning rate.")
    parser.add_argument("--feature_dim", type=int, choices=[768, 1024], default=None, help="Feature dimension: 768 uses original AUVIRE, 1024 adds projection layers.")
    parser.add_argument("--logging", type=str_to_bool, default=None, help="Enable/disable JSON logging.")
    parser.add_argument("--disable_tqdm", type=str_to_bool, default=None, help="Enable/disable tqdm output.")
    parser.add_argument("--delete_ckpt", type=str_to_bool, default=None, help="Delete best checkpoint after training.")
    add_synthesis_args(parser)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override nested config values, e.g. --set data.T_train=512. May be used multiple times.",
    )
    parser.add_argument("--print_config", action="store_true", help="Print the effective config before training.")
    args = parser.parse_args()

    config = build_config(args)
    if args.print_config:
        print_config(config)

    from utils import seed_run

    for seed in config["seeds"]:
        seed_run(seed)
        config_for_seed = copy.deepcopy(config)
        config_for_seed["seeds"] = [seed]
        result = run_training(config_for_seed)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
