import argparse
import json
import sys
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import pandas as pd
import os
import cv2
import torch.nn.functional as F
from pathlib import Path
import re

from model import (
    ConvBoundaryTemporalFusionAV_only,
    FusionModel,
    SimpleTemporalFusion,
    SimpleTemporalFusionAV_only,
    TemporalFusionModel,
)

from utils import seed_run
from sklearn.metrics import roc_curve

AUVIRE_ROOT = Path(__file__).resolve().parent / "auvire"
if str(AUVIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(AUVIRE_ROOT))

from src.metrics import AP, AR  # noqa: E402

Out_features_path = "Out_Features_trimmed_with_visual_map"
if not os.path.exists(Out_features_path):
    os.makedirs(Out_features_path)
    
Validation_meta_path = "data/DeepfakeDatasets/AV_Deepfake1M/AV-Deepfake1M-PlusPlus/val_metadata.json"

SEGMENT_AP_IOU_THRESHOLDS = [0.5, 0.75, 0.9, 0.95]
SEGMENT_AR_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
SEGMENT_AR_PROPOSAL_COUNTS = [100, 50, 30, 20, 10, 5]
SEGMENT_FPS = 25.0

MODEL_REGISTRY = {
    "FusionModel": FusionModel,
    "TemporalFusionModel": TemporalFusionModel,
    "SimpleTemporalFusion": SimpleTemporalFusion,
    "SimpleTemporalFusionAV_only": SimpleTemporalFusionAV_only,
    "ConvBoundaryTemporalFusionAV_only": ConvBoundaryTemporalFusionAV_only,
}


def build_model(model_name, checkpoint, device):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")

    model_kwargs = {}
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    checkpoint_model_config = checkpoint_config.get("model", {}) if isinstance(checkpoint_config, dict) else {}
    if checkpoint_model_config.get("name") == model_name:
        model_kwargs = checkpoint_model_config.get("params", {}) or {}

    try:
        return MODEL_REGISTRY[model_name](**model_kwargs).to(device)
    except TypeError as exc:
        raise TypeError(
            f"Could not instantiate {model_name} with checkpoint model params {model_kwargs}. "
            "Pass matching eval args or check the saved config."
        ) from exc


def get_framewise_labels(path, output_frames, video_root="data/DeepfakeDatasets/AV_Deepfake1M/AV-Deepfake1M-PlusPlus/val/val/vox_celeb_2"):

    with open(Validation_meta_path, "r") as f:
        metadata = json.load(f)

    entry = next(item for item in metadata if item["file"].endswith(path))

    video_frames_meta = entry["video_frames"]
    fake_segments = entry.get("fake_segments", [])

    full_video_path = f"{video_root}/{entry['file']}"
    cap = cv2.VideoCapture(full_video_path)

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps > 1e-3:
        duration = frame_count / fps
    else:
        # fallback: estimate duration using last fake segment
        if len(fake_segments) > 0:
            duration = max(seg[1] for seg in fake_segments)
        else:
            # final fallback assumption (common datasets use 25 fps)
            duration = video_frames_meta / 25.0

    cap.release()

    # compute fps from metadata frame count
    fps = video_frames_meta / duration

    # build labels
    labels = np.zeros(video_frames_meta, dtype=np.int32)

    for (start_t, end_t) in fake_segments:
        start_frame = int(np.floor(start_t * fps))
        end_frame = int(np.ceil(end_t * fps))

        start_frame = max(0, start_frame)
        end_frame = min(video_frames_meta, end_frame)

        labels[start_frame:end_frame] = 1

    # trim earliest frames
    if output_frames < video_frames_meta:
        trim = video_frames_meta - output_frames
        labels = labels[trim:]

    return labels

def process_visual_map_zero_shot(data, fusion_model, device):
    multimodal = torch.tensor(data["multimodal"]).float()
    visual_map = torch.tensor(data["visual_map"]).float()
    multimodal_norm = F.normalize(multimodal, dim=-1)
    # visual_norm = F.normalize(visual, dim=-1)
    visual_map_norm = F.normalize(visual_map, dim=-1)

    visual_diag = torch.diagonal(visual_map_norm, dim1=0, dim2=1).transpose(0,1)
    output = torch.sum(visual_diag * multimodal_norm, dim=-1)

   
   
    # score = torch.logsumexp(-output, dim=0).detach().cpu().squeeze()
    score=output.min().item()
    # output = fusion_model(visual_tensor, audio_tensor)
    # score = torch.logsumexp(-output, dim=0).detach().cpu().squeeze()

    return score,output

def extract_frame_logits(model_output):
    if isinstance(model_output, dict):
        if "frame_logits" not in model_output:
            raise KeyError("Dict model outputs must include 'frame_logits'.")
        return model_output["frame_logits"]
    return model_output


def process_video(data, fusion_model, device):
    visual_tensor = torch.from_numpy(data["visual"]).to(device)
    audio_tensor = torch.from_numpy(data["audio"]).to(device) # change here according to the  model
    # breakpoint()
    # L2 norm
    visual_tensor = visual_tensor / (torch.linalg.norm(visual_tensor, ord=2, dim=-1, keepdim=True)) #T,D
    audio_tensor = audio_tensor / (torch.linalg.norm(audio_tensor, ord=2, dim=-1, keepdim=True)) #T,D
    # print(visual_tensor.shape, audio_tensor.shape)
    
    if args.model_name == "FusionModel":
        output = fusion_model(visual_tensor, audio_tensor)
        frame_scores = -output.squeeze()
        score = torch.logsumexp(frame_scores, dim=0).detach().cpu().squeeze()
    else:
        stacked_feat = torch.stack((visual_tensor, audio_tensor), dim=1)  # T, 2, D
        stacked_feat = stacked_feat.transpose(0, 1).unsqueeze(0)  # 1, 2, T, D
        model_output = fusion_model(stacked_feat)
        frame_scores = extract_frame_logits(model_output)
        if frame_scores.ndim == 2 and frame_scores.shape[0] == 1:
            frame_scores = frame_scores[0]
        else:
            frame_scores = frame_scores.squeeze()
        frame_scores = frame_scores.reshape(-1)
        # score = frame_scores.max().detach().cpu().item()
        score = torch.logsumexp(frame_scores, dim=0).detach().cpu().squeeze()

    return score, frame_scores.reshape(-1)

def Zero_shot_process_video(data, fusion_model, device):
    visual_tensor = torch.from_numpy(data["visual"]).to(device)
    audio_tensor = torch.from_numpy(data["audio"]).to(device)

    # L2 norm
    visual_tensor = visual_tensor / (torch.linalg.norm(visual_tensor, ord=2, dim=-1, keepdim=True))
    audio_tensor = audio_tensor / (torch.linalg.norm(audio_tensor, ord=2, dim=-1, keepdim=True))

    # score = torch.nn.functional.cosine_similarity(visual_tensor, audio_tensor, dim=-1).detach().cpu().squeeze().mean()
    output = (visual_tensor * audio_tensor).sum(dim=-1)
    score = torch.logsumexp(-output, dim=0).detach().cpu().squeeze()
    # output = fusion_model(visual_tensor, audio_tensor)
    # score = torch.logsumexp(-output, dim=0).detach().cpu().squeeze()

    return score,output


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


def framewise_labels_to_time_segments(labels, fps=SEGMENT_FPS):
    return [(start / fps, end / fps) for start, end in binary_segments(labels)]


def prepare_proposals(proposals, label_length=None):
    proposals = np.asarray(proposals, dtype=float)
    if proposals.size == 0:
        return np.empty((0, 3), dtype=float)
    proposals = proposals.reshape(-1, 3)
    proposals = proposals[np.isfinite(proposals).all(axis=1)]
    if label_length is not None:
        proposals[:, 1] = np.clip(proposals[:, 1], 0, label_length)
        proposals[:, 2] = np.clip(proposals[:, 2], 0, label_length)
    proposals = proposals[proposals[:, 2] > proposals[:, 1]]
    order = np.argsort(-proposals[:, 0], kind="mergesort")
    return proposals[order]


def proposal_span_iou(proposal_a, proposal_b):
    start = max(float(proposal_a[1]), float(proposal_b[1]))
    end = min(float(proposal_a[2]), float(proposal_b[2]))
    intersection = max(0.0, end - start)
    union = max(float(proposal_a[2]), float(proposal_b[2])) - min(float(proposal_a[1]), float(proposal_b[1]))
    return intersection / union if union > 0 else 0.0


def span_confidence(scores, start, end, top_fraction=0.25):
    span_scores = np.asarray(scores[start:end], dtype=float)
    if span_scores.size == 0:
        return -np.inf
    top_count = max(1, int(np.ceil(span_scores.size * top_fraction)))
    top_scores = np.partition(span_scores, -top_count)[-top_count:]
    return float(np.mean(top_scores))


def temporal_soft_nms(proposals, max_proposals, sigma=0.5, iou_threshold=0.6):
    proposals = np.asarray(proposals, dtype=float)
    if proposals.size == 0:
        return np.empty((0, 3), dtype=float)

    remaining = proposals.copy()
    selected = []

    while len(remaining) > 0 and len(selected) < max_proposals:
        order = np.lexsort((remaining[:, 1], -(remaining[:, 2] - remaining[:, 1]), -remaining[:, 0]))
        best = remaining[order[0]].copy()
        selected.append(best)

        remaining = np.delete(remaining, order[0], axis=0)
        if len(remaining) == 0:
            break

        for idx in range(len(remaining)):
            iou = proposal_span_iou(best, remaining[idx])
            if iou > iou_threshold:
                remaining[idx, 0] *= np.exp(-(iou * iou) / sigma)

        remaining = remaining[np.isfinite(remaining).all(axis=1)]

    return np.asarray(selected, dtype=float)


def frame_scores_to_proposals(scores, label_length=None, max_proposals=None, threshold_count=128):
    scores = np.asarray(scores).astype(float).reshape(-1)
    if label_length is not None:
        scores = scores[:label_length]
    if scores.size == 0:
        return np.empty((0, 3), dtype=float)

    finite_mask = np.isfinite(scores)
    finite_scores = scores[finite_mask]
    if finite_scores.size == 0:
        return np.empty((0, 3), dtype=float)

    min_score = float(np.min(finite_scores))
    max_score = float(np.max(finite_scores))
    scores = np.nan_to_num(
        scores,
        nan=min_score,
        posinf=max_score,
        neginf=min_score,
    )
    finite_scores = scores[np.isfinite(scores)]

    if max_proposals is None:
        max_proposals = max(SEGMENT_AR_PROPOSAL_COUNTS)

    proposal_by_span = {}
    unique_scores = np.unique(finite_scores)
    if unique_scores.size <= threshold_count:
        thresholds = unique_scores[::-1]
    else:
        quantiles = np.linspace(0.0, 1.0, threshold_count)
        thresholds = np.unique(np.quantile(finite_scores, quantiles))[::-1]

    for threshold in thresholds:
        for start, end in binary_segments(scores >= threshold):
            confidence = span_confidence(scores, start, end)
            span = (int(start), int(end))
            proposal_by_span[span] = max(confidence, proposal_by_span.get(span, -np.inf))

    proposals = np.asarray(
        [[confidence, start, end] for (start, end), confidence in proposal_by_span.items()],
        dtype=float,
    )
    proposals = prepare_proposals(proposals, label_length=label_length)
    if proposals.size == 0:
        return proposals

    length = proposals[:, 2] - proposals[:, 1]
    order = np.lexsort((proposals[:, 1], -length, -proposals[:, 0]))
    proposals = proposals[order]
    proposals = temporal_soft_nms(proposals, max_proposals=max_proposals)
    return prepare_proposals(proposals, label_length=label_length)[:max_proposals]


def compute_test_style_segment_metrics(framewise_labels, proposal_segments, metrics_device):
    ground_truth_segments = [framewise_labels_to_time_segments(labels) for labels in framewise_labels]
    max_proposals = max(SEGMENT_AR_PROPOSAL_COUNTS)
    proposal_tensors = []

    for labels, proposals in zip(framewise_labels, proposal_segments):
        proposals = prepare_proposals(proposals, label_length=len(labels))
        proposals = proposals[:max_proposals]
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
    for iou_threshold in SEGMENT_AP_IOU_THRESHOLDS:
        print(f"AP@{iou_threshold:g}: {ap_metrics.get(iou_threshold, 0.0):.4f}")
    for proposal_count in SEGMENT_AR_PROPOSAL_COUNTS:
        print(f"AR@{proposal_count}: {ar_metrics.get(proposal_count, 0.0):.4f}")


def flatten_framewise(framewise_scores, framewise_labels):
    flat_scores = []
    flat_labels = []

    for scores, labels in zip(framewise_scores, framewise_labels):
        scores = np.asarray(scores).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        if not len(scores)==len(labels):
            print(f"Warning: Inconsistent lengths in framewise data: {len(scores)} vs {len(labels)}")
        keep = min(len(scores), len(labels))
        flat_scores.append(scores[:keep])
        flat_labels.append(labels[:keep])

    return np.concatenate(flat_scores), np.concatenate(flat_labels)


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
    proposal_segments=None,
    max_plot_proposals=20,
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
            top_proposals = np.empty((0, 3), dtype=float)
            if proposal_segments is not None and idx < len(proposal_segments):
                top_proposals = prepare_proposals(proposal_segments[idx], label_length=keep)[:max_plot_proposals]

            for proposal_rank, proposal in enumerate(top_proposals):
                start = max(0.0, min(float(proposal[1]), float(keep)))
                end = max(0.0, min(float(proposal[2]), float(keep)))
                if end <= start:
                    continue
                alpha = max(0.08, 0.32 * (1.0 - proposal_rank / max(1, max_plot_proposals)))
                label = "Top proposals" if proposal_rank == 0 else None
                axes[1].axvspan(start, end, color="#ff7f0e", alpha=alpha, linewidth=0, label=label)

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


def main(args):
    seed_run()

    print(f"Evaluating AVH-Align on {args.dataset} with pretrained weights saved at {args.checkpoint_path} ...")

    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_model_weights = torch.load(args.checkpoint_path, weights_only=False)

    fusion_model = build_model(args.model_name, fusion_model_weights, device)
    fusion_model.load_state_dict(fusion_model_weights["state_dict"])
    fusion_model.eval()
    
    # Load metadata for access to labels
    metadata = pd.read_csv(args.metadata)

    outputs = []
    ground_truths = []
    framewise_scores = []
    path_names = []
    framewis_labels=[]
    count = 0
    for _, row in tqdm(metadata.iterrows()):
        count += 1
        # if count >100:
        #     break
        if args.dataset == "AV1M" :
            data = np.load(os.path.join(args.features_path, row["path"].replace(".mp4", ".npz")), allow_pickle=True)
            data_labels= np.load(os.path.join(args.features_path, row["path"].replace(".mp4", "_labels.npz")), allow_pickle=True)
            label = row["label"]
        elif args.dataset == "LavDF" :
            data = np.load(os.path.join(args.features_path, row["path"].replace(".mp4", ".npz")), allow_pickle=True)
            data_labels= np.load(os.path.join(args.features_path, row["path"].replace(".mp4", "_labels.npz")), allow_pickle=True)
            label = row["label"]    
        elif args.dataset == "FakeAVCeleb":
            path=row["path"].replace("FakeAVCeleb/", "")
            data = np.load(os.path.join(args.features_path, path,row["filename"].replace(".mp4", ".npz")), allow_pickle=True)
            label = 0 if row["type"] == "RealVideo-RealAudio" else 1
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        # score,output = process_visual_map_zero_shot(data, fusion_model, device)
        # score,output = process_video(data, fusion_model, device)
        # score,output= Zero_shot_process_video(data, fusion_model, device)
        score,output = process_video(data, fusion_model, device)
        outputs.append(to_float(score))
        framewise_scores.append(output.detach().cpu().numpy())
        ground_truths.append(label)
        path_names.append(row["path"])
        
        if args.dataset == "FakeAVCeleb":
            if row["type"] == "RealVideo-RealAudio":
                framewis_labels.append(np.zeros(output.shape[0], dtype=int))
            else:
                framewis_labels.append(np.ones(output.shape[0], dtype=int))
        else:
            # framewis_labels.append(get_framewise_labels(row["path"], output.shape[0])) # This is wrong, either because of the metadata or something else. 
            framewis_labels.append(data_labels["framewise_labels"])
        # breakpoint()
        
    outputs = np.array(outputs)
    if args.frame_score_type == "sigmoid":
        # Sigmoid scores use a 0.5 confidence threshold. Raw logits use 0.0.
        framewise_scores = [torch.sigmoid(torch.from_numpy(scores)).numpy() for scores in framewise_scores]
        framewise_confidence_threshold = 0.9
    else:
        framewise_confidence_threshold = 3.0
    ground_truths = np.array(ground_truths)
    metrics_device = torch.device(args.metrics_device)
    video_metrics = compute_thresholded_metrics(ground_truths, outputs)
    video_threshold = video_metrics["threshold"]
    if np.isfinite(video_threshold):
        predicted_fake_videos = outputs >= video_threshold
    else:
        predicted_fake_videos = np.zeros_like(ground_truths, dtype=bool)

    proposal_segments = [
        frame_scores_to_proposals(
            scores,
            label_length=len(labels),
            max_proposals=args.max_proposed_segments,
        )
        if predicted_fake else np.empty((0, 3), dtype=float)
        for scores, labels, predicted_fake in zip(framewise_scores, framewis_labels, predicted_fake_videos)
    ]

    if args.plot_framewise:
        plot_framewise_predictions(
            framewise_scores,
            framewis_labels,
            path_names,
            args.dataset,
            args.checkpoint_path,
            args.number_of_plots,
            ranking_threshold=framewise_confidence_threshold,
            proposal_segments=proposal_segments,
            max_plot_proposals=20,
        )

    flat_framewise_scores, flat_framewise_labels = flatten_framewise(framewise_scores, framewis_labels)

    video_ap = average_precision_score(ground_truths, outputs)
    framewise_ap = average_precision_score(flat_framewise_labels, flat_framewise_scores)
    framewise_metrics = compute_thresholded_metrics(flat_framewise_labels, flat_framewise_scores)
    segment_ap, segment_ar = compute_test_style_segment_metrics(
        framewis_labels,
        proposal_segments,
        metrics_device=metrics_device,
    )

    print(f"Video-wise AP: {video_ap:.4f}")
    print(f"Framewise AP: {framewise_ap:.4f}")
    print_metric_block("VIDEO-WISE METRICS", video_metrics)
    print_metric_block("FRAMEWISE METRICS", framewise_metrics)
    print(f"Framewise confidence threshold: {framewise_confidence_threshold:.1f} ({args.frame_score_type})")
    print_ap_ar_block("TIME SEGMENT LOCALIZATION METRICS", segment_ap, segment_ar)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truths, outputs)

    # Target FPRs
    target_fprs = [0.05, 0.10]

    for target in target_fprs:
        # Find closest FPR index
        idx = np.argmin(np.abs(fpr - target))
        print(f"TPR @ {target*100:.1f}% FPR: {tpr[idx]:.4f}")
    
    # save outputs and ground truths for future analysis
    np.save(os.path.join(Out_features_path, f"{args.test_name}_outputs.npy"), outputs)
    np.save(os.path.join(Out_features_path, f"{args.test_name}_ground_truths.npy"), ground_truths)
    framewise_scores_np = np.array(framewise_scores, dtype=object)
    framewis_labels_np = np.array(framewis_labels, dtype=object)
    # Save with pickle enabled
    np.save(
        os.path.join(Out_features_path, f"{args.test_name}_framewise_scores.npy"),
        framewise_scores_np,
        allow_pickle=True
    )
    np.save(
        os.path.join(Out_features_path, f"{args.test_name}_framewise_labels.npy"),
        framewis_labels_np,
        allow_pickle=True
    )
    np.save(
        os.path.join(Out_features_path, f"{args.test_name}_proposals.npy"),
        np.array(proposal_segments, dtype=object),
        allow_pickle=True
    )
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fusion Model on Deepfake Dataset")

    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/AVH-Align_AV1M.pt",
                        help="Path to the pretrained fusion model checkpoint.")
    parser.add_argument("--features_path", type=str,
                        default=f"av1m_features/val/",
                        help="Path to the root folder of test data.")
    parser.add_argument("--metadata", type=str,
                        default="av1m_metadata/test_metadata.csv",
                        help="CSV file containing ground truth labels.")
    parser.add_argument("--dataset", type=str, default="AV1M",
                        help="Dataset name")
    parser.add_argument("--model_name", type=str, default="FusionModel",
                        help="Model architecture to use (FusionModel, TemporalFusionModel, SimpleTemporalFusion, SimpleTemporalFusionAV_only, ConvBoundaryTemporalFusionAV_only)")
    parser.add_argument("--test_name", type=str, default="debug",
                        help="Test name")
    parser.add_argument("--frame_score_type", choices=("sigmoid", "logit"), default="sigmoid",
                        help="Use sigmoid scores with threshold 0.5, or raw logits with threshold 0.0 for framewise segment AP/AR.")
    parser.add_argument("--plot_framewise", action="store_true",
                        help="Save framewise ground-truth and prediction plots.")
    parser.add_argument("--number_of_plots", type=int, default=10,
                        help="Number of framewise prediction plots to save when --plot_framewise is set.")
    parser.add_argument("--max_proposed_segments", type=int, default=5,
                        help="Maximum number of NMS-filtered temporal proposals to keep per video.")
    parser.add_argument("--metrics_device", type=str, default="cpu",
                        help="Device for AUVIRE AP/AR metric tensors.")
    

    args = parser.parse_args()
    main(args)
