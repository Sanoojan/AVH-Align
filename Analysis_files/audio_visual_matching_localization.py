import argparse
import csv
import re
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from scipy.optimize import linear_sum_assignment
except ModuleNotFoundError:
    linear_sum_assignment = None

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from sklearn.metrics import (
        accuracy_score as sklearn_accuracy_score,
        average_precision_score as sklearn_average_precision_score,
        f1_score as sklearn_f1_score,
        precision_score as sklearn_precision_score,
        recall_score as sklearn_recall_score,
        roc_auc_score as sklearn_roc_auc_score,
        roc_curve as sklearn_roc_curve,
    )
except ModuleNotFoundError:
    sklearn_accuracy_score = None
    sklearn_average_precision_score = None
    sklearn_f1_score = None
    sklearn_precision_score = None
    sklearn_recall_score = None
    sklearn_roc_auc_score = None
    sklearn_roc_curve = None

REPO_ROOT = Path(__file__).resolve().parents[1]
AUVIRE_ROOT = REPO_ROOT / "auvire"
if str(AUVIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(AUVIRE_ROOT))

try:
    from src.metrics import AP, AR
except ModuleNotFoundError:
    AP = None
    AR = None

DEFAULT_METADATA = REPO_ROOT / "av1m_metadata" / "test_metadata_cleaned.csv"
DEFAULT_FEATURES_ROOT = REPO_ROOT / "Features" / "AV1M-Trimmed" / "test"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Analysis_files" / "audio_visual_matching_localization_outputs"
SEGMENT_AP_IOU_THRESHOLDS = [0.5, 0.75, 0.9, 0.95]
SEGMENT_AR_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
SEGMENT_AR_PROPOSAL_COUNTS = [100, 50, 30, 20, 10, 5]
SEGMENT_FPS = 25.0

ASSUMPTIONS = [
    "Zero-shot: no learned model, classifier, checkpoint, or calibration is used.",
    "Fake regions are harder to recover when matching audio frames to shuffled visual frames.",
    "Audio and visual arrays in each feature .npz are already frame-aligned.",
    "Feature-frame labels in *_labels.npz use the same temporal resolution as audio/visual features.",
    f"Temporal AP/AR proposals interpret feature frames at {SEGMENT_FPS:g} fps, matching eval_new_time.py.",
    "Multiple deshuffling rounds estimate a stable fake score by averaging independent visual shuffles.",
    "Video-level scoring emphasizes suspicious spans by averaging the highest-scoring frame fraction.",
]


def print_section(title):
    print(f"\n{title}")
    print("=" * len(title))


def video_type(path):
    return Path(path).stem


def is_real_row(row):
    return int(row["label"]) == 0 or video_type(row["path"]) == "real"


def feature_path_for_row(features_root, row):
    return Path(features_root) / row["path"].replace(".mp4", ".npz")


def label_path_for_feature_path(feature_path):
    return feature_path.with_name(f"{feature_path.stem}_labels.npz")


def read_metadata(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_rows(rows, max_videos):
    if max_videos <= 0:
        return list(rows)
    return list(rows[:max_videos])


def load_audio_visual(features_root, row, max_frames):
    path = feature_path_for_row(features_root, row)
    data = np.load(path, allow_pickle=True)
    if "audio" not in data.files or "visual" not in data.files:
        raise KeyError(f"{path} must contain 'audio' and 'visual' arrays; found {data.files}")

    audio = np.asarray(data["audio"], dtype=np.float32)
    visual = np.asarray(data["visual"], dtype=np.float32)
    t = min(len(audio), len(visual))
    if max_frames > 0:
        t = min(t, max_frames)
    if t <= 0:
        raise ValueError(f"{path} has no aligned audio/visual frames")
    return audio[:t], visual[:t], path


def load_framewise_labels_for_path(feature_path, length):
    label_path = label_path_for_feature_path(feature_path)
    if not label_path.exists():
        return np.zeros(length, dtype=np.int32)

    data = np.load(label_path, allow_pickle=True)
    if "framewise_labels" not in data.files:
        return np.zeros(length, dtype=np.int32)

    labels = np.asarray(data["framewise_labels"], dtype=np.int32).reshape(-1)
    if len(labels) < length:
        padded = np.zeros(length, dtype=np.int32)
        padded[: len(labels)] = labels
        return padded
    return labels[:length]


def binary_segments(values):
    values = np.asarray(values, dtype=np.int32).reshape(-1)
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


def segment_list_to_string(segments):
    return ";".join(f"{start}-{end}" for start, end in segments)


def l2_normalize(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-8)


def cosine_similarity_matrix(audio, visual):
    return l2_normalize(audio) @ l2_normalize(visual).T


def hungarian_match(similarity):
    if linear_sum_assignment is None:
        return greedy_match(similarity)

    row_ind, col_ind = linear_sum_assignment(-similarity)
    order = np.empty(similarity.shape[0], dtype=np.int64)
    order[row_ind] = col_ind
    return order


def greedy_match(similarity):
    warnings.warn(
        "scipy is not installed; using greedy one-to-one matching instead of Hungarian matching.",
        RuntimeWarning,
        stacklevel=2,
    )
    t = similarity.shape[0]
    order = np.empty(t, dtype=np.int64)
    used_cols = np.zeros(similarity.shape[1], dtype=bool)
    row_best = np.max(similarity, axis=1)
    rows = np.argsort(-row_best, kind="mergesort")

    for row in rows:
        ranked_cols = np.argsort(-similarity[row], kind="mergesort")
        for col in ranked_cols:
            if not used_cols[col]:
                order[row] = col
                used_cols[col] = True
                break
    return order


def smooth_scores(scores, window):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if window <= 1 or scores.size == 0:
        return scores.copy()
    kernel = np.ones(window, dtype=np.float32) / float(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(scores, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def minmax_scale(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < 1e-8:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def deshuffle_localization(audio, visual, rng):
    t = min(len(audio), len(visual))
    target = np.arange(t)
    permutation = rng.permutation(t)
    shuffled_visual = visual[permutation]

    similarity = cosine_similarity_matrix(audio, shuffled_visual)
    matched_shuffled_idx = hungarian_match(similarity)
    recovered_original_idx = permutation[matched_shuffled_idx]

    original_similarity = cosine_similarity_matrix(audio, visual)
    original_diag = np.diag(original_similarity)
    matched_scores = similarity[target, matched_shuffled_idx]
    abs_error = np.abs(recovered_original_idx - target).astype(np.float32)
    normalized_abs_error = abs_error / max(t - 1, 1)

    low_matched_score = (1.0 - matched_scores) / 2.0
    low_original_score = (1.0 - original_diag) / 2.0
    fake_score = 0.70 * minmax_scale(normalized_abs_error) + 0.15 * minmax_scale(low_matched_score)
    fake_score += 0.15 * minmax_scale(low_original_score)

    return {
        "permutation": permutation,
        "matched_shuffled_idx": matched_shuffled_idx,
        "recovered_original_idx": recovered_original_idx.astype(np.float32),
        "matched_scores": matched_scores.astype(np.float32),
        "original_diag": original_diag.astype(np.float32),
        "abs_recovery_error": abs_error,
        "normalized_abs_recovery_error": normalized_abs_error.astype(np.float32),
        "low_matched_score": low_matched_score.astype(np.float32),
        "low_original_score": low_original_score.astype(np.float32),
        "fake_score": fake_score.astype(np.float32),
    }


def deshuffle_localization_rounds(audio, visual, rng, rounds):
    rounds = max(1, int(rounds))
    results = [deshuffle_localization(audio, visual, rng) for _ in range(rounds)]
    averaged_keys = [
        "recovered_original_idx",
        "matched_scores",
        "original_diag",
        "abs_recovery_error",
        "normalized_abs_recovery_error",
        "low_matched_score",
        "low_original_score",
        "fake_score",
    ]
    out = {key: np.mean([result[key] for result in results], axis=0).astype(np.float32) for key in averaged_keys}
    out["rounds"] = rounds
    return out


def accuracy_score(labels, preds):
    if sklearn_accuracy_score is not None:
        return float(sklearn_accuracy_score(labels, preds))
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    return float(np.mean(labels == preds)) if labels.size else np.nan


def precision_score(labels, preds, zero_division=0):
    if sklearn_precision_score is not None:
        return float(sklearn_precision_score(labels, preds, zero_division=zero_division))
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    tp = np.sum((labels == 1) & (preds == 1))
    fp = np.sum((labels == 0) & (preds == 1))
    denom = tp + fp
    return float(tp / denom) if denom else float(zero_division)


def recall_score(labels, preds, zero_division=0):
    if sklearn_recall_score is not None:
        return float(sklearn_recall_score(labels, preds, zero_division=zero_division))
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    tp = np.sum((labels == 1) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    denom = tp + fn
    return float(tp / denom) if denom else float(zero_division)


def f1_score(labels, preds, zero_division=0):
    if sklearn_f1_score is not None:
        return float(sklearn_f1_score(labels, preds, zero_division=zero_division))
    precision = precision_score(labels, preds, zero_division=zero_division)
    recall = recall_score(labels, preds, zero_division=zero_division)
    denom = precision + recall
    return float(2.0 * precision * recall / denom) if denom else float(zero_division)


def roc_auc_score(labels, scores):
    if sklearn_roc_auc_score is not None:
        return float(sklearn_roc_auc_score(labels, scores))
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
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
    if sklearn_average_precision_score is not None:
        return float(sklearn_average_precision_score(labels, scores))
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
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
    if sklearn_roc_curve is not None:
        return sklearn_roc_curve(labels, scores)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if labels.size == 0:
        return np.asarray([]), np.asarray([]), np.asarray([])

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


def safe_ap(labels, scores):
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0 or np.sum(labels == 1) == 0:
        return np.nan
    return float(average_precision_score(labels, scores))


def compute_thresholded_metrics(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(labels) == 0:
        threshold = np.nan
        preds = np.asarray([], dtype=int)
        auc = np.nan
    elif len(np.unique(labels)) < 2:
        threshold = np.nan
        preds = np.zeros_like(labels)
        auc = np.nan
    else:
        auc = float(roc_auc_score(labels, scores))
        fpr, tpr, thresholds = roc_curve(labels, scores)
        finite = np.isfinite(thresholds)
        if finite.any():
            best_idx = int(np.argmax(tpr[finite] - fpr[finite]))
            threshold = float(thresholds[finite][best_idx])
        else:
            threshold = 0.5
        preds = (scores >= threshold).astype(int)

    return {
        "auc": float(auc),
        "threshold": float(threshold),
        "precision": float(precision_score(labels, preds, zero_division=0)) if len(labels) else np.nan,
        "recall": float(recall_score(labels, preds, zero_division=0)) if len(labels) else np.nan,
        "f1": float(f1_score(labels, preds, zero_division=0)) if len(labels) else np.nan,
        "accuracy": float(accuracy_score(labels, preds)) if len(labels) else np.nan,
    }


def top_fraction_mean(values, fraction):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    k = int(np.ceil(len(values) * fraction))
    k = min(max(k, 1), len(values))
    top = np.partition(values, len(values) - k)[-k:]
    return float(np.mean(top))


def aggregate_video_score(frame_scores, mode, top_fraction):
    if mode == "topk_mean":
        return top_fraction_mean(frame_scores, top_fraction)
    if mode == "mean":
        return float(np.mean(frame_scores)) if len(frame_scores) else 0.0
    if mode == "max":
        return float(np.max(frame_scores)) if len(frame_scores) else 0.0
    raise ValueError(f"Unknown video score aggregation: {mode}")


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
    order = np.argsort(-proposals[:, 0], kind="mergesort")
    return proposals[order]


def frame_scores_to_proposals(scores, label_length=None, max_proposals=None):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if label_length is not None:
        scores = scores[:label_length]
    if scores.size == 0:
        return np.empty((0, 3), dtype=float)

    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(np.float32).max, neginf=-np.inf)
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

    proposals = [[confidence, start, end] for (start, end), confidence in proposal_by_span.items()]
    return prepare_proposals(proposals, label_length=label_length)[:max_proposals]


def compute_test_style_segment_metrics(framewise_labels, proposal_segments, metrics_device):
    if AP is None or AR is None or torch is None or metrics_device is None:
        return None, None

    ground_truth_segments = [framewise_labels_to_time_segments(labels) for labels in framewise_labels]
    max_proposals = max(SEGMENT_AR_PROPOSAL_COUNTS)
    proposal_tensors = []
    for labels, proposals in zip(framewise_labels, proposal_segments):
        proposals = prepare_proposals(proposals, label_length=len(labels))[:max_proposals]
        if len(proposals) < max_proposals:
            padded = np.zeros((max_proposals, 3), dtype=np.float32)
            padded[: len(proposals)] = proposals.astype(np.float32, copy=False)
            proposals = padded
        proposal_tensors.append(torch.as_tensor(proposals, dtype=torch.float32, device=metrics_device))

    proposals_tensor = torch.stack(proposal_tensors, dim=0) if proposal_tensors else torch.empty(
        (0, max_proposals, 3), dtype=torch.float32, device=metrics_device
    )
    ap = AP(iou_thresholds=SEGMENT_AP_IOU_THRESHOLDS, device=str(metrics_device))
    ar = AR(n_proposals_list=SEGMENT_AR_PROPOSAL_COUNTS, iou_thresholds=SEGMENT_AR_IOU_THRESHOLDS, device=str(metrics_device))
    return ap(proposals_tensor, ground_truth_segments), ar(proposals_tensor, ground_truth_segments)


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
        return np.asarray([]), np.asarray([])
    return np.concatenate(flat_scores), np.concatenate(flat_labels)


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def finite_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def summarize_numeric(rows, keys):
    out = {}
    for key in keys:
        values = [row[key] for row in rows]
        out[f"{key}_mean"] = finite_mean(values)
        out[f"{key}_median"] = finite_median(values)
    return out


def clean_csv_value(value):
    if isinstance(value, float):
        return f"{value:.8f}" if np.isfinite(value) else "nan"
    return value


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_csv_value(row.get(key, "")) for key in fieldnames})


def write_assumptions(path, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("Audio/visual deshuffling localization assumptions\n")
        handle.write("================================================\n")
        for idx, assumption in enumerate(ASSUMPTIONS, start=1):
            handle.write(f"{idx}. {assumption}\n")
        handle.write("\nRun settings\n")
        handle.write("------------\n")
        handle.write(f"metadata={args.metadata}\n")
        handle.write(f"features_root={args.features_root}\n")
        handle.write(f"max_videos={args.max_videos} (0 means all)\n")
        handle.write(f"max_frames={args.max_frames} (0 means all aligned frames)\n")
        handle.write(f"rounds={args.rounds}\n")
        handle.write(f"smooth_window={args.smooth_window}\n")
        handle.write(f"video_score_aggregation={args.video_score_aggregation}\n")
        handle.write(f"video_top_fraction={args.video_top_fraction}\n")
        handle.write(f"skip_segment_metrics={args.skip_segment_metrics}\n")


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
    if ap_metrics is None or ar_metrics is None:
        print("Skipped: auvire/torch metric dependencies are unavailable.")
        return
    for iou_threshold in SEGMENT_AP_IOU_THRESHOLDS:
        print(f"AP@{iou_threshold:g}: {ap_metrics.get(iou_threshold, 0.0):.4f}")
    for proposal_count in SEGMENT_AR_PROPOSAL_COUNTS:
        print(f"AR@{proposal_count}: {ar_metrics.get(proposal_count, 0.0):.4f}")


def sanitize_filename(value):
    value = str(value).replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._") or "video"


def add_gt_segment_spans(axis, gt_segments):
    for idx, (start, end) in enumerate(gt_segments):
        axis.axvspan(start, end, color="#d62728", alpha=0.16, label="GT fake segment" if idx == 0 else None)


def score_segments(scores, threshold, min_len):
    mask = np.asarray(scores) >= threshold
    segments = []
    for start, end in binary_segments(mask):
        if end - start >= min_len:
            confidence = float(np.mean(scores[start:end]))
            segments.append((confidence, start, end))
    segments.sort(key=lambda item: (-item[0], item[1], item[2]))
    return segments


def plot_localization(path, row):
    if plt is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.arange(row["frames"])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].step(frames, row["frame_labels"], where="post", color="black", label="GT")
    axes[0].set_ylabel("GT")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    add_gt_segment_spans(axes[1], row["gt_segments"])
    axes[1].plot(frames, row["recovered_original_idx"], linewidth=1.0, label="mean recovered visual index")
    axes[1].plot(frames, frames, linestyle="--", linewidth=1.0, label="target")
    axes[1].set_ylabel("Visual frame")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    add_gt_segment_spans(axes[2], row["gt_segments"])
    axes[2].plot(frames, row["fake_score"], linewidth=1.0, label="fake score")
    axes[2].plot(frames, row["smoothed_fake_score"], linewidth=1.0, label="smoothed fake score")
    axes[2].set_xlabel("Audio frame index")
    axes[2].set_ylabel("Score")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    fig.suptitle(f"{row['path']} | label={row['label']} | video_score={row['video_score']:.4f} | frame_ap={row['frame_ap']:.4f}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyze_row(
    features_root,
    row,
    max_frames,
    rng,
    rounds,
    smooth_window,
    threshold_percentile,
    min_segment_len,
    video_score_aggregation,
    video_top_fraction,
):
    audio, visual, feature_path = load_audio_visual(features_root, row, max_frames)
    labels = load_framewise_labels_for_path(feature_path, len(audio))
    result = deshuffle_localization_rounds(audio, visual, rng, rounds)
    fake_score = result["fake_score"]
    smoothed = smooth_scores(fake_score, smooth_window)
    gt_segments = binary_segments(labels)
    frame_metrics = compute_thresholded_metrics(labels, smoothed)
    threshold = float(np.percentile(smoothed, threshold_percentile)) if smoothed.size else np.inf
    pred_segments = score_segments(smoothed, threshold, min_segment_len)

    return {
        "path": row["path"],
        "feature_path": str(feature_path),
        "label": int(row["label"]),
        "type": video_type(row["path"]),
        "frames": len(audio),
        "rounds": int(rounds),
        "gt_segments": gt_segments,
        "gt_segments_str": segment_list_to_string(gt_segments),
        "pred_segments": pred_segments,
        "pred_segments_str": segment_list_to_string([(start, end) for _, start, end in pred_segments]),
        "frame_labels": labels,
        "fake_score": fake_score,
        "smoothed_fake_score": smoothed,
        "video_score": aggregate_video_score(smoothed, video_score_aggregation, video_top_fraction),
        "frame_auc": frame_metrics["auc"],
        "frame_ap": safe_ap(labels, smoothed),
        "mean_fake_score": float(np.mean(fake_score)),
        "max_fake_score": float(np.max(fake_score)),
        "mean_smoothed_fake_score": float(np.mean(smoothed)),
        "max_smoothed_fake_score": float(np.max(smoothed)),
        "mean_abs_recovery_error": float(np.mean(result["abs_recovery_error"])),
        "median_abs_recovery_error": float(np.median(result["abs_recovery_error"])),
        "within_1_frames": float(np.mean(result["abs_recovery_error"] <= 1)),
        "within_3_frames": float(np.mean(result["abs_recovery_error"] <= 3)),
        "within_5_frames": float(np.mean(result["abs_recovery_error"] <= 5)),
        "matched_mean": float(np.mean(result["matched_scores"])),
        "diag_mean": float(np.mean(result["original_diag"])),
        **result,
    }


def framewise_rows(video_rows):
    rows = []
    for row in video_rows:
        for idx in range(row["frames"]):
            rows.append(
                {
                    "path": row["path"],
                    "label": row["label"],
                    "type": row["type"],
                    "frame": idx,
                    "gt_label": int(row["frame_labels"][idx]),
                    "fake_score": float(row["fake_score"][idx]),
                    "smoothed_fake_score": float(row["smoothed_fake_score"][idx]),
                    "abs_recovery_error": float(row["abs_recovery_error"][idx]),
                    "recovered_original_idx": float(row["recovered_original_idx"][idx]),
                    "matched_score": float(row["matched_scores"][idx]),
                    "original_diag": float(row["original_diag"][idx]),
                }
            )
    return rows


def resolve_metrics_device(args):
    if torch is None:
        return None
    if args.metrics_device is not None:
        if args.metrics_device.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(args.metrics_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Localize fake regions by matching audio against shuffled visual frames.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-videos", type=int, default=0, help="0 means process every metadata row.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means use every aligned audio/visual frame.")
    parser.add_argument("--rounds", "-R", type=int, default=1, help="Number of independent deshuffling rounds per video.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--threshold-percentile", type=float, default=85.0)
    parser.add_argument("--min-segment-len", type=int, default=3)
    parser.add_argument("--video-score-aggregation", choices=("topk_mean", "mean", "max"), default="topk_mean")
    parser.add_argument("--video-top-fraction", type=float, default=0.10)
    parser.add_argument("--metrics-device", type=str, default=None, help="Device for AP/AR segment metrics. Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--plot-examples", type=int, default=20)
    parser.add_argument("--skip-segment-metrics", action="store_true", help="Skip AP/AR proposal metrics, which can be slow on the full test set.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print_section("ASSUMPTIONS")
    for assumption in ASSUMPTIONS:
        print(f"- {assumption}")

    print_section("STEP 1: LOAD METADATA")
    metadata_rows = read_metadata(args.metadata)
    selected_rows = select_rows(metadata_rows, args.max_videos)
    print(f"Metadata rows: {len(metadata_rows)}")
    print(f"Selected rows: {len(selected_rows)}")
    print(f"Real selected: {sum(is_real_row(row) for row in selected_rows)}")
    print(f"Fake selected: {sum(not is_real_row(row) for row in selected_rows)}")
    print(f"Max frames: {'all' if args.max_frames <= 0 else args.max_frames}")
    print(f"Deshuffling rounds: {max(1, args.rounds)}")

    print_section("STEP 2: SHUFFLED-VISUAL LOCALIZATION")
    video_rows = []
    failed = []
    for idx, row in enumerate(selected_rows, start=1):
        try:
            video_rows.append(
                analyze_row(
                    args.features_root,
                    row,
                    args.max_frames,
                    rng,
                    args.rounds,
                    args.smooth_window,
                    args.threshold_percentile,
                    args.min_segment_len,
                    args.video_score_aggregation,
                    args.video_top_fraction,
                )
            )
        except Exception as exc:
            failed.append((row.get("path", f"row_{idx}"), repr(exc)))
            continue
        if idx % 25 == 0 or idx == len(selected_rows):
            print(f"Processed {idx}/{len(selected_rows)} rows")

    if failed:
        print(f"Failed rows: {len(failed)}")
        for path, error in failed[:10]:
            print(f"  {path}: {error}")
    print(f"Analyzed rows: {len(video_rows)}")

    print_section("STEP 3: EVAL-STYLE METRICS")
    ground_truths = np.asarray([row["label"] for row in video_rows], dtype=int)
    video_scores = np.asarray([row["video_score"] for row in video_rows], dtype=float)
    framewise_scores = [row["smoothed_fake_score"] for row in video_rows]
    framewise_labels = [row["frame_labels"] for row in video_rows]
    flat_framewise_scores, flat_framewise_labels = flatten_framewise(framewise_scores, framewise_labels)

    video_ap = safe_ap(ground_truths, video_scores)
    framewise_ap = safe_ap(flat_framewise_labels, flat_framewise_scores)
    video_metrics = compute_thresholded_metrics(ground_truths, video_scores)
    framewise_metrics = compute_thresholded_metrics(flat_framewise_labels, flat_framewise_scores)
    print(f"Video-wise AP: {video_ap:.4f}")
    print(f"Framewise AP: {framewise_ap:.4f}")
    print_metric_block("VIDEO-WISE METRICS", video_metrics)
    print_metric_block("FRAMEWISE METRICS", framewise_metrics)

    if args.skip_segment_metrics:
        proposal_segments = []
        segment_ap, segment_ar = None, None
    else:
        print("Building time-segment proposals...")
        proposal_segments = [
            frame_scores_to_proposals(scores, label_length=len(labels))
            for scores, labels in zip(framewise_scores, framewise_labels)
        ]
        metrics_device = resolve_metrics_device(args)
        print(f"Computing AP/AR time-segment metrics on {metrics_device}...")
        segment_ap, segment_ar = compute_test_style_segment_metrics(framewise_labels, proposal_segments, metrics_device)

    print_ap_ar_block("TIME SEGMENT LOCALIZATION METRICS", segment_ap, segment_ar)

    tpr_rows = []
    if len(np.unique(ground_truths)) > 1:
        fpr, tpr, _ = roc_curve(ground_truths, video_scores)
        for target in [0.05, 0.10]:
            best_idx = int(np.argmin(np.abs(fpr - target)))
            value = float(tpr[best_idx])
            tpr_rows.append({"target_fpr": target, "tpr": value})
            print(f"TPR @ {target * 100:.1f}% FPR: {value:.4f}")

    print_section("STEP 4: SAVE OUTPUTS")
    numeric_keys = [
        "video_score",
        "frame_auc",
        "frame_ap",
        "mean_fake_score",
        "max_fake_score",
        "mean_smoothed_fake_score",
        "max_smoothed_fake_score",
        "mean_abs_recovery_error",
        "median_abs_recovery_error",
        "within_1_frames",
        "within_3_frames",
        "within_5_frames",
        "matched_mean",
        "diag_mean",
    ]
    video_fieldnames = [
        "path",
        "feature_path",
        "label",
        "type",
        "frames",
        "rounds",
        "gt_segments_str",
        "pred_segments_str",
        *numeric_keys,
    ]
    write_csv(args.output_dir / "localization_video_summary.csv", video_rows, video_fieldnames)

    write_csv(
        args.output_dir / "localization_frame_scores.csv",
        framewise_rows(video_rows),
        [
            "path",
            "label",
            "type",
            "frame",
            "gt_label",
            "fake_score",
            "smoothed_fake_score",
            "abs_recovery_error",
            "recovered_original_idx",
            "matched_score",
            "original_diag",
        ],
    )

    metric_rows = [
        {"scope": "video", "ap": video_ap, **video_metrics},
        {"scope": "framewise", "ap": framewise_ap, **framewise_metrics},
    ]
    write_csv(
        args.output_dir / "localization_eval_metrics.csv",
        metric_rows,
        ["scope", "ap", "auc", "threshold", "precision", "recall", "f1", "accuracy"],
    )
    write_csv(args.output_dir / "localization_frame_metrics.csv", metric_rows, ["scope", "ap", "auc", "threshold", "precision", "recall", "f1", "accuracy"])

    if segment_ap is not None and segment_ar is not None:
        segment_rows = []
        for iou_threshold in SEGMENT_AP_IOU_THRESHOLDS:
            segment_rows.append({"metric": f"AP@{iou_threshold:g}", "value": float(segment_ap.get(iou_threshold, 0.0))})
        for proposal_count in SEGMENT_AR_PROPOSAL_COUNTS:
            segment_rows.append({"metric": f"AR@{proposal_count}", "value": float(segment_ar.get(proposal_count, 0.0))})
        write_csv(args.output_dir / "localization_segment_metrics.csv", segment_rows, ["metric", "value"])

    if tpr_rows:
        write_csv(args.output_dir / "localization_tpr_at_fpr.csv", tpr_rows, ["target_fpr", "tpr"])

    group_rows = []
    for group in sorted({row["type"] for row in video_rows}):
        rows = [row for row in video_rows if row["type"] == group]
        group_rows.append({"type": group, "count": len(rows), **summarize_numeric(rows, numeric_keys)})
    group_fieldnames = ["type", "count"] + list(group_rows[0].keys())[2:] if group_rows else ["type", "count"]
    write_csv(args.output_dir / "localization_group_summary.csv", group_rows, group_fieldnames)

    write_assumptions(args.output_dir / "matching_assumptions.txt", args)

    if plt is not None:
        plot_candidates = sorted(
            video_rows,
            key=lambda row: (
                row["label"] == 0,
                -row["frame_ap"] if np.isfinite(row["frame_ap"]) else np.inf,
                row["path"],
            ),
        )
        for plotted, row in enumerate(plot_candidates[: args.plot_examples]):
            safe_name = sanitize_filename(Path(row["path"]).stem)
            plot_localization(plots_dir / f"localization_{plotted:03d}_{safe_name}_{row['label']}.png", row)

    print(f"Saved CSVs to: {args.output_dir}")
    if plt is None:
        print("Plots skipped because matplotlib is not installed.")
    else:
        print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
