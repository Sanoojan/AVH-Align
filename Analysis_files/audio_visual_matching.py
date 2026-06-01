import argparse
import csv
import sys
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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = REPO_ROOT / "av1m_metadata" / "test_metadata_cleaned.csv"
DEFAULT_FEATURES_ROOT = REPO_ROOT / "Features" / "AV1M-Trimmed" / "test"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Analysis_files" / "audio_visual_matching_outputs"
AUVIRE_ROOT = REPO_ROOT / "auvire"
if str(AUVIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(AUVIRE_ROOT))

try:
    import torch
    from src.metrics import AP, AR
except ModuleNotFoundError:
    torch = None
    AP = None
    AR = None

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

SEGMENT_AP_IOU_THRESHOLDS = [0.5, 0.75, 0.9, 0.95]
SEGMENT_AR_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
SEGMENT_AR_PROPOSAL_COUNTS = [100, 50, 30, 20, 10, 5]
SEGMENT_FPS = 25.0


ASSUMPTIONS = [
    "Zero-shot: no classifier, threshold, or calibration is learned from labels; labels are used only for reporting metrics.",
    "Audio and visual feature arrays are temporally aligned before any artificial shuffle is applied.",
    "A genuine frame should recover its own or a nearby visual frame after visual frames are shuffled and rematched to audio.",
    "A fake visual/audio region is harder to recover, so larger recovery displacement and lower audio-visual cosine similarity imply a higher fake score.",
    "Multiple deshuffling rounds are independent random visual permutations; the final frame score is their average.",
    "The video-level score is the mean of the highest-scoring frame fraction, assuming fake evidence may occupy only part of a video.",
    f"Temporal AP/AR proposals interpret feature frames at {SEGMENT_FPS:g} fps, matching eval_new_time.py.",
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


def select_rows(rows, max_real_videos, max_fake_videos):
    if max_real_videos <= 0 and max_fake_videos <= 0:
        return list(rows)

    selected = []
    real_count = 0
    fake_count = 0
    for row in rows:
        if is_real_row(row):
            if max_real_videos > 0 and real_count >= max_real_videos:
                continue
            real_count += 1
        else:
            if max_fake_videos > 0 and fake_count >= max_fake_videos:
                continue
            fake_count += 1
        selected.append(row)
    return selected


# Backward-compatible alias for older notebooks/imports.
def sample_rows(rows, max_real_videos, max_fake_videos):
    return select_rows(rows, max_real_videos, max_fake_videos)


def load_audio_visual(features_root, row, max_frames):
    path = feature_path_for_row(features_root, row)
    data = np.load(path, allow_pickle=True)
    if "audio" not in data.files or "visual" not in data.files:
        raise KeyError(f"{path} must contain 'audio' and 'visual' arrays; found {data.files}")

    audio = np.asarray(data["audio"], dtype=np.float32)
    visual = np.asarray(data["visual"], dtype=np.float32)
    t = min(len(audio), len(visual))
    if max_frames and max_frames > 0:
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
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

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


def top_fraction_mean(values, fraction):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    k = int(np.ceil(len(values) * fraction))
    k = min(max(k, 1), len(values))
    top = np.partition(values, len(values) - k)[-k:]
    return float(np.mean(top))


def deshuffle_prediction_round(audio, visual, rng):
    t = min(len(audio), len(visual))
    target = np.arange(t)
    permutation = rng.permutation(t)
    shuffled_visual = visual[permutation]

    similarity = cosine_similarity_matrix(audio, shuffled_visual)
    matched_shuffled_idx = hungarian_match(similarity)
    recovered_original_idx = permutation[matched_shuffled_idx]

    original_diag = np.diag(cosine_similarity_matrix(audio, visual))
    matched_scores = similarity[target, matched_shuffled_idx]
    abs_error = np.abs(recovered_original_idx - target)
    normalized_abs_error = abs_error / max(t - 1, 1)

    low_matched_score = np.clip((1.0 - matched_scores) / 2.0, 0.0, 1.0)
    low_original_score = np.clip((1.0 - original_diag) / 2.0, 0.0, 1.0)
    frame_scores = 0.70 * normalized_abs_error + 0.15 * low_matched_score + 0.15 * low_original_score

    return {
        "frame_scores": frame_scores.astype(np.float32),
        "abs_recovery_error": abs_error.astype(np.float32),
        "recovered_original_idx": recovered_original_idx,
        "matched_scores": matched_scores.astype(np.float32),
        "original_diag": original_diag.astype(np.float32),
    }


def deshuffle_prediction(audio, visual, rounds, seed):
    frame_scores = []
    abs_errors = []
    matched_scores = []
    original_diags = []
    recovered_by_round = []

    for round_idx in range(rounds):
        rng = np.random.default_rng(seed + round_idx)
        result = deshuffle_prediction_round(audio, visual, rng)
        frame_scores.append(result["frame_scores"])
        abs_errors.append(result["abs_recovery_error"])
        matched_scores.append(result["matched_scores"])
        original_diags.append(result["original_diag"])
        recovered_by_round.append(result["recovered_original_idx"])

    return {
        "frame_scores": np.mean(frame_scores, axis=0),
        "abs_recovery_error": np.mean(abs_errors, axis=0),
        "matched_scores": np.mean(matched_scores, axis=0),
        "original_diag": np.mean(original_diags, axis=0),
        "recovered_original_idx": recovered_by_round[0],
    }


def assignment_stats(similarity, assignment):
    t = len(assignment)
    frame_idx = np.arange(t)
    diag = np.diag(similarity)
    matched = similarity[frame_idx, assignment]
    abs_disp = np.abs(assignment - frame_idx)
    step = np.diff(assignment) if t > 1 else np.asarray([0])

    return {
        "diag_mean": float(np.mean(diag)),
        "matched_mean": float(np.mean(matched)),
        "matching_gain": float(np.mean(matched) - np.mean(diag)),
        "mean_abs_displacement": float(np.mean(abs_disp)),
        "median_abs_displacement": float(np.median(abs_disp)),
        "within_1_frames": float(np.mean(abs_disp <= 1)),
        "within_3_frames": float(np.mean(abs_disp <= 3)),
        "within_5_frames": float(np.mean(abs_disp <= 5)),
        "assignment_smoothness": float(np.mean(np.abs(step - 1))) if t > 1 else 0.0,
    }


def analyze_row(features_root, row, max_frames, rounds, seed, video_top_fraction):
    audio, visual, path = load_audio_visual(features_root, row, max_frames)
    labels = load_framewise_labels_for_path(path, len(audio))
    gt_segments = binary_segments(labels)

    similarity = cosine_similarity_matrix(audio, visual)
    assignment = hungarian_match(similarity)
    stats = assignment_stats(similarity, assignment)
    pred = deshuffle_prediction(audio, visual, rounds, seed)
    frame_scores = pred["frame_scores"]
    frame_auc, frame_ap = safe_binary_ranking(labels, frame_scores)

    stats.update(
        {
            "path": row["path"],
            "feature_path": str(path),
            "label": int(row["label"]),
            "type": video_type(row["path"]),
            "frames": len(audio),
            "gt_segments": gt_segments,
            "gt_segments_str": segment_list_to_string(gt_segments),
            "assignment": assignment,
            "frame_labels": labels,
            "frame_scores": frame_scores,
            "video_score": top_fraction_mean(frame_scores, video_top_fraction),
            "frame_auc": frame_auc,
            "frame_ap": frame_ap,
            "mean_frame_score": float(np.mean(frame_scores)),
            "max_frame_score": float(np.max(frame_scores)),
            "mean_abs_recovery_error": float(np.mean(pred["abs_recovery_error"])),
            "median_abs_recovery_error": float(np.median(pred["abs_recovery_error"])),
            "matched_score_mean": float(np.mean(pred["matched_scores"])),
            "original_diag_mean": float(np.mean(pred["original_diag"])),
            "abs_recovery_error": pred["abs_recovery_error"],
            "recovered_original_idx": pred["recovered_original_idx"],
            "matched_scores_per_frame": pred["matched_scores"],
            "original_diag_per_frame": pred["original_diag"],
        }
    )
    return stats


def safe_binary_ranking(labels, scores):
    labels = np.asarray(labels, dtype=int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    keep = min(len(labels), len(scores))
    labels = labels[:keep]
    scores = scores[:keep]
    if keep == 0 or len(np.unique(labels)) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def compute_thresholded_metrics(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(labels) == 0:
        preds = np.asarray([], dtype=int)
        threshold = np.nan
        auc = np.nan
    elif len(np.unique(labels)) < 2:
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
        "auc": float(auc),
        "threshold": float(threshold),
        "precision": float(precision_score(labels, preds, zero_division=0)) if len(labels) else np.nan,
        "recall": float(recall_score(labels, preds, zero_division=0)) if len(labels) else np.nan,
        "f1": float(f1_score(labels, preds, zero_division=0)) if len(labels) else np.nan,
        "accuracy": float(accuracy_score(labels, preds)) if len(labels) else np.nan,
    }


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
    if AP is None or AR is None or torch is None:
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


def summarize_numeric(rows, keys):
    out = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        out[f"{key}_mean"] = float(np.mean(finite)) if finite.size else np.nan
        out[f"{key}_median"] = float(np.median(finite)) if finite.size else np.nan
    return out


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean_row = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float):
                    clean_row[key] = f"{value:.8f}" if np.isfinite(value) else "nan"
                else:
                    clean_row[key] = value
            writer.writerow(clean_row)


def write_assumptions(path, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("Audio/visual matching zero-shot assumptions\n")
        handle.write("===========================================\n")
        for idx, assumption in enumerate(ASSUMPTIONS, start=1):
            handle.write(f"{idx}. {assumption}\n")
        handle.write("\nRun settings\n")
        handle.write("------------\n")
        handle.write(f"deshuffle_rounds={args.deshuffle_rounds}\n")
        handle.write(f"video_top_fraction={args.video_top_fraction}\n")
        handle.write(f"max_frames={args.max_frames} (0 means full video)\n")
        handle.write(f"max_real_videos={args.max_real_videos} (0 means all)\n")
        handle.write(f"max_fake_videos={args.max_fake_videos} (0 means all)\n")
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
        print("Skipped: auvire metric dependencies are unavailable.")
        return
    for iou_threshold in SEGMENT_AP_IOU_THRESHOLDS:
        print(f"AP@{iou_threshold:g}: {ap_metrics.get(iou_threshold, 0.0):.4f}")
    for proposal_count in SEGMENT_AR_PROPOSAL_COUNTS:
        print(f"AR@{proposal_count}: {ar_metrics.get(proposal_count, 0.0):.4f}")


def add_gt_segment_spans(axis, gt_segments):
    for idx, (start, end) in enumerate(gt_segments):
        axis.axvspan(start, end, color="#d62728", alpha=0.16, label="GT fake segment" if idx == 0 else None)


def plot_prediction(path, row):
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
    axes[1].plot(frames, row["frame_scores"], linewidth=1.0, label="zero-shot fake score")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    add_gt_segment_spans(axes[2], row["gt_segments"])
    axes[2].plot(frames, row["recovered_original_idx"], linewidth=1.0, label="round-1 recovered visual index")
    axes[2].plot(frames, frames, linestyle="--", linewidth=1.0, label="target")
    axes[2].set_xlabel("Audio frame")
    axes[2].set_ylabel("Visual frame")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    fig.suptitle(f"{row['path']} | video_score={row['video_score']:.4f} | frame_ap={row['frame_ap']:.4f}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_histograms(path, rows, metric):
    if plt is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    real = [row[metric] for row in rows if row["label"] == 0]
    fake = [row[metric] for row in rows if row["label"] == 1]
    plt.figure(figsize=(8, 5))
    plt.hist(real, bins=30, alpha=0.65, label="real", density=True)
    plt.hist(fake, bins=30, alpha=0.65, label="fake", density=True)
    plt.xlabel(metric)
    plt.ylabel("density")
    plt.title(f"Real vs fake: {metric}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


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
                    "score": float(row["frame_scores"][idx]),
                    "abs_recovery_error": float(row["abs_recovery_error"][idx]),
                    "recovered_original_idx": int(row["recovered_original_idx"][idx]),
                    "matched_score": float(row["matched_scores_per_frame"][idx]),
                    "original_diag": float(row["original_diag_per_frame"][idx]),
                }
            )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot audio/visual matching with shuffled-visual deshuffling.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-real-videos", type=int, default=0, help="0 means process all real videos.")
    parser.add_argument("--max-fake-videos", type=int, default=0, help="0 means process all fake videos.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means use every available frame per video.")
    parser.add_argument("--deshuffle-rounds", type=int, default=1, help="Number of random visual shuffles to average.")
    parser.add_argument("--video-top-fraction", type=float, default=0.10, help="Top frame fraction averaged for video score.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-examples", type=int, default=20)
    parser.add_argument("--metrics-device", type=str, default=None, help="Device for AP/AR segment metrics. Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--skip-segment-metrics", action="store_true", help="Skip AP/AR proposal metrics, which can be slow on the full test set.")
    return parser.parse_args()


def resolve_metrics_device(args):
    if torch is None:
        return None
    if args.metrics_device is not None:
        return torch.device(args.metrics_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rounds = max(1, args.deshuffle_rounds)

    print_section("ASSUMPTIONS")
    for assumption in ASSUMPTIONS:
        print(f"- {assumption}")

    print_section("STEP 1: LOAD METADATA")
    metadata_rows = read_metadata(args.metadata)
    selected_rows = select_rows(metadata_rows, args.max_real_videos, args.max_fake_videos)
    print(f"Metadata rows: {len(metadata_rows)}")
    print(f"Selected rows: {len(selected_rows)}")
    print(f"Real selected: {sum(is_real_row(row) for row in selected_rows)}")
    print(f"Fake selected: {sum(not is_real_row(row) for row in selected_rows)}")
    print(f"Max frames: {'all' if args.max_frames <= 0 else args.max_frames}")
    print(f"Deshuffle rounds: {rounds}")

    print_section("STEP 2: ZERO-SHOT DESHUFFLE PREDICTIONS")
    video_rows = []
    failed = []
    for idx, row in enumerate(selected_rows, start=1):
        try:
            stats = analyze_row(
                args.features_root,
                row,
                args.max_frames,
                rounds,
                args.seed + idx * max(rounds, 1),
                args.video_top_fraction,
            )
            video_rows.append(stats)
        except Exception as exc:
            failed.append((row["path"], repr(exc)))
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
    framewise_scores = [row["frame_scores"] for row in video_rows]
    framewise_labels = [row["frame_labels"] for row in video_rows]
    flat_framewise_scores, flat_framewise_labels = flatten_framewise(framewise_scores, framewise_labels)

    video_ap = float(average_precision_score(ground_truths, video_scores)) if len(np.unique(ground_truths)) > 1 else np.nan
    framewise_ap = (
        float(average_precision_score(flat_framewise_labels, flat_framewise_scores))
        if len(np.unique(flat_framewise_labels)) > 1
        else np.nan
    )
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

    if len(np.unique(ground_truths)) > 1:
        fpr, tpr, _ = roc_curve(ground_truths, video_scores)
        for target in [0.05, 0.10]:
            best_idx = int(np.argmin(np.abs(fpr - target)))
            print(f"TPR @ {target * 100:.1f}% FPR: {tpr[best_idx]:.4f}")

    print_section("STEP 4: SAVE OUTPUTS")
    numeric_keys = [
        "video_score",
        "frame_auc",
        "frame_ap",
        "mean_frame_score",
        "max_frame_score",
        "mean_abs_recovery_error",
        "median_abs_recovery_error",
        "matched_score_mean",
        "original_diag_mean",
        "diag_mean",
        "matched_mean",
        "matching_gain",
        "mean_abs_displacement",
        "within_3_frames",
        "assignment_smoothness",
    ]
    video_fieldnames = ["path", "feature_path", "label", "type", "frames", "gt_segments_str", *numeric_keys]
    write_csv(args.output_dir / "matching_video_summary.csv", video_rows, video_fieldnames)

    group_rows = []
    for group in sorted({row["type"] for row in video_rows}):
        rows = [row for row in video_rows if row["type"] == group]
        group_rows.append({"type": group, "count": len(rows), **summarize_numeric(rows, numeric_keys)})
    group_fieldnames = ["type", "count"] + list(group_rows[0].keys())[2:] if group_rows else ["type", "count"]
    write_csv(args.output_dir / "matching_group_summary.csv", group_rows, group_fieldnames)

    write_csv(
        args.output_dir / "matching_frame_scores.csv",
        framewise_rows(video_rows),
        [
            "path",
            "label",
            "type",
            "frame",
            "gt_label",
            "score",
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
    write_csv(args.output_dir / "matching_eval_metrics.csv", metric_rows, ["scope", "ap", "auc", "threshold", "precision", "recall", "f1", "accuracy"])

    if segment_ap is not None and segment_ar is not None:
        segment_rows = []
        for iou_threshold in SEGMENT_AP_IOU_THRESHOLDS:
            segment_rows.append({"metric": f"AP@{iou_threshold:g}", "value": float(segment_ap.get(iou_threshold, 0.0))})
        for proposal_count in SEGMENT_AR_PROPOSAL_COUNTS:
            segment_rows.append({"metric": f"AR@{proposal_count}", "value": float(segment_ar.get(proposal_count, 0.0))})
        write_csv(args.output_dir / "matching_segment_metrics.csv", segment_rows, ["metric", "value"])

    write_assumptions(args.output_dir / "matching_assumptions.txt", args)

    if plt is not None:
        for metric in ["video_score", "mean_abs_recovery_error", "matched_score_mean", "diag_mean"]:
            plot_histograms(plots_dir / "histograms" / f"{metric}.png", video_rows, metric)
        for plot_idx, row in enumerate(video_rows[: args.plot_examples]):
            plot_prediction(plots_dir / "predictions" / f"example_{plot_idx:03d}_{Path(row['path']).stem}_{row['label']}.png", row)

    print(f"Saved CSVs to: {args.output_dir}")
    if plt is None:
        print("Plots skipped because matplotlib is not installed.")
    else:
        print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
