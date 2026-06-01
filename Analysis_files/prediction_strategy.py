import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
AUVIRE_ROOT = REPO_ROOT / "auvire"
if str(AUVIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(AUVIRE_ROOT))

from src.metrics import AP, AR  # noqa: E402


AP_IOU_THRESHOLDS = [0.5, 0.75, 0.9, 0.95]
AR_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
AR_PROPOSAL_COUNTS = [100, 50, 30, 20, 10, 5]
FPS = 25.0


def print_section(title):
    print(f"\n{title}")
    print("=" * len(title))


def load_required_array(features_dir, test_name, suffix):
    path = features_dir / f"{test_name}_{suffix}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path, allow_pickle=True)


def load_optional_array(features_dir, test_name, suffix):
    path = features_dir / f"{test_name}_{suffix}.npy"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def as_1d_float(values):
    return np.asarray(values, dtype=float).reshape(-1)


def as_1d_int(values):
    return np.asarray(values, dtype=int).reshape(-1)


def aligned_scores_labels(scores, labels):
    scores = as_1d_float(scores)
    labels = as_1d_int(labels)
    keep = min(len(scores), len(labels))
    return scores[:keep], labels[:keep]


def binary_segments(values):
    values = as_1d_int(values)
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


def framewise_labels_to_time_segments(labels):
    return [(start / FPS, end / FPS) for start, end in binary_segments(labels)]


def segment_iou(segment_a, segment_b):
    start = max(segment_a[0], segment_b[0])
    end = min(segment_a[1], segment_b[1])
    intersection = max(0, end - start)
    union = max(segment_a[1], segment_b[1]) - min(segment_a[0], segment_b[0])
    return intersection / union if union > 0 else 0.0


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


def smooth_scores(scores, window):
    scores = as_1d_float(scores)
    if window <= 1 or scores.size == 0:
        return scores.copy()

    kernel = np.ones(window, dtype=float) / float(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(scores, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def topk_mean(values, k):
    values = as_1d_float(values)
    if values.size == 0:
        return 0.0
    k = min(max(int(k), 1), len(values))
    top = np.partition(values, len(values) - k)[-k:]
    return float(np.mean(top))


def segment_confidence(scores, start, end, mode):
    region = as_1d_float(scores[start:end])
    if region.size == 0:
        return 0.0
    if mode == "max":
        return float(np.max(region))
    if mode == "mean":
        return float(np.mean(region))
    if mode == "top3":
        return topk_mean(region, 3)
    raise ValueError(f"Unknown confidence mode: {mode}")


def merge_segments(segments, max_gap):
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def threshold_component_proposals(
    scores,
    threshold,
    max_segments,
    smooth_window,
    merge_gap,
    min_len,
    expand,
    confidence_mode,
):
    raw_scores = as_1d_float(scores)
    work_scores = smooth_scores(raw_scores, smooth_window)
    segments = binary_segments(work_scores >= threshold)
    segments = merge_segments(segments, merge_gap)

    proposals = []
    for start, end in segments:
        if end - start < min_len:
            continue
        proposal_start = max(0, start - expand)
        proposal_end = min(len(raw_scores), end + expand)
        confidence = segment_confidence(work_scores, start, end, confidence_mode)
        proposals.append([confidence, proposal_start, proposal_end])

    return prepare_proposals(proposals, label_length=len(raw_scores))[:max_segments]


def eval_style_proposals(scores, max_proposals=100):
    scores = as_1d_float(scores)
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

    proposal_by_span = {}
    for threshold in np.unique(finite_scores)[::-1]:
        for start, end in binary_segments(scores >= threshold):
            confidence = float(np.max(scores[start:end]))
            span = (int(start), int(end))
            proposal_by_span[span] = max(confidence, proposal_by_span.get(span, -np.inf))

        if len(proposal_by_span) >= max_proposals:
            break

    proposals = [[conf, start, end] for (start, end), conf in proposal_by_span.items()]
    return prepare_proposals(proposals, label_length=len(scores))[:max_proposals]


def temporal_nms(proposals, iou_threshold, max_segments):
    proposals = prepare_proposals(proposals)
    selected = []

    for proposal in proposals:
        start, end = float(proposal[1]), float(proposal[2])
        keep = True
        for chosen in selected:
            if segment_iou((start, end), (chosen[1], chosen[2])) > iou_threshold:
                keep = False
                break
        if keep:
            selected.append(proposal.tolist())
        if len(selected) >= max_segments:
            break

    return np.asarray(selected, dtype=float).reshape(-1, 3) if selected else np.empty((0, 3), dtype=float)


def top_window_proposals(
    scores,
    window_lengths,
    max_segments,
    smooth_window,
    confidence_mode,
    nms_iou,
    candidates_per_length,
    min_window_confidence=0.0,
):
    raw_scores = as_1d_float(scores)
    work_scores = smooth_scores(raw_scores, smooth_window)
    proposals = []

    for window_length in window_lengths:
        window_length = int(window_length)
        if window_length <= 0 or window_length > len(work_scores):
            continue

        if confidence_mode == "mean":
            cumsum = np.r_[0.0, np.cumsum(work_scores)]
            confidences = (cumsum[window_length:] - cumsum[:-window_length]) / window_length
        elif confidence_mode == "max":
            confidences = np.asarray(
                [np.max(work_scores[start : start + window_length]) for start in range(len(work_scores) - window_length + 1)]
            )
        elif confidence_mode == "top3":
            confidences = np.asarray(
                [topk_mean(work_scores[start : start + window_length], 3) for start in range(len(work_scores) - window_length + 1)]
            )
        else:
            raise ValueError(f"Unknown window confidence mode: {confidence_mode}")

        top_count = min(len(confidences), max_segments * candidates_per_length)
        if top_count == 0:
            continue
        candidate_starts = np.argsort(-confidences, kind="mergesort")[:top_count]
        for start in candidate_starts:
            confidence = float(confidences[start])
            if confidence >= min_window_confidence:
                proposals.append([confidence, int(start), int(start + window_length)])

    return temporal_nms(proposals, iou_threshold=nms_iou, max_segments=max_segments)


def oracle_boundary_proposals(scores, labels, max_segments):
    scores, labels = aligned_scores_labels(scores, labels)
    proposals = []
    for start, end in binary_segments(labels):
        confidence = segment_confidence(scores, start, end, "max")
        proposals.append([confidence, start, end])
    return prepare_proposals(proposals, label_length=len(labels))[:max_segments]


def build_metric_tensors(all_proposals, labels, metrics_device, max_proposals):
    tensors = []
    ground_truth_segments = []

    for proposals, labels_for_video in zip(all_proposals, labels):
        labels_for_video = as_1d_int(labels_for_video)
        proposals = prepare_proposals(proposals, label_length=len(labels_for_video))[:max_proposals]
        padded = np.zeros((max_proposals, 3), dtype=np.float32)
        if len(proposals):
            padded[: len(proposals)] = proposals.astype(np.float32, copy=False)
        tensors.append(torch.as_tensor(padded, dtype=torch.float32, device=metrics_device))
        ground_truth_segments.append(framewise_labels_to_time_segments(labels_for_video))

    if tensors:
        proposal_tensor = torch.stack(tensors, dim=0)
    else:
        proposal_tensor = torch.empty((0, max_proposals, 3), dtype=torch.float32, device=metrics_device)

    return proposal_tensor, ground_truth_segments


def evaluate_proposals(all_proposals, labels, metrics_device, max_proposals=100):
    proposal_tensor, ground_truth_segments = build_metric_tensors(
        all_proposals,
        labels,
        metrics_device=metrics_device,
        max_proposals=max_proposals,
    )

    ap = AP(iou_thresholds=AP_IOU_THRESHOLDS, device=str(metrics_device))(proposal_tensor, ground_truth_segments)
    ar = AR(
        n_proposals_list=AR_PROPOSAL_COUNTS,
        iou_thresholds=AR_IOU_THRESHOLDS,
        device=str(metrics_device),
    )(proposal_tensor, ground_truth_segments)

    counts = np.asarray([len(prepare_proposals(p)) for p in all_proposals], dtype=float)
    return {
        "ap50": float(ap.get(0.5, 0.0)),
        "ap75": float(ap.get(0.75, 0.0)),
        "ap90": float(ap.get(0.9, 0.0)),
        "ap95": float(ap.get(0.95, 0.0)),
        "ar100": float(ar.get(100, 0.0)),
        "ar50": float(ar.get(50, 0.0)),
        "ar30": float(ar.get(30, 0.0)),
        "ar20": float(ar.get(20, 0.0)),
        "ar10": float(ar.get(10, 0.0)),
        "ar5": float(ar.get(5, 0.0)),
        "mean_proposals": float(np.mean(counts)) if counts.size else 0.0,
        "zero_proposal_pct": float(np.mean(counts == 0) * 100.0) if counts.size else 0.0,
    }


def flatten_aligned(framewise_scores, framewise_labels):
    scores_out = []
    labels_out = []

    for scores, labels in zip(framewise_scores, framewise_labels):
        scores, labels = aligned_scores_labels(scores, labels)
        scores_out.append(scores)
        labels_out.append(labels)

    return np.concatenate(scores_out), np.concatenate(labels_out)


def quantiles(values, qs=(10, 25, 50, 75, 90)):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {}
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def summarize_data(framewise_scores, framewise_labels, outputs=None, ground_truths=None):
    print_section("STEP 1: DATA SUMMARY")
    lengths = [min(len(as_1d_float(s)), len(as_1d_int(l))) for s, l in zip(framewise_scores, framewise_labels)]
    print(f"Videos: {len(framewise_scores)}")
    print(f"Frame length quantiles: {quantiles(lengths)}")
    print(f"Total aligned frames: {int(np.sum(lengths))}")

    if outputs is not None:
        outputs = as_1d_float(outputs)
        print(f"Video outputs: shape={outputs.shape}, min={outputs.min():.4f}, max={outputs.max():.4f}, mean={outputs.mean():.4f}")
    if ground_truths is not None:
        ground_truths = as_1d_int(ground_truths)
        print(f"Video labels: shape={ground_truths.shape}, positive_rate={ground_truths.mean():.4f}")


def summarize_gt_segments(framewise_labels):
    print_section("STEP 2: GT SEGMENT STATISTICS")
    counts = []
    lengths = []
    positive_rates = []

    for labels in framewise_labels:
        labels = as_1d_int(labels)
        segments = binary_segments(labels)
        counts.append(len(segments))
        positive_rates.append(float(np.mean(labels)) if len(labels) else 0.0)
        lengths.extend(end - start for start, end in segments)

    unique_counts, count_freq = np.unique(counts, return_counts=True)
    print(f"Positive videos: {int(np.sum(np.asarray(counts) > 0))}")
    print(f"GT segment count distribution: {dict(zip(unique_counts.astype(int).tolist(), count_freq.astype(int).tolist()))}")
    print(f"Max GT segments in one video: {int(max(counts)) if counts else 0}")
    print(f"GT segment length quantiles: {quantiles(lengths)}")
    print(f"Positive frame-rate quantiles: {quantiles(positive_rates)}")


def summarize_score_calibration(framewise_scores, framewise_labels):
    print_section("STEP 3: SCORE CALIBRATION")
    flat_scores, flat_labels = flatten_aligned(framewise_scores, framewise_labels)
    positive_scores = flat_scores[flat_labels == 1]
    negative_scores = flat_scores[flat_labels == 0]

    print(f"Frame positive rate: {float(np.mean(flat_labels)):.4f}")
    print(f"Positive score quantiles: {quantiles(positive_scores, qs=(50, 75, 90, 95, 99))}")
    print(f"Negative score quantiles: {quantiles(negative_scores, qs=(50, 75, 90, 95, 99))}")


def strategy_rows_to_print(rows, limit=12):
    columns = ["name", "ap50", "ap75", "ap90", "ap95", "ar100", "ar5", "mean_proposals", "zero_proposal_pct"]
    print(",".join(columns))
    for row in rows[:limit]:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        print(",".join(values))


def make_strategy_specs(max_segments, full_sweep=False):
    threshold_specs = []
    if full_sweep:
        threshold_grid = [
            (smooth_window, threshold)
            for smooth_window in [1, 3, 5, 9, 15]
            for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]
        ]
    else:
        threshold_grid = [(1, 0.4), (1, 0.5), (5, 0.5), (9, 0.5), (15, 0.5)]

    for smooth_window, threshold in threshold_grid:
        threshold_specs.append(
            {
                "kind": "threshold",
                "name": (
                    f"threshold_s{smooth_window}_t{threshold:g}_"
                    f"merge5_min5_expand3_top{max_segments}_maxconf"
                ),
                "threshold": threshold,
                "smooth_window": smooth_window,
                "merge_gap": 5,
                "min_len": 5,
                "expand": 3,
                "confidence_mode": "max",
            }
        )

    if full_sweep:
        window_grid = [
            (smooth_window, confidence_mode, window_lengths)
            for smooth_window in [1, 3, 5, 9]
            for confidence_mode in ["mean", "top3"]
            for window_lengths in [
                [4, 6, 8, 10, 12, 16, 24, 32],
                [6, 8, 10, 12, 16],
                [8, 12, 16, 24, 32],
                [4,6,8,10,12,16]
            ]
        ]
    else:
        window_grid = [
            (1, "mean", [4, 6, 8, 10, 12, 16, 24, 32]),
            (1, "top3", [4, 6, 8, 10, 12, 16, 24, 32]),
            (3, "mean", [4, 6, 8, 10, 12, 16, 24, 32]),
            (5, "mean", [4, 6, 8, 10, 12, 16, 24, 32]),
            (1, "mean", [6, 8, 10, 12, 16]),
            (1, "mean", [8, 12, 16, 24, 32]),
        ]

    window_specs = []
    for smooth_window, confidence_mode, window_lengths in window_grid:
        length_name = "-".join(str(value) for value in window_lengths)
        window_specs.append(
            {
                "kind": "top_window",
                "name": f"top_window_s{smooth_window}_{confidence_mode}_w{length_name}_top{max_segments}",
                "smooth_window": smooth_window,
                "window_lengths": window_lengths,
                "confidence_mode": confidence_mode,
                "nms_iou": 0.5,
                "candidates_per_length": 4,
            }
        )

    return threshold_specs + window_specs


def proposals_for_spec(spec, framewise_scores, framewise_labels, max_segments, min_window_confidence=0.0):
    if spec["kind"] == "eval_style":
        return [eval_style_proposals(scores) for scores in framewise_scores]

    if spec["kind"] == "saved":
        return [prepare_proposals(proposals, label_length=len(as_1d_int(labels))) for proposals, labels in zip(spec["proposals"], framewise_labels)]

    if spec["kind"] == "threshold":
        return [
            threshold_component_proposals(
                scores,
                threshold=spec["threshold"],
                max_segments=max_segments,
                smooth_window=spec["smooth_window"],
                merge_gap=spec["merge_gap"],
                min_len=spec["min_len"],
                expand=spec["expand"],
                confidence_mode=spec["confidence_mode"],
            )
            for scores in framewise_scores
        ]

    if spec["kind"] == "top_window":
        return [
            top_window_proposals(
                scores,
                window_lengths=spec["window_lengths"],
                max_segments=max_segments,
                smooth_window=spec["smooth_window"],
                confidence_mode=spec["confidence_mode"],
                nms_iou=spec["nms_iou"],
                candidates_per_length=spec["candidates_per_length"],
                min_window_confidence=min_window_confidence,
            )
            for scores in framewise_scores
        ]

    if spec["kind"] == "oracle":
        return [oracle_boundary_proposals(scores, labels, max_segments=max_segments) for scores, labels in zip(framewise_scores, framewise_labels)]

    raise ValueError(f"Unknown strategy kind: {spec['kind']}")


def evaluate_strategy(spec, framewise_scores, framewise_labels, metrics_device, max_segments, min_window_confidence=0.0):
    proposals = proposals_for_spec(
        spec,
        framewise_scores,
        framewise_labels,
        max_segments=max_segments,
        min_window_confidence=min_window_confidence,
    )
    metrics = evaluate_proposals(proposals, framewise_labels, metrics_device=metrics_device, max_proposals=100)
    row = {"name": spec["name"], "kind": spec["kind"]}
    row.update(metrics)
    return row, proposals


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "kind",
        "ap50",
        "ap75",
        "ap90",
        "ap95",
        "ar100",
        "ar50",
        "ar30",
        "ar20",
        "ar10",
        "ar5",
        "mean_proposals",
        "zero_proposal_pct",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved strategy CSV: {path}")


def sanitize_filename(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._") or "item"


def best_iou_for_video(proposals, labels):
    labels = as_1d_int(labels)
    gt_segments = binary_segments(labels)
    proposals = prepare_proposals(proposals, label_length=len(labels))
    if len(gt_segments) == 0:
        return None
    if len(proposals) == 0:
        return 0.0

    best = 0.0
    for proposal in proposals:
        pred = (int(proposal[1]), int(proposal[2]))
        for gt in gt_segments:
            best = max(best, segment_iou(pred, gt))
    return best


def print_failure_diagnostics(best_name, best_proposals, framewise_scores, framewise_labels, limit=10):
    print_section("STEP 7: LOW-IOU FAILURE DIAGNOSTICS")
    failures = []
    for idx, (proposals, scores, labels) in enumerate(zip(best_proposals, framewise_scores, framewise_labels)):
        best_iou = best_iou_for_video(proposals, labels)
        if best_iou is None:
            continue
        scores, labels = aligned_scores_labels(scores, labels)
        gt_segments = binary_segments(labels)
        pred_segments = [(int(p[1]), int(p[2]), float(p[0])) for p in prepare_proposals(proposals, len(labels))]
        failures.append((best_iou, idx, len(labels), gt_segments, pred_segments, float(np.max(scores)) if len(scores) else 0.0))

    failures.sort(key=lambda item: (item[0], item[1]))
    print(f"Best strategy: {best_name}")
    print("idx,best_iou,length,gt_segments,pred_segments,max_score")
    for best_iou, idx, length, gt_segments, pred_segments, max_score in failures[:limit]:
        print(f"{idx},{best_iou:.4f},{length},{gt_segments},{pred_segments},{max_score:.4f}")


def segment_list_to_string(segments):
    return ";".join(f"{start / FPS:.3f}-{end / FPS:.3f}" for start, end in segments)


def proposal_list_to_string(proposals):
    values = []
    for confidence, start, end in proposals:
        values.append(f"{start / FPS:.3f}-{end / FPS:.3f}@{confidence:.4f}")
    return ";".join(values)


def write_plot_summary_csv(path, best_proposals, framewise_scores, framewise_labels):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_index",
                "duration_seconds",
                "best_iou",
                "gt_segments_seconds",
                "pred_segments_seconds_confidence",
                "max_score",
            ],
        )
        writer.writeheader()
        for idx, (proposals, scores, labels) in enumerate(zip(best_proposals, framewise_scores, framewise_labels)):
            scores, labels = aligned_scores_labels(scores, labels)
            prepared = prepare_proposals(proposals, label_length=len(labels))
            gt_segments = binary_segments(labels)
            best_iou = best_iou_for_video(prepared, labels)
            writer.writerow(
                {
                    "video_index": idx,
                    "duration_seconds": f"{len(labels) / FPS:.3f}",
                    "best_iou": "" if best_iou is None else f"{best_iou:.6f}",
                    "gt_segments_seconds": segment_list_to_string(gt_segments),
                    "pred_segments_seconds_confidence": proposal_list_to_string(prepared),
                    "max_score": f"{float(np.max(scores)):.6f}" if len(scores) else "0.000000",
                }
            )


def select_plot_indices(best_proposals, framewise_labels, plot_limit):
    total = len(framewise_labels)
    if plot_limit == 0 or plot_limit >= total:
        return list(range(total))

    scored = []
    for idx, (proposals, labels) in enumerate(zip(best_proposals, framewise_labels)):
        best_iou = best_iou_for_video(proposals, labels)
        if best_iou is None:
            best_iou = 1.0
        scored.append((best_iou, idx))

    low_iou = [idx for _, idx in sorted(scored, key=lambda item: (item[0], item[1]))[: plot_limit // 2]]
    remaining = plot_limit - len(low_iou)
    spread = np.linspace(0, total - 1, num=remaining, dtype=int).tolist() if remaining > 0 else []
    selected = []
    seen = set()
    for idx in low_iou + spread:
        if idx not in seen:
            selected.append(int(idx))
            seen.add(int(idx))
    return selected[:plot_limit]


def has_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def plot_single_prediction(path, video_idx, scores, labels, proposals, best_name):
    if path.suffix.lower() == ".svg":
        plot_single_prediction_svg(path, video_idx, scores, labels, proposals, best_name)
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores, labels = aligned_scores_labels(scores, labels)
    proposals = prepare_proposals(proposals, label_length=len(labels))
    gt_segments = binary_segments(labels)
    best_iou = best_iou_for_video(proposals, labels)
    times = np.arange(len(scores)) / FPS
    duration = len(scores) / FPS

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 4.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    axes[0].plot(times, scores, color="#222222", linewidth=1.0)
    axes[0].set_ylabel("Score")
    upper = min(1.05, max(1.0, float(np.max(scores)) + 0.05) if len(scores) else 1.0)
    axes[0].set_ylim(-0.03, upper)
    axes[0].grid(True, alpha=0.25)

    for start, end in gt_segments:
        axes[0].axvspan(start / FPS, end / FPS, color="#d62728", alpha=0.18)
    for confidence, start, end in proposals:
        axes[0].axvspan(start / FPS, end / FPS, color="#1f77b4", alpha=0.16)
        axes[0].text(
            (start + end) / (2 * FPS),
            upper * 0.92,
            f"{confidence:.2f}",
            color="#1f77b4",
            ha="center",
            va="top",
            fontsize=8,
        )

    for start, end in gt_segments:
        axes[1].broken_barh([(start / FPS, (end - start) / FPS)], (0.60, 0.28), facecolors="#d62728")
    for confidence, start, end in proposals:
        axes[1].broken_barh([(start / FPS, (end - start) / FPS)], (0.15, 0.28), facecolors="#1f77b4")

    axes[1].set_yticks([0.29, 0.74])
    axes[1].set_yticklabels(["Pred", "GT"])
    axes[1].set_ylim(0, 1)
    axes[1].set_xlim(0, max(duration, 1.0 / FPS))
    axes[1].set_xlabel("Time (seconds)")
    axes[1].grid(True, axis="x", alpha=0.25)

    iou_text = "none" if best_iou is None else f"{best_iou:.3f}"
    fig.suptitle(f"video={video_idx} | best_iou={iou_text} | {best_name}", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_single_prediction_svg(path, video_idx, scores, labels, proposals, best_name):
    scores, labels = aligned_scores_labels(scores, labels)
    proposals = prepare_proposals(proposals, label_length=len(labels))
    gt_segments = binary_segments(labels)
    best_iou = best_iou_for_video(proposals, labels)

    width, height = 1200, 460
    left, right = 70, 30
    top, score_bottom = 60, 260
    lane_top, lane_bottom = 315, 405
    plot_width = width - left - right
    duration = max(len(scores) / FPS, 1.0 / FPS)
    upper = min(1.05, max(1.0, float(np.max(scores)) + 0.05) if len(scores) else 1.0)

    def x_from_frame(frame):
        return left + (frame / FPS) / duration * plot_width

    def y_from_score(score):
        return score_bottom - (score / upper) * (score_bottom - top)

    def rect_for_segment(start, end, y, h, color, opacity):
        x = x_from_frame(start)
        w = max(1.0, x_from_frame(end) - x)
        return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{color}" opacity="{opacity}" />'

    points = []
    for frame, score in enumerate(scores):
        points.append(f"{x_from_frame(frame):.2f},{y_from_score(score):.2f}")

    iou_text = "none" if best_iou is None else f"{best_iou:.3f}"
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{left}" y="28" font-family="Arial" font-size="16" fill="#222">video={video_idx} | best_iou={iou_text} | {best_name}</text>',
        f'<line x1="{left}" y1="{score_bottom}" x2="{width - right}" y2="{score_bottom}" stroke="#999" stroke-width="1" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{score_bottom}" stroke="#999" stroke-width="1" />',
        f'<text x="18" y="160" font-family="Arial" font-size="13" fill="#333">Score</text>',
        f'<text x="18" y="354" font-family="Arial" font-size="13" fill="#333">GT</text>',
        f'<text x="18" y="394" font-family="Arial" font-size="13" fill="#333">Pred</text>',
    ]
    for tick in np.linspace(0, duration, num=6):
        x = left + tick / duration * plot_width
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{lane_bottom}" stroke="#ddd" stroke-width="1" />')
        svg.append(f'<text x="{x:.2f}" y="435" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">{tick:.1f}s</text>')
    for start, end in gt_segments:
        svg.append(rect_for_segment(start, end, top, score_bottom - top, "#d62728", "0.18"))
        svg.append(rect_for_segment(start, end, 336, 26, "#d62728", "0.90"))
    for confidence, start, end in proposals:
        svg.append(rect_for_segment(start, end, top, score_bottom - top, "#1f77b4", "0.16"))
        svg.append(rect_for_segment(start, end, 376, 26, "#1f77b4", "0.90"))
        svg.append(f'<text x="{x_from_frame((start + end) / 2):.2f}" y="52" text-anchor="middle" font-family="Arial" font-size="11" fill="#1f77b4">{confidence:.2f}</text>')
    if points:
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#222" stroke-width="1.2" />')
    svg.append(f'<text x="{left + plot_width / 2:.2f}" y="455" text-anchor="middle" font-family="Arial" font-size="13" fill="#333">Time (seconds)</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg))


def save_best_strategy_plots(best_name, best_proposals, framewise_scores, framewise_labels, output_dir, plot_limit):
    print_section("STEP 8: BEST STRATEGY PLOTS")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "segments_summary.csv"
    write_plot_summary_csv(summary_path, best_proposals, framewise_scores, framewise_labels)

    plot_ext = ".png" if has_matplotlib() else ".svg"
    if plot_ext == ".svg":
        print("matplotlib is not installed; saving SVG timeline plots instead of PNG files.")

    selected_indices = select_plot_indices(best_proposals, framewise_labels, plot_limit)
    for idx in selected_indices:
        best_iou = best_iou_for_video(best_proposals[idx], framewise_labels[idx])
        iou_label = "none" if best_iou is None else f"{best_iou:.3f}"
        filename = f"video_{idx:05d}_iou_{sanitize_filename(iou_label)}{plot_ext}"
        plot_single_prediction(
            output_dir / filename,
            idx,
            framewise_scores[idx],
            framewise_labels[idx],
            best_proposals[idx],
            best_name,
        )

    print(f"Saved segment summary CSV: {summary_path}")
    print(f"Saved {len(selected_indices)} timeline plots to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and tune framewise-score-to-segment prediction strategies."
    )
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "Out_Features_trimmed_with_visual_map")
    parser.add_argument("--test-name", type=str, default="hard_synth_boundary_conv_av_only_with_full_eval_new_sc_50_prop_wo_real_boundary")
    parser.add_argument("--metrics-device", type=str, default="cpu")
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--rank-metric", choices=["ap50", "ap75", "ap90", "ap95"], default="ap50")
    parser.add_argument("--save-csv", type=Path, default=REPO_ROOT / "Analysis_files" / "hard_synth_boundary_conv_av_only_with_full_eval_new_sc_50_prop_wo_real_boundary.csv")
    parser.add_argument("--include-oracle", action="store_true", help="Add a label-using upper-bound diagnostic row.")
    parser.add_argument("--full-sweep", action="store_true", help="Run the wider strategy grid; the default grid is faster.")
    parser.add_argument("--failure-limit", type=int, default=10)
    parser.add_argument("--plot-dir", type=Path, default=REPO_ROOT / "Plots" / "prediction_strategy"/"hard_synth_boundary_conv_av_only_with_full_eval_new_sc_50_prop_wo_real_boundary")
    parser.add_argument("--plot-limit", type=int, default=100, help="Number of timeline PNGs to save. Use 0 to plot every video.")
    parser.add_argument(
        "--min-window-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence required for top-window proposals. Use values like 0.05-0.2 to suppress low-score real-video predictions.",
    )
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    features_dir = args.features_dir
    metrics_device = torch.device(args.metrics_device)

    framewise_scores = load_required_array(features_dir, args.test_name, "framewise_scores")
    framewise_labels = load_required_array(features_dir, args.test_name, "framewise_labels")
    saved_proposals = load_optional_array(features_dir, args.test_name, "proposals")
    outputs = load_optional_array(features_dir, args.test_name, "outputs")
    ground_truths = load_optional_array(features_dir, args.test_name, "ground_truths")

    if len(framewise_scores) != len(framewise_labels):
        raise ValueError(f"Score/label video count mismatch: {len(framewise_scores)} vs {len(framewise_labels)}")

    summarize_data(framewise_scores, framewise_labels, outputs=outputs, ground_truths=ground_truths)
    summarize_gt_segments(framewise_labels)
    summarize_score_calibration(framewise_scores, framewise_labels)
    print(f"\nTop-window minimum confidence: {args.min_window_confidence:.4f}")

    print_section("STEP 4: BASELINE PROPOSAL METRICS")
    baseline_specs = [{"kind": "eval_style", "name": "baseline_eval_style_unique_thresholds"}]
    if saved_proposals is not None:
        baseline_specs.append({"kind": "saved", "name": "baseline_saved_debug_proposals", "proposals": saved_proposals})

    rows = []
    proposals_by_name = {}
    for spec in baseline_specs:
        row, proposals = evaluate_strategy(
            spec,
            framewise_scores,
            framewise_labels,
            metrics_device=metrics_device,
            max_segments=args.max_segments,
            min_window_confidence=args.min_window_confidence,
        )
        rows.append(row)
        proposals_by_name[row["name"]] = proposals
    strategy_rows_to_print(rows, limit=len(rows))

    print_section("STEP 5: STRATEGY SWEEP RESULTS")
    strategy_specs = make_strategy_specs(max_segments=args.max_segments, full_sweep=args.full_sweep)
    for idx, spec in enumerate(strategy_specs, start=1):
        row, proposals = evaluate_strategy(
            spec,
            framewise_scores,
            framewise_labels,
            metrics_device=metrics_device,
            max_segments=args.max_segments,
            min_window_confidence=args.min_window_confidence,
        )
        rows.append(row)
        proposals_by_name[row["name"]] = proposals
        print(
            f"[{idx:03d}/{len(strategy_specs):03d}] {row['name']} "
            f"AP@0.5={row['ap50']:.4f} AP@0.75={row['ap75']:.4f} "
            f"mean_props={row['mean_proposals']:.2f}"
        )

    if args.include_oracle:
        print_section("STEP 6A: ORACLE DIAGNOSTIC")
        oracle_spec = {"kind": "oracle", "name": f"oracle_gt_boundaries_top{args.max_segments}"}
        oracle_row, oracle_proposals = evaluate_strategy(
            oracle_spec,
            framewise_scores,
            framewise_labels,
            metrics_device=metrics_device,
            max_segments=args.max_segments,
            min_window_confidence=args.min_window_confidence,
        )
        rows.append(oracle_row)
        proposals_by_name[oracle_row["name"]] = oracle_proposals
        strategy_rows_to_print([oracle_row], limit=1)

    rows_for_ranking = [row for row in rows if row["kind"] != "oracle"]
    rows_sorted = sorted(rows_for_ranking, key=lambda row: (-row[args.rank_metric], row["name"]))

    print_section("STEP 6: BEST STRATEGY DETAILS")
    print(f"Rank metric: {args.rank_metric}")
    strategy_rows_to_print(rows_sorted, limit=15)
    best_row = rows_sorted[0]
    best_proposals = proposals_by_name[best_row["name"]]
    print(f"\nBest strategy: {best_row['name']}")
    print(
        f"Best metrics: AP@0.5={best_row['ap50']:.4f}, AP@0.75={best_row['ap75']:.4f}, "
        f"AP@0.9={best_row['ap90']:.4f}, AP@0.95={best_row['ap95']:.4f}"
    )

    print_failure_diagnostics(
        best_name=best_row["name"],
        best_proposals=best_proposals,
        framewise_scores=framewise_scores,
        framewise_labels=framewise_labels,
        limit=args.failure_limit,
    )

    plot_output_dir = args.plot_dir / f"prediction_strategy_{sanitize_filename(args.test_name)}_{sanitize_filename(best_row['name'])}"
    save_best_strategy_plots(
        best_name=best_row["name"],
        best_proposals=best_proposals,
        framewise_scores=framewise_scores,
        framewise_labels=framewise_labels,
        output_dir=plot_output_dir,
        plot_limit=args.plot_limit,
    )

    if args.save_csv:
        write_csv(args.save_csv, rows)


if __name__ == "__main__":
    main()
