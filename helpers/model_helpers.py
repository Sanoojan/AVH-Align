import numpy as np
import torch
import torch.nn.functional as F


def _as_1d_numpy(values):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float).reshape(-1)


def temporal_iou(segment_a, segment_b):
    start = max(float(segment_a[0]), float(segment_b[0]))
    end = min(float(segment_a[1]), float(segment_b[1]))
    intersection = max(0.0, end - start)
    union = max(float(segment_a[1]), float(segment_b[1])) - min(float(segment_a[0]), float(segment_b[0]))
    return intersection / union if union > 0 else 0.0


def temporal_nms(proposals, iou_threshold=0.5, max_segments=3):
    proposals = np.asarray(proposals, dtype=float)
    if proposals.size == 0:
        return np.empty((0, 3), dtype=float)

    proposals = proposals.reshape(-1, 3)
    proposals = proposals[np.isfinite(proposals).all(axis=1)]
    proposals = proposals[proposals[:, 2] > proposals[:, 1]]
    if proposals.size == 0:
        return np.empty((0, 3), dtype=float)

    order = np.argsort(-proposals[:, 0], kind="mergesort")
    selected = []

    for proposal in proposals[order]:
        span = (proposal[1], proposal[2])
        if all(temporal_iou(span, (chosen[1], chosen[2])) <= iou_threshold for chosen in selected):
            selected.append(proposal.tolist())
            if len(selected) >= max_segments:
                break

    return np.asarray(selected, dtype=float).reshape(-1, 3) if selected else np.empty((0, 3), dtype=float)

def make_boundary_labels(frame_labels, radius=2):
    """
    frame_labels: [B,T], binary 0/1
    returns soft_start_labels, soft_end_labels
    """
    B, T = frame_labels.shape
    start = torch.zeros_like(frame_labels).float()
    end = torch.zeros_like(frame_labels).float()

    labels = frame_labels.cpu().numpy()

    for b in range(B):
        y = labels[b]
        segments = []
        s = None

        for i, v in enumerate(y):
            if v == 1 and s is None:
                s = i
            elif v == 0 and s is not None:
                segments.append((s, i - 1))
                s = None

        if s is not None:
            segments.append((s, T - 1))

        for st, ed in segments:
            for offset in range(-radius, radius + 1):
                weight = 1.0 - abs(offset) / (radius + 1)

                si = st + offset
                ei = ed + offset

                if 0 <= si < T:
                    start[b, si] = max(start[b, si], weight)
                if 0 <= ei < T:
                    end[b, ei] = max(end[b, ei], weight)

    return start.to(frame_labels.device), end.to(frame_labels.device)


def boundary_proposals(
    fake_prob,
    start_prob,
    end_prob,
    min_len=3,
    max_len=40,
    topk_starts=20,
    topk_ends=20,
    max_segments=3,
    nms_iou=0.5,
):
    fake_prob = _as_1d_numpy(fake_prob)
    start_prob = _as_1d_numpy(start_prob)
    end_prob = _as_1d_numpy(end_prob)

    T = min(len(fake_prob), len(start_prob), len(end_prob))
    fake_prob = fake_prob[:T]
    start_prob = start_prob[:T]
    end_prob = end_prob[:T]
    if T == 0:
        return np.empty((0, 3), dtype=float)

    start_candidates = np.argsort(-start_prob)[:topk_starts]
    end_candidates = np.argsort(-end_prob)[:topk_ends]

    proposals = []

    for s in start_candidates:
        for e in end_candidates:
            if e < s:
                continue

            length = e - s + 1
            if length < min_len or length > max_len:
                continue

            inside_score = float(np.mean(fake_prob[s:e + 1]))

            score = (
                float(start_prob[s])
                * float(end_prob[e])
                * inside_score
            )

            proposals.append([score, s, e + 1])

    return temporal_nms(
        proposals,
        iou_threshold=nms_iou,
        max_segments=max_segments,
    )