import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import expit  # for sigmoid (optional)

# ---- Load ----
scores = np.load(
    "Out_Features_trimmed_with_visual_map/debug_framewise_scores.npy",
    allow_pickle=True
)

labels = np.load(
    "Out_Features_trimmed_with_visual_map/debug_framewise_labels.npy",
    allow_pickle=True
)

# ---- Inspect ----
print("Scores type:", type(scores))
print("Labels type:", type(labels))
print("Length:", len(scores), len(labels))

# ---- Convert to flat arrays (frame-level) ----
if isinstance(scores[0], (list, np.ndarray)):
    # object array → list of arrays
    scores_flat = np.concatenate(scores)
    labels_flat = np.concatenate(labels)
else:
    # already flat
    scores_flat = scores
    labels_flat = labels

print("Flat shapes:", scores_flat.shape, labels_flat.shape)

# ---- Ensure numeric ----
scores_flat = scores_flat.astype(np.float32)
labels_flat = labels_flat.astype(np.int32)

# ---- Optional: apply sigmoid if logits ----
# comment this out if scores are already probabilities
scores_flat = expit(scores_flat)

# ---- Frame-level AUC ----
frame_auc = roc_auc_score(labels_flat, scores_flat)
print("Frame-level AUC:", frame_auc)


