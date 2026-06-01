import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from model import FusionModel, SimpleTemporalFusionAV_only, TemporalFusionModel, SimpleTemporalFusion


def parse_args():
    parser = argparse.ArgumentParser(description="Plot baseline vs ours framewise comparisons.")
    parser.add_argument("--dataset", type=str, default="AV1M", choices=["AV1M", "FakeAVCeleb"])
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--features_root", type=str, default=None)
    parser.add_argument("--labels_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_quads", type=int, default=200)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/AVH-Align_AV1M.pt")
    parser.add_argument(
        "--checkpoint_path_ours",
        type=str,
        default="checkpoints/AVH_A_V_simple_mlp_train_val_splits_video_wise_trimm_compare_500.pt",
    )
    return parser.parse_args()


args = parse_args()

# ==========================================
# Reproducibility
# ==========================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# Model Loading
# ==========================================
checkpoint_path = args.checkpoint_path
checkpoint_path_ours = args.checkpoint_path_ours
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fusion_model_weights = torch.load(checkpoint_path, weights_only=False)
simple_fusion_weights = torch.load(checkpoint_path_ours, weights_only=False)

fusion_model = FusionModel().to(device)
fusion_model.load_state_dict(fusion_model_weights["state_dict"])
fusion_model.eval()

SimpleTemporalFusion = SimpleTemporalFusionAV_only().to(device)
SimpleTemporalFusion.load_state_dict(simple_fusion_weights["state_dict"])
SimpleTemporalFusion.eval()

# ==========================================
# Paths
# ==========================================
default_metadata = {
    "AV1M": "av1m_metadata/test_metadata.csv",
    "FakeAVCeleb": "data/DeepfakeDatasets/FakeAVCeleb/video/meta_data.csv",
}
default_features_root = {
    "AV1M": "Features/AV1M-Trimmed/test",
    "FakeAVCeleb": "Features/FakeAVCeleb-Trimmed",
}

meta_data_path = args.metadata or default_metadata[args.dataset]
features_root = args.features_root or default_features_root[args.dataset]
labels_root = args.labels_root or ("Features/Labels/test" if args.dataset == "AV1M" else None)
output_dir = args.output_dir or f"Plots/random_quadruplets_{args.dataset}_baseline_ours_final_comp"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# Load Metadata
# ==========================================
df = pd.read_csv(meta_data_path)

if args.dataset == "AV1M":
    df["plot_type"] = df["path"].apply(lambda x: x.split("/")[-1].replace(".mp4", ""))
elif args.dataset == "FakeAVCeleb":
    favc_type_map = {
        "RealVideo-RealAudio": "real",
        "FakeVideo-RealAudio": "fake_video_real_audio",
        "FakeVideo-FakeAudio": "fake_video_fake_audio",
        "RealVideo-FakeAudio": "real_video_fake_audio",
    }
    df["plot_type"] = df["type"].map(favc_type_map)
    df = df[df["plot_type"].notna()].copy()
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

real_df = df[df["plot_type"] == "real"]
fvra_df = df[df["plot_type"] == "fake_video_real_audio"]
fvfa_df = df[df["plot_type"] == "fake_video_fake_audio"]
rvfa_df = df[df["plot_type"] == "real_video_fake_audio"]

print("Counts:")
print("Real:", len(real_df))
print("Fake Video Real Audio:", len(fvra_df))
print("Fake Video Fake Audio:", len(fvfa_df))
print("Real Video Fake Audio:", len(rvfa_df))

required_groups = {
    "real": real_df,
    "fake_video_real_audio": fvra_df,
    "fake_video_fake_audio": fvfa_df,
    "real_video_fake_audio": rvfa_df,
}
missing_groups = [name for name, group_df in required_groups.items() if len(group_df) == 0]
if missing_groups:
    raise ValueError(f"Metadata is missing required groups: {missing_groups}")

# ==========================================
# Utilities
# ==========================================
def build_paths(row):
    if args.dataset == "AV1M":
        base = row["path"].replace(".mp4", "")
        feature_path = f"{features_root}/{base}.npz"
        label_path = f"{labels_root}/{base}_labels.npz"
        return feature_path, label_path

    base = row["path"].replace("FakeAVCeleb/", "", 1)
    filename = row["filename"].replace(".mp4", ".npz")
    feature_path = os.path.join(features_root, base, filename)
    label_path = None
    return feature_path, label_path


def load_labels(label_path):
    if label_path is None:
        return None

    if not os.path.exists(label_path):
        return None

    data = np.load(label_path, allow_pickle=True)

    if "framewise_labels" in data:
        labels = data["framewise_labels"]
    else:
        labels = list(data.values())[0]

    return labels.astype(int)


def build_labels(row, label_path, num_frames):
    if args.dataset == "FakeAVCeleb":
        label = 0 if row["type"] == "RealVideo-RealAudio" else 1
        return np.full(num_frames, label, dtype=int)

    labels = load_labels(label_path)
    if labels is not None:
        labels = labels[:num_frames]
    return labels

def process_video_ours(visual_tensor, audio_tensor,multimodal_tensor):
    visual_tensor = visual_tensor.to(device)
    audio_tensor = audio_tensor.to(device)
    multimodal_tensor = multimodal_tensor.to(device)
   
   # features shape  (B, 3, T, D)
   
    # features=torch.stack([visual_tensor, audio_tensor, multimodal_tensor], dim=1)  # (T, 3, D)
    features=torch.stack([visual_tensor, audio_tensor], dim=1) 
    features = features.transpose(0,1).unsqueeze(0)  # (1, 3, T, D)

    output = SimpleTemporalFusion(features)
    ours_logits = output.detach().cpu().squeeze(1).view(-1)  # (T,)
    # breakpoint()

    score_ours = torch.logsumexp(ours_logits, dim=0) 

    return score_ours.item(), ours_logits.numpy()

def process_video(visual_tensor, audio_tensor):
    visual_tensor = visual_tensor.to(device)
    audio_tensor = audio_tensor.to(device)

    output = fusion_model(visual_tensor, audio_tensor)
    baseline_logits = output.detach().cpu().squeeze(1)

    score_baseline = torch.logsumexp(-baseline_logits, dim=0) 

    return score_baseline.item(), baseline_logits.numpy()

# ==========================================
# Metric Computation
# ==========================================
def compute_metrics(feature_path):
    data = np.load(feature_path, allow_pickle=True)

    multimodal = torch.tensor(data["multimodal"]).float()
    visual = torch.tensor(data["visual"]).float()
    audio = torch.tensor(data["audio"]).float()
    visual_map = torch.tensor(data["visual_map"]).float()

    multimodal_norm = F.normalize(multimodal, dim=-1)
    visual_norm = F.normalize(visual, dim=-1)
    audio_norm = F.normalize(audio, dim=-1)
    visual_map_norm = F.normalize(visual_map, dim=-1)

    # ---- OURS (Diagonal cosine aggregation) ----
    visual_diag = torch.diagonal(visual_map_norm, dim1=0, dim2=1).transpose(0,1)
    diag_cos = torch.sum(visual_diag * multimodal_norm, dim=-1)
    
    # behaves like a mean with low values. 0.52
    # score_ours = torch.logsumexp(-diag_cos, dim=0) 
    
    #min 0.73
    # score_ours = diag_cos.min() 

    # top -10 0.72 top-5 0.733
    # k = max(1, int(0.05 * len(diag_cos)))
    # score_ours = -torch.topk(diag_cos, k, largest=False).values.mean()
    
    # # quantile: 0.69
    # score_ours = -torch.quantile(diag_cos, 0.05)
    
    # threshold = 0.85
    # score_ours = (diag_cos < threshold).float().sum() # 0.71
    
    # tail = torch.clamp(0.85 - diag_cos, min=0)
    # score_ours = tail.sum() # 0.7327
    
    # smooth = torch.nn.functional.avg_pool1d(
    #     diag_cos.view(1,1,-1),
    #     kernel_size=5,
    #     stride=1,
    #     padding=2
    # ).view(-1)

    # score_ours = -smooth.min(). # 0.5628
    
    # lowest = torch.topk(diag_cos, 10, largest=False).values.mean()
    # highest = torch.topk(diag_cos, 10, largest=True).values.mean()

    # score_ours = highest - lowest # 0.72
    
    score_ours= diag_cos.std()  #0.73
    
    # mu = diag_cos.mean()
    # score_ours = torch.clamp(mu - diag_cos, min=0).sum() 0.66
        
    

    # ---- Row Mean Cosine ----
    multimodal_exp = multimodal_norm.unsqueeze(0).expand_as(visual_map_norm)
    cos_full = torch.sum(visual_map_norm * multimodal_exp, dim=-1)
    row_mean_cos = cos_full.mean(dim=1)
    # score_ours = torch.logsumexp(-row_mean_cos, dim=0) / row_mean_cos.size(0)

    # ---- Audio vs Visual ----
    audio_vs_visual = torch.sum(audio_norm * visual_norm, dim=-1)

    # ---- Baseline Score ----
    score_baseline, baseline_logits = process_video(visual_norm, audio_norm)

    return {
        "diag_cos": diag_cos.numpy(),
        "row_mean_cos": row_mean_cos.numpy(),
        "audio_vs_visual": audio_vs_visual.numpy(),
        "baseline": baseline_logits,
        "score_ours": score_ours.item(),
        "score_baseline": score_baseline,
    }

def compute_metrics(feature_path, drop_last_n=0):

    data = np.load(feature_path, allow_pickle=True)
    # breakpoint()
    multimodal = torch.tensor(data["multimodal"]).float()
    visual = torch.tensor(data["visual"]).float()
    audio = torch.tensor(data["audio"]).float()
    

    multimodal_norm = F.normalize(multimodal, dim=-1)
    visual_norm = F.normalize(visual, dim=-1)
    audio_norm = F.normalize(audio, dim=-1)
    
    # ---- OURS (Diagonal cosine aggregation) ----
    audio_multimodal = torch.sum(audio_norm * multimodal_norm, dim=-1)
    visual_multimodal = torch.sum(visual_norm * multimodal_norm, dim=-1)
    audio_vs_visual = torch.sum(audio_norm * visual_norm, dim=-1)
    score_baseline, baseline_logits = process_video( visual_norm, audio_norm)
    baseline_logits = torch.tensor(baseline_logits)
    # breakpoint()
    score_ours, logits_ours = process_video_ours( visual_norm, audio_norm, multimodal_norm)
    logits_ours = torch.tensor(logits_ours)
    

    
    if drop_last_n > 0:
        T = len(multimodal_norm)
        keep_T = max(1, T - drop_last_n)
        multimodal_norm = multimodal_norm[:keep_T]
        visual_norm = visual_norm[:keep_T]
        audio_norm = audio_norm[:keep_T]
        audio_multimodal = audio_multimodal[:keep_T]
        visual_multimodal = visual_multimodal[:keep_T]
        baseline_logits = baseline_logits[:keep_T]
        logits_ours = logits_ours[:keep_T]

        

    # ==========================================
    # OUR SCORE 
    # ==========================================
    # mu = diag_cos.mean()
    # score_ours = torch.clamp(mu - diag_cos, min=0).sum()
    
    # k = max(1, int(0.05 * len(diag_cos)))
    # score_ours = -torch.topk(diag_cos, k, largest=False).values.mean()
    
    # score_ours= diag_cos.min()

    return {
        "audio_multimodal": audio_multimodal.numpy(),
        "visual_multimodal": visual_multimodal.numpy(),
        "audio_vs_visual": audio_vs_visual.numpy(),
        "baseline": baseline_logits,
        "ours": -logits_ours,
        "score_ours": score_ours,
        "score_baseline": score_baseline,
        "num_frames": len(multimodal_norm)
    }

# ==========================================
# Plotting Function
# ==========================================
def plot_metric(ax, metric_name, ylabel, metrics_dict, labels_dict):

    color_map = {
        "real": "black",
        "fake_video_real_audio": "red",
        "fake_video_fake_audio": "blue",
        "real_video_fake_audio": "gray",
    }

    for video_type, metrics in metrics_dict.items():

        values = metrics[metric_name]
        frames = np.arange(len(values))
        color = color_map.get(video_type, "gray")

        ax.plot(frames,
                values,
                label=video_type,
                color=color,
                linewidth=2)

        # Highlight fake frames (aligned after trimming)
        labels = labels_dict.get(video_type)

        if labels is not None and len(labels) == len(values):

            fake_idx = np.where(labels == 1)[0]

            if len(fake_idx) > 0:
                ax.scatter(fake_idx,
                           values[fake_idx],
                           color=color,
                           edgecolors="yellow",
                           marker="o",
                           s=60,
                           zorder=3)

    metric_name = metric_name.capitalize()
    ax.set_title(metric_name)
    ax.set_xlabel("Frame Index (trimmed)")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(fontsize=8)

# ==========================================
# Generate Quadruplets
# ==========================================
num_quads = args.num_quads
quadruplets = []

for _ in range(num_quads):
    quad = {
        "real": real_df.sample(1).iloc[0].to_dict(),
        "fake_video_real_audio": fvra_df.sample(1).iloc[0].to_dict(),
        "fake_video_fake_audio": fvfa_df.sample(1).iloc[0].to_dict(),
        "real_video_fake_audio": rvfa_df.sample(1).iloc[0].to_dict(),
    }
    quadruplets.append(quad)

# ==========================================
# Evaluation Storage
# ==========================================
all_labels = []
all_scores_ours = []
all_scores_baseline = []

# ==========================================
# Process Quadruplets
# ==========================================
for idx, quad in enumerate(quadruplets):

    metrics_dict = {}
    labels_dict = {}

    for video_type, row in quad.items():
        # if video_type == "fake_video_real_audio":
        #     continue
        feature_path, label_path = build_paths(row)

        if not os.path.exists(feature_path):
            print("Missing feature:", feature_path)
            continue

        try:
            metrics = compute_metrics(feature_path)
            labels = build_labels(row, label_path, metrics["num_frames"])

            metrics_dict[video_type] = metrics
            labels_dict[video_type] = labels
            
            

            label = 0 if video_type == "real" else 1

            all_labels.append(label)
            all_scores_ours.append(metrics["score_ours"])
            all_scores_baseline.append(metrics["score_baseline"])

        except Exception as e:
            print("Error:", feature_path, e)
            continue

    if len(metrics_dict) != 4:
        continue

    # ---- Plot ----
    plt.figure(figsize=(18, 6))

    plt.subplot(1,2,1)
    plot_metric(plt.gca(), "baseline", "Logits", metrics_dict, labels_dict)

    plt.subplot(1,2,2)
    plot_metric(plt.gca(), "ours", "Negated Logits", metrics_dict, labels_dict)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"quad_{idx}.jpeg")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved {save_path}")

# ==========================================
# AUC Evaluation
# ==========================================
all_labels = np.array(all_labels)
all_scores_ours = np.array(all_scores_ours)
all_scores_baseline = np.array(all_scores_baseline)

# Ensure correct direction
if roc_auc_score(all_labels, all_scores_ours) < 0.5:
    all_scores_ours = -all_scores_ours

if roc_auc_score(all_labels, all_scores_baseline) < 0.5:
    all_scores_baseline = -all_scores_baseline

auc_ours = roc_auc_score(all_labels, all_scores_ours)
auc_baseline = roc_auc_score(all_labels, all_scores_baseline)

from sklearn.metrics import roc_curve

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(all_labels, all_scores_ours)

# Target FPRs
target_fprs = [0.01,0.02,0.05, 0.10]

for target in target_fprs:
    # Find closest FPR index
    idx = np.argmin(np.abs(fpr - target))
    print(f"Ours TPR @ {target*100:.1f}% FPR: {tpr[idx]:.4f}")

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(all_labels, all_scores_baseline)

# Target FPRs
target_fprs = [0.01,0.02,0.05, 0.10]

for target in target_fprs:
    # Find closest FPR index
    idx = np.argmin(np.abs(fpr - target))
    print(f"Baseline TPR @ {target*100:.1f}% FPR: {tpr[idx]:.4f}")

print("\n==============================")
print("VIDEO LEVEL AUC COMPARISON")
print("==============================")
print(f"Our Score AUC     : {auc_ours:.4f}")
print(f"Baseline AUC      : {auc_baseline:.4f}")
print("==============================")

# ==========================================
# ROC Curve
# ==========================================
fpr_ours, tpr_ours, _ = roc_curve(all_labels, all_scores_ours)
fpr_base, tpr_base, _ = roc_curve(all_labels, all_scores_baseline)

plt.figure()
plt.plot(fpr_ours, tpr_ours, label=f"Ours (AUC={auc_ours:.3f})")
plt.plot(fpr_base, tpr_base, label=f"Baseline (AUC={auc_baseline:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()

roc_path = os.path.join(output_dir, "auc_comparison.png")
plt.savefig(roc_path, dpi=300)
plt.show()

print("ROC curve saved to:", roc_path)
print("Done ✅")
