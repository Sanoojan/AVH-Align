import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from model import FusionModel

# ==========================================
# Paths
# ==========================================
meta_data_path="av1m_metadata/test_metadata.csv"


real_path = "Features/AV1M-Trimmed_with_visual_map/test/id01036/cUU3puNO39M/00082/real.npz"
fake_path = "Features/AV1M-Trimmed_with_visual_map/test/id00943/9gsva0mLrVc/00031/fake_video_real_audio.npz"

real_video = "test/id01036/cUU3puNO39M/00082/real"
fake_video = "test/id00943/9gsva0mLrVc/00031/fake_video_real_audio"

labels_parent_dir = "Features/Labels"

real_framewise_labels_path = f"{labels_parent_dir}/{real_video}_labels.npz"
fake_framewise_labels_path = f"{labels_parent_dir}/{fake_video}_labels.npz"


# ==========================================
# Metric Computation
# ==========================================

checkpoint_path = "checkpoints/AVH-Align_AV1M.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion_model_weights = torch.load(checkpoint_path, weights_only=False)

fusion_model = FusionModel().to(device)
fusion_model.load_state_dict(fusion_model_weights["state_dict"])
fusion_model.eval()

def process_video(visual_tensor, audio_tensor, fusion_model, device):
    

    output = fusion_model(visual_tensor, audio_tensor)
    score = torch.logsumexp(-output, dim=0).detach().cpu().squeeze()

    return score,output


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

    # ---- Diagonal cosine ----
    visual_diag = torch.diagonal(visual_map_norm, dim1=0, dim2=1).transpose(0,1)
    diag_cos = torch.sum(visual_diag * multimodal_norm, dim=-1)
    
    #AVH align score (negative log-sum-exp of diagonal cosines)
    scor,align_cos = process_video(visual_norm, audio_norm, fusion_model, device)

    # ---- Row mean cosine ----
    multimodal_exp = multimodal_norm.unsqueeze(0).expand_as(visual_map_norm)
    cos_full = torch.sum(visual_map_norm * multimodal_exp, dim=-1)
    row_mean_cos = cos_full.mean(dim=1)

    # ---- Visual vs MM ----
    visual_vs_mm = torch.sum(visual_norm * multimodal_norm, dim=-1)
    
    #---- audio vs visual --
    audio_vs_visual = torch.sum(audio_norm * visual_norm, dim=-1)
    

    # ---- L2 diagonal ----
    visual_diag_raw = torch.diagonal(visual_map, dim1=0, dim2=1).transpose(0,1)
    l2_diag = torch.norm(visual_diag_raw - multimodal, dim=-1)

    return {
        "diag_cos": diag_cos.numpy(),
        "row_mean_cos": row_mean_cos.numpy(),
        "visual_vs_mm": visual_vs_mm.numpy(),
        "audio_vs_visual": audio_vs_visual.numpy(),
        "l2_diag": l2_diag.numpy(),
    }


# ==========================================
# Load Framewise Labels
# ==========================================

def load_labels(label_path):
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return None

    data = np.load(label_path, allow_pickle=True)

    # Change key here if needed
    if "framewise_labels" in data:
        labels = data["framewise_labels"]
    else:
        # fallback: assume first array
        labels = list(data.values())[0]

    return labels.astype(int)


# ==========================================
# Compute Metrics
# ==========================================

real_metrics = compute_metrics(real_path)
fake_metrics = compute_metrics(fake_path)

real_labels = load_labels(real_framewise_labels_path)
fake_labels = load_labels(fake_framewise_labels_path)

T_real = len(real_metrics["diag_cos"])
T_fake = len(fake_metrics["diag_cos"])

frames_real = np.arange(T_real)
frames_fake = np.arange(T_fake)


# ==========================================
# Plotting Helper
# ==========================================

# breakpoint()

def plot_metric(ax, metric_name, ylabel):
    real_values = real_metrics[metric_name]
    fake_values = fake_metrics[metric_name]

    # Plot real video
    ax.plot(frames_real, real_values,
            marker='o', linestyle='-',
            markevery=5, label="Real")

    # Plot fake video
    ax.plot(frames_fake, fake_values,
            marker='x', linestyle='--',
            markevery=5, label="Fake")

    # Highlight fake frames inside fake video
    if fake_labels is not None:
        fake_frame_indices = np.where(fake_labels == 1)[0]
        ax.scatter(fake_frame_indices,
                   fake_values[fake_frame_indices],
                   marker='s',
                   s=60,
                   label="Fake Frames")

    ax.set_title(metric_name)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()


# ==========================================
# Create Plots
# ==========================================

plt.figure(figsize=(18, 12))

plt.subplot(2,2,1)
plot_metric(plt.gca(), "diag_cos", "Cosine")

plt.subplot(2,2,2)
plot_metric(plt.gca(), "row_mean_cos", "Cosine")

plt.subplot(2,2,3)
plot_metric(plt.gca(), "audio_vs_visual", "Cosine")

plt.subplot(2,2,4)
plot_metric(plt.gca(), "l2_diag", "L2 Distance")

plt.tight_layout()
plt.savefig("Plots/visual_map_analysis_compare_with_labels_4.png")


print("Plot saved to Plots/visual_map_analysis_compare_with_labels_4.png")