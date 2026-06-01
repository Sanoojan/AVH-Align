import argparse
import json
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
import os
import cv2
import torch.nn.functional as F

from model import FusionModel
from utils import seed_run
from sklearn.metrics import roc_curve

Out_features_path = "Out_Features_trimmed_with_visual_map"
if not os.path.exists(Out_features_path):
    os.makedirs(Out_features_path)
    
Validation_meta_path = "data/DeepfakeDatasets/AV_Deepfake1M/AV-Deepfake1M-PlusPlus/val_metadata.json"


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

def process_video(data, fusion_model, device):
    visual_tensor = torch.from_numpy(data["visual"]).to(device)
    audio_tensor = torch.from_numpy(data["audio"]).to(device)
    # breakpoint()
    # L2 norm
    visual_tensor = visual_tensor / (torch.linalg.norm(visual_tensor, ord=2, dim=-1, keepdim=True))
    audio_tensor = audio_tensor / (torch.linalg.norm(audio_tensor, ord=2, dim=-1, keepdim=True))

    output = fusion_model(visual_tensor, audio_tensor)
    score = torch.logsumexp(-output, dim=0).detach().cpu().squeeze()

    return score,output

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


def main(args):
    seed_run()

    print(f"Evaluating AVH-Align on {args.dataset} with pretrained weights saved at {args.checkpoint_path} ...")

    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_model_weights = torch.load(args.checkpoint_path, weights_only=False)

    fusion_model = FusionModel().to(device)
    fusion_model.load_state_dict(fusion_model_weights["state_dict"])
    fusion_model.eval()
    
    # Load metadata for access to labels
    metadata = pd.read_csv(args.metadata)

    outputs = []
    ground_truths = []
    framewise_scores = []
    path_names = []
    framewis_labels=[]
    for _, row in tqdm(metadata.iterrows()):
        data = np.load(os.path.join(args.features_path, row["path"].replace(".mp4", ".npz")), allow_pickle=True)
        label = row["label"]
        score,output = process_visual_map_zero_shot(data, fusion_model, device)
        # score,output = process_video(data, fusion_model, device)
        # score,output= Zero_shot_process_video(data, fusion_model, device)
        outputs.append(score)
        framewise_scores.append(output.detach().cpu().numpy())
        ground_truths.append(label)
        path_names.append(row["path"])
        framewis_labels.append(get_framewise_labels(row["path"], output.shape[0]))
        # breakpoint()
        
    outputs = np.array(outputs)
    ground_truths = np.array(ground_truths)

    auc = roc_auc_score(ground_truths, outputs)
    ap = average_precision_score(ground_truths, outputs)

    print(f"AP: {ap}")
    print(f"AUC: {auc}")
    
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
    parser.add_argument("--test_name", type=str, default="debug",
                        help="Test name")

    args = parser.parse_args()
    main(args)
