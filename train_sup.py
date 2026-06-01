import os
import torch
import wandb
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from config import get_args
from dataset import AVFeatureDataset
from model import TemporalFusionModel, SimpleTemporalFusion, SimpleTemporalFusionAV_only
from utils import print_args


# =========================================================
# CHECKPOINT
# =========================================================
def save_checkpoint(state, is_best, save_path, model_name):
    if is_best:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"{model_name}.pt")
        torch.save(state, save_file)
        return save_file
    return None


def class_std_loss(logits, labels, eps=1e-6):

    """

    logits : (B,T)

    labels : (B,T)

    Encourages low variance inside each class.

    """

    

    fake_mask = labels > 0.5

    real_mask = labels < 0.5

    loss = 0.0

    count = 0

    if fake_mask.sum() > 1:

        fake_logits = logits[fake_mask]

        fake_std = fake_logits.std(unbiased=False)

        loss += fake_std

        count += 1

    if real_mask.sum() > 1:

        real_logits = logits[real_mask]

        real_std = real_logits.std(unbiased=False)

        loss += real_std

        count += 1

    if count == 0:

        return torch.tensor(0.0, device=logits.device)

    return loss / count

# =========================================================
# SAFE AUC
# =========================================================
def safe_auc(labels, probs):
    try:
        if len(np.unique(labels)) < 2:
            return 0.0
        return roc_auc_score(labels, probs)
    except:
        return 0.0


# =========================================================
# RUN EPOCH
# =========================================================
def run_epoch(dataloader, model, device, optimizer=None, use_tqdm=False):
    is_training = optimizer is not None

    model.train() if is_training else model.eval()

    total_loss = 0

    # Frame-level
    total_frame_correct = 0
    total_frames = 0
    all_frame_probs = []
    all_frame_labels = []

    # Video-level
    total_video_correct = 0
    total_videos = 0
    all_video_probs = []
    all_video_labels = []

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0], device=device))
    criterion_video = torch.nn.BCEWithLogitsLoss()

    loader = tqdm(dataloader) if use_tqdm else dataloader
    num_batches=0
    with torch.set_grad_enabled(is_training):
        for batch in loader:

            features, labels = batch  # (B,3,T,D), (B,T)
            # breakpoint()
            features = features.to(device)
            labels = labels.to(device).float()
            # print(features.shape, labels.shape)
            logits = model(features)  # (B,T)
            # video_logits = logits.max(dim=1)[0]  # (B,)
            video_logits= torch.logsumexp(logits, dim=1)
            video_labels = labels.max(dim=1)[0]  # (B,)
            # loss =  criterion_video(video_logits, video_labels) 
            
            # loss = criterion(logits, labels)
            bce_loss = criterion(logits, labels)
            std_loss = class_std_loss(logits, labels)
            loss = bce_loss + 0.1 * std_loss
            num_batches += 1
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # ---------------------------
            # FRAME METRICS
            # ---------------------------
            total_frame_correct += (preds == labels).sum().item()
            total_frames += labels.numel()
            total_loss += loss.item()

            all_frame_probs.append(probs.detach().cpu())
            all_frame_labels.append(labels.detach().cpu())

            # ---------------------------
            # VIDEO METRICS
            # ---------------------------
            video_labels = labels.max(dim=1)[0]   # ANY fake frame
            video_probs = probs.max(dim=1)[0]

            video_preds = (video_probs > 0.5).float()

            total_video_correct += (video_preds == video_labels).sum().item()
            total_videos += labels.shape[0]

            all_video_probs.append(video_probs.detach().cpu())
            all_video_labels.append(video_labels.detach().cpu())

    # ---------------------------
    # AGGREGATE
    # ---------------------------
    avg_loss = total_loss / num_batches

    frame_acc = total_frame_correct / total_frames
    video_acc = total_video_correct / total_videos

    all_frame_probs = torch.cat(all_frame_probs).view(-1).numpy()
    all_frame_labels = torch.cat(all_frame_labels).view(-1).numpy()

    all_video_probs = torch.cat(all_video_probs).view(-1).numpy()
    all_video_labels = torch.cat(all_video_labels).view(-1).numpy()

    print("Val frame label distribution:",np.unique(all_frame_labels, return_counts=True))

    print("Val video label distribution:",np.unique(all_video_labels, return_counts=True))
    # 
    
    frame_auc = safe_auc(all_frame_labels, all_frame_probs)
    video_auc = safe_auc(all_video_labels, all_video_probs)

    return (
        avg_loss,
        frame_acc,
        frame_auc,
        video_acc,
        video_auc
    )


# =========================================================
# MAIN
# =========================================================
def main():

    args = get_args()
    print_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------
    # W&B INIT
    # -------------------------------------------------
    wandb.init(
        project="AV_DF",
        name=args.name,
        config=vars(args)
    )

    # -------------------------------------------------
    # DATA
    # -------------------------------------------------
    # breakpoint()
    train_dataset = AVFeatureDataset(
        os.path.join(args.metadata_root_path, "train_metadata_with_synth_fake_segments.csv"),
        os.path.join(args.data_root_path, "train"),
        T_train=args.T_train
    )

    val_dataset = AVFeatureDataset(
        os.path.join(args.metadata_root_path, "test_metadata_supervised.csv"),
        os.path.join(args.data_val_root_path, "test"),
        T_train=args.T_train
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------
    # model = TemporalFusionModel(
    #     input_dim=args.feature_dim,
    #     hidden_dim=args.hidden_dim,
    #     num_layers=args.num_layers,
    #     nhead=args.nhead
    # ).to(device)
    model = SimpleTemporalFusionAV_only(
        input_dim=args.feature_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    wandb.watch(model, log="gradients", log_freq=100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.scheduler_patience
    )

    best_val_video_auc = 0
    epochs_without_improvement = 0

    # -------------------------------------------------
    # TRAIN LOOP
    # -------------------------------------------------
    for epoch in range(args.epochs):
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        (
            train_loss,
            train_frame_acc,
            train_frame_auc,
            train_video_acc,
            train_video_auc
        ) = run_epoch(
            train_loader,
            model,
            device,
            optimizer=optimizer,
            use_tqdm=args.use_tqdm
        )

        (
            val_loss,
            val_frame_acc,
            val_frame_auc,
            val_video_acc,
            val_video_auc
        ) = run_epoch(
            val_loader,
            model,
            device,
            optimizer=None,
            use_tqdm=args.use_tqdm
        )

        print(
            f"Train | Loss: {train_loss:.4f} "
            f"| Frame AUC: {train_frame_auc:.4f} "
            f"| Video AUC: {train_video_auc:.4f}"
        )

        print(
            f"Val   | Loss: {val_loss:.4f} "
            f"| Frame AUC: {val_frame_auc:.4f} "
            f"| Video AUC: {val_video_auc:.4f}"
        )

        # -------------------------------------------------
        # W&B LOG
        # -------------------------------------------------
        wandb.log({
            "epoch": epoch + 1,

            "train_loss": train_loss,
            "val_loss": val_loss,

            "train_frame_acc": train_frame_acc,
            "val_frame_acc": val_frame_acc,

            "train_frame_auc": train_frame_auc,
            "val_frame_auc": val_frame_auc,

            "train_video_acc": train_video_acc,
            "val_video_acc": val_video_acc,

            "train_video_auc": train_video_auc,
            "val_video_auc": val_video_auc,

            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        scheduler.step(val_video_auc)

        is_best = val_video_auc > best_val_video_auc

        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_video_auc": best_val_video_auc,
            "args": args
        }

        save_path = save_checkpoint(checkpoint, is_best, args.save_path, args.name)

        if is_best:
            best_val_video_auc = val_video_auc
            epochs_without_improvement = 0
            print("New best model saved.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs.")

        if epochs_without_improvement >= args.early_stopping_patience:
            print("Early stopping triggered.")
            break

    wandb.finish()
    print("Training finished.")


if __name__ == "__main__":
    main()