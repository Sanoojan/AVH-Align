import torch
import torch.nn.functional as F

def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def smoothness_loss_from_logits(logits):
    probs = torch.sigmoid(logits)
    return torch.mean(torch.abs(probs[:, 1:] - probs[:, :-1]))


def focal_bce_with_logits(logits, targets, alpha=0.75, gamma=2.0):
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * ((1 - pt) ** gamma) * bce
    return loss.mean()


def boundary_localization_loss(
    outputs,
    frame_labels,
    start_labels,
    end_labels,
    lambda_start=1.0,
    lambda_end=1.0,
    lambda_dice=1.0,
    lambda_smooth=0.03,
    focal_alpha=0.75,
    focal_gamma=2.0,
):
    fake_logits = outputs["fake_logits"]
    start_logits = outputs["start_logits"]
    end_logits = outputs["end_logits"]

    loss_fake = focal_bce_with_logits(fake_logits, frame_labels, alpha=focal_alpha, gamma=focal_gamma)
    loss_start = F.binary_cross_entropy_with_logits(start_logits, start_labels.float())
    loss_end = F.binary_cross_entropy_with_logits(end_logits, end_labels.float())
    loss_dice = dice_loss_from_logits(fake_logits, frame_labels)
    loss_smooth = smoothness_loss_from_logits(fake_logits)

    total = (
        loss_fake
        + lambda_start * loss_start
        + lambda_end * loss_end
        + lambda_dice * loss_dice
        + lambda_smooth * loss_smooth
    )

    return total, {
        "loss_fake": loss_fake.item(),
        "loss_start": loss_start.item(),
        "loss_end": loss_end.item(),
        "loss_dice": loss_dice.item(),
        "loss_smooth": loss_smooth.item(),
    }
    
    
    
