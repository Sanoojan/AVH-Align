import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModel(nn.Module):
    def __init__(self, visual_dim=1024, audio_dim=1024, hidden_dim=1024):
        super(FusionModel, self).__init__()
        
        self.visual_proj = nn.Linear(visual_dim, hidden_dim // 2)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim // 2)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, visual_features, audio_features):
        # Project visual and audio features separately and concatenate
        visual_proj = self.visual_proj(visual_features)
        audio_proj = self.audio_proj(audio_features)
        fused_features = torch.cat((visual_proj, audio_proj), dim=-1)
        
        output = self.mlp(fused_features)
            
        return output


class TemporalFusionModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=4, nhead=8):
        super().__init__()

        self.input_proj = nn.Linear(input_dim * 3, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, 3, T, D)

        B, C, T, D = x.shape

        # reshape to (B, T, 3D)
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * D)

        x = self.input_proj(x)

        x = self.transformer(x)

        logits = self.classifier(x).squeeze(-1)  # (B, T)

        return logits
    
class SimpleTemporalFusion(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=768):
        super().__init__()

        self.fc1 = nn.Linear(input_dim * 3, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, 3, T, D)
        B, C, T, D = x.shape

        # (B, T, 3D)
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * D)

        # Apply MLP per frame
        x = self.fc1(x)
        x = self.act(x)
        logits = self.fc2(x).squeeze(-1)  # (B, T)

        return logits
    
class SimpleTemporalFusionAV_only(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=768):
        super().__init__()

        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, 2, T, D)
        # if x is a tuple of (visual, audio), concatenate along the channel dimension
        
        
        B, C, T, D = x.shape

        # (B, T, 2D)
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * D)

        # Apply MLP per frame
        x = self.fc1(x)
        x = self.act(x)
        logits = self.fc2(x).squeeze(-1)  # (B, T)

        return logits

class ResidualConv1DBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ConvBoundaryTemporalFusionAV_only(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=512,
        num_layers=4,
        kernel_size=5,
        dropout=0.1,
    ):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even, got {hidden_dim}")

        self.visual_proj = nn.Linear(input_dim, hidden_dim // 2)
        self.audio_proj = nn.Linear(input_dim, hidden_dim // 2)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_act = nn.GELU()
        self.input_dropout = nn.Dropout(dropout)

        dilations = [2 ** (idx % 4) for idx in range(num_layers)]
        self.temporal = nn.Sequential(
            *[
                ResidualConv1DBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )

        self.coarse_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.refine_head = nn.Sequential(
            nn.Conv1d(hidden_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )
        self.boundary_head = nn.Conv1d(hidden_dim, 2, kernel_size=1)
        self.offset_head = nn.Conv1d(hidden_dim, 2, kernel_size=1)

    def forward(self, x):
        # x: (B, 2, T, D), channel order is visual then audio.
        if x.ndim != 4 or x.shape[1] < 2:
            raise ValueError(f"Expected input shape (B, 2, T, D), got {tuple(x.shape)}")

        visual = x[:, 0]
        audio = x[:, 1]
        fused = torch.cat([self.visual_proj(visual), self.audio_proj(audio)], dim=-1)
        fused = self.input_dropout(self.input_act(self.input_norm(fused)))
        features = fused.transpose(1, 2)
        features = self.temporal(features)

        coarse_logits = self.coarse_head(features).squeeze(1)
        refined_input = torch.cat([features, coarse_logits.unsqueeze(1)], dim=1)
        frame_logits = self.refine_head(refined_input).squeeze(1)
        boundary_logits = self.boundary_head(features).transpose(1, 2)
        offsets = torch.nn.functional.softplus(self.offset_head(features).transpose(1, 2))

        return {
            "frame_logits": frame_logits,
            "coarse_logits": coarse_logits,
            "boundary_logits": boundary_logits,
            "offsets": offsets,
        }






class ResidualDilatedConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class BoundaryAwareAVLocalizer(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=512,
        num_layers=6,
        kernel_size=5,
        dropout=0.1,
    ):
        super().__init__()

        self.visual_proj = nn.Linear(input_dim, hidden_dim // 2)
        self.audio_proj = nn.Linear(input_dim, hidden_dim // 2)

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        dilations = [1, 2, 4, 8, 1, 2]

        self.temporal = nn.Sequential(
            *[
                ResidualDilatedConvBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilations[i % len(dilations)],
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

        self.fake_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.start_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.end_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        # x: [B, 2, T, D]
        visual = x[:, 0]  # [B,T,D]
        audio = x[:, 1]   # [B,T,D]

        visual = self.visual_proj(visual)
        audio = self.audio_proj(audio)

        feat = torch.cat([visual, audio], dim=-1)  # [B,T,H]
        feat = self.input_norm(feat)
        feat = F.gelu(feat)
        feat = self.input_dropout(feat)

        feat = feat.transpose(1, 2)  # [B,H,T]
        feat = self.temporal(feat)

        return {
            "fake_logits": self.fake_head(feat).squeeze(1),   # [B,T]
            "start_logits": self.start_head(feat).squeeze(1), # [B,T]
            "end_logits": self.end_head(feat).squeeze(1),     # [B,T]
        }


class AuvireAVDeepfake1MLocalizer(nn.Module):
    """AUVIRE AVDeepfake1M architecture adapted to train_sup_new outputs."""

    def __init__(
        self,
        input_dim=1024,
        auvire_dim=768,
        max_length=512,
        d_model=128,
        win_size=15,
        num_heads=8,
        operation="subtraction",
        reconstruction=None,
        encoder=None,
        dropout=None,
        conv=None,
        model_type=None,
        device="cpu",
        return_aux=True,
    ):
        super().__init__()
        if input_dim not in (auvire_dim, 1024):
            raise ValueError(
                "AuvireAVDeepfake1MLocalizer supports native AUVIRE features "
                f"({auvire_dim}) or 1024-dim features with projection, got {input_dim}."
            )

        project_root = Path(__file__).resolve().parent
        auvire_root = project_root / "auvire"
        if str(auvire_root) not in sys.path:
            sys.path.insert(0, str(auvire_root))

        from src.models import Model as AuvireModel

        reconstruction = reconstruction or {
            "nlayers": {"pre": 2, "downsample": 1, "upsample": 1, "post": 2},
            "modality": ["av", "aa", "vv"],
        }
        encoder = encoder or {"nlayers": {"retain": 1, "downsample": 1}, "fpn": True}
        dropout = dropout or {"main": 0.1, "head": 0.5}
        conv = conv or {"use_ln": True, "use_rl": True, "use_do": False}
        model_type = model_type or {"reconstruction": "cnn", "encoder": "cnn"}
        factor = [1] * encoder["nlayers"]["retain"] + [
            2 ** (idx + 1) for idx in range(encoder["nlayers"]["downsample"])
        ]

        self.max_length = max_length
        self.factor = factor
        self.input_dim = input_dim
        self.auvire_dim = auvire_dim
        self.return_aux = return_aux
        self.visual_projection = nn.Identity() if input_dim == auvire_dim else nn.Linear(input_dim, auvire_dim)
        self.audio_projection = nn.Identity() if input_dim == auvire_dim else nn.Linear(input_dim, auvire_dim)
        self.auvire_model = AuvireModel(
            max_length=max_length,
            d_model=d_model,
            win_size=win_size,
            num_heads=num_heads,
            operation=operation,
            reconstruction=reconstruction,
            encoder=encoder,
            dropout=dropout,
            use_ln=conv["use_ln"],
            use_rl=conv["use_rl"],
            use_do=conv["use_do"],
            model_type=model_type,
            factor=factor,
            device=device,
        )

    def _fit_temporal_length(self, x):
        original_length = x.shape[2]
        if original_length == 0:
            raise ValueError("Expected a non-empty temporal dimension for AUVIRE input.")
        if original_length == self.max_length:
            return x, original_length
        if original_length > self.max_length:
            return x[:, :, : self.max_length], self.max_length

        repeat_count = (self.max_length + original_length - 1) // original_length
        x = x.repeat(1, 1, repeat_count, 1)[:, :, : self.max_length]
        return x, original_length

    def forward(self, x):
        # x: [B, 2, T, D], channel order is visual then audio.
        if x.ndim != 4 or x.shape[1] < 2:
            raise ValueError(f"Expected input shape [B, 2, T, D], got {tuple(x.shape)}")
        x, output_length = self._fit_temporal_length(x)

        visual = self.visual_projection(x[:, 0])
        audio = self.audio_projection(x[:, 1])
        outputs, dissimilarity = self.auvire_model([visual, audio])
        highest_resolution = outputs[0][:, :output_length]

        result = {
            "frame_logits": highest_resolution[:, :, 0],
            "boundary_logits": highest_resolution[:, :, 1:3],
            "offsets": F.softplus(highest_resolution[:, :, 1:3]),
        }
        if self.return_aux:
            result["auvire_outputs"] = outputs
            result["dissimilarity"] = dissimilarity
        return result

