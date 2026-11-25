
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Time embedding
# -----------------------------

def time_embedding(t, dim=128):
    half = dim // 2

    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(
            half,
            device=t.device
        ).float() / half
    )

    angles = t.float()[:, None] * frequencies[None, :]

    return torch.cat(
        [torch.sin(angles), torch.cos(angles)],
        dim=-1
    )


# -----------------------------
# Text encoder
# -----------------------------

class TextEncoder(nn.Module):

    def __init__(self, vocab_size, dim=64):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            dim,
            padding_idx=0
        )

        self.network = nn.Sequential(
            nn.Linear(dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

    def forward(self, tokens):

        embedded = self.embedding(tokens)

        mask = (tokens != 0).float().unsqueeze(-1)

        pooled = (
            (embedded * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        return self.network(pooled)


# -----------------------------
# Residual block
# -----------------------------

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, cond_dim=256):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1
        )

        self.norm2 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1
        )

        self.condition = nn.Linear(
            cond_dim, out_channels
        )

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x, condition):

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = h + self.condition(condition)[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


# -----------------------------
# Conditional UNet
# -----------------------------

class ConditionalUNet(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.text_encoder = TextEncoder(vocab_size)

        self.time_network = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )

        self.text_network = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )

        self.input = nn.Conv2d(3, 32, 3, padding=1)

        self.down1 = ResBlock(32, 32)
        self.downsample1 = nn.Conv2d(
            32, 64, 4, stride=2, padding=1
        )

        self.down2 = ResBlock(64, 64)
        self.downsample2 = nn.Conv2d(
            64, 128, 4, stride=2, padding=1
        )

        self.middle = ResBlock(128, 128)

        self.up1 = ResBlock(192, 64)
        self.up2 = ResBlock(96, 32)

        self.output_norm = nn.GroupNorm(8, 32)
        self.output = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x, t, tokens):

        text = self.text_encoder(tokens)

        time = self.time_network(
            time_embedding(t)
        )

        condition = (
            time + self.text_network(text)
        )

        x = self.input(x)

        skip1 = self.down1(x, condition)

        x = self.downsample1(skip1)

        skip2 = self.down2(x, condition)

        x = self.downsample2(skip2)

        x = self.middle(x, condition)

        x = F.interpolate(
            x, scale_factor=2, mode="nearest"
        )

        x = self.up1(
            torch.cat([x, skip2], dim=1),
            condition
        )

        x = F.interpolate(
            x, scale_factor=2, mode="nearest"
        )

        x = self.up2(
            torch.cat([x, skip1], dim=1),
            condition
        )

        return self.output(
            F.silu(self.output_norm(x))
        )