import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

class _ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(ch), nn.GELU(),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))

class ImageEncoder3D(nn.Module):
    def __init__(self, in_ch: int = 1, base_f: int = 32, latent_dim: int = 512):
        super().__init__()
        f = base_f
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, f, 3, padding=1, bias=False),
            nn.InstanceNorm3d(f), nn.GELU(),
        )
        self.down1 = nn.Sequential(
            _ResBlock3D(f),
            nn.Conv3d(f, 2*f, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(2*f), nn.GELU(),
        )
        self.down2 = nn.Sequential(
            _ResBlock3D(2*f),
            nn.Conv3d(2*f, 4*f, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(4*f), nn.GELU(),
        )
        self.down3 = nn.Sequential(
            _ResBlock3D(4*f),
            nn.Conv3d(4*f, 8*f, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(8*f), nn.GELU(),
        )
        self.global_pool = nn.AdaptiveAvgPool3d(4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*f*64, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        z = self.fc(self.global_pool(s3))
        return z

class HarmonicOscillatorLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1.0 / self.linear.in_features, 1.0 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6.0 / self.linear.in_features) / self.omega_0, 
                                             np.sqrt(6.0 / self.linear.in_features) / self.omega_0)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))

class HarmonicRegressionNet(nn.Module):
    def __init__(self, in_ch: int = 1, base_f: int = 32, latent_dim: int = 512, target_dim: int = 120):
        super().__init__()
        self.encoder = ImageEncoder3D(in_ch, base_f, latent_dim)
        
        self.harmonic_block = nn.Sequential(
            HarmonicOscillatorLayer(latent_dim, latent_dim, omega_0=30.0, is_first=True),
            HarmonicOscillatorLayer(latent_dim, latent_dim // 2, omega_0=30.0),
            HarmonicOscillatorLayer(latent_dim // 2, latent_dim // 2, omega_0=30.0),
        )
        
        self.regressor_head = nn.Linear(latent_dim // 2, target_dim)
        nn.init.xavier_uniform_(self.regressor_head.weight)
        nn.init.zeros_(self.regressor_head.bias)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        z = self.encoder(volume)
        harmonics_feat = self.harmonic_block(z)
        pred_coeffs = self.regressor_head(harmonics_feat)
        return pred_coeffs