import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

try:
    from scipy.special import sph_harm_y as _sph_harm_y_new
    def scipy_sph_harm(m: int, l: int, phi, theta):
        return _sph_harm_y_new(l, m, theta, phi)
except ImportError:
    from scipy.special import sph_harm as _sph_harm_old
    def scipy_sph_harm(m: int, l: int, phi, theta):
        return _sph_harm_old(m, l, phi, theta)

def _make_icosphere(subdivisions: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts[0])
 
    faces = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ]
 
    for _ in range(subdivisions):
        new_faces, edge_mid, vlist = [], {}, verts.tolist()
 
        def get_mid(a, b, _vm=vlist, _em=edge_mid):
            key = (min(a, b), max(a, b))
            if key not in _em:
                m = (np.array(_vm[a]) + np.array(_vm[b])) / 2.0
                m /= np.linalg.norm(m)
                _em[key] = len(_vm)
                _vm.append(m.tolist())
            return _em[key]
 
        for a, b, c in faces:
            ab, bc, ca = get_mid(a, b), get_mid(b, c), get_mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
 
        verts, faces = np.array(vlist), new_faces
 
    return verts.astype(np.float32), np.array(faces, dtype=np.int64)

def build_canonical_sphere(subdivisions: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    verts, faces = _make_icosphere(subdivisions=subdivisions)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True).clip(min=1e-8)
    return verts, faces

def compute_real_sh_basis(unit_verts: np.ndarray, max_degree: int) -> np.ndarray:
    x, y, z = unit_verts[:, 0], unit_verts[:, 1], unit_verts[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    cols = []
    for l in range(max_degree + 1):
        cols.append(np.real(scipy_sph_harm(0, l, phi, theta)))
        for m in range(1, l + 1):
            Y_m = scipy_sph_harm(m, l, phi, theta)
            cols.append(np.sqrt(2.0) * np.real(Y_m))
            cols.append(-np.sqrt(2.0) * np.imag(Y_m))
    return np.stack(cols, axis=1).astype(np.float32)

def faces_to_edge_index(faces: np.ndarray) -> np.ndarray:
    edges = set()
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[0], f[2])):
            edges.add((min(a, b), max(a, b)))
    E = np.array(sorted(edges), dtype=np.int64)
    return np.concatenate([E, E[:, ::-1]], axis=0).T

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
        self.proj = nn.ModuleList([
            nn.Conv3d(f, 32, 1),
            nn.Conv3d(2*f, 32, 1),
            nn.Conv3d(4*f, 32, 1),
            nn.Conv3d(8*f, 32, 1),
        ])
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
        feat_maps = [self.proj[i](s) for i, s in enumerate([s0, s1, s2, s3])]
        z = self.fc(self.global_pool(s3))
        return z, feat_maps

class SHDecoder(nn.Module):
    def __init__(self, latent_dim: int, max_degree: int, sphere_verts: np.ndarray, sphere_faces: np.ndarray):
        super().__init__()
        self.max_degree = max_degree
        n_harm = (max_degree + 1) ** 2
        self.n_harm = n_harm
        Y = compute_real_sh_basis(sphere_verts, max_degree)
        self.register_buffer("Y", torch.from_numpy(Y))
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim), nn.GELU(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2), nn.GELU(),
            nn.Linear(latent_dim // 2, 3 * n_harm),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        C_init, _, _, _ = np.linalg.lstsq(Y, sphere_verts, rcond=None)
        bias_init = np.concatenate([C_init[:, 0], C_init[:, 1], C_init[:, 2]]).astype(np.float32)
        self.net[-1].bias = nn.Parameter(torch.from_numpy(bias_init))

    def forward(self, z: torch.Tensor):
        coeffs = self.net(z).view(-1, 3, self.n_harm)
        verts = torch.einsum("bci,ni->bnc", coeffs, self.Y)
        return verts, coeffs

class _GraphConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.LayerNorm(out_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        src, dst = edge_index[0], edge_index[1]
        
        nbr = x[:, src, :]
        agg = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)
        
        dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, C)
        agg.scatter_add_(1, dst_expanded, nbr)
        
        cnt = torch.zeros(1, N, 1, device=x.device, dtype=x.dtype)
        cnt.scatter_add_(1, dst.unsqueeze(0).unsqueeze(-1), torch.ones(1, src.shape[0], 1, device=x.device))
        
        agg = agg / cnt.clamp(min=1.0)
        return self.mlp(torch.cat([x, agg], dim=-1))

class GraphRefiner(nn.Module):
    def __init__(self, n_img_feat: int = 128, hidden_dim: int = 64, n_layers: int = 3, n_steps: int = 3):
        super().__init__()
        self.n_steps = n_steps
        self.input_proj = nn.Sequential(
            nn.Linear(n_img_feat + 3, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.convs = nn.ModuleList([_GraphConvLayer(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.update_head = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.update_head.weight)
        nn.init.zeros_(self.update_head.bias)

    @staticmethod
    def _sample_feats(feat_maps: List[torch.Tensor], verts: torch.Tensor) -> torch.Tensor:
        grid = verts.unsqueeze(2).unsqueeze(3)
        sampled = []
        for fm in feat_maps:
            s = F.grid_sample(fm, grid, mode="bilinear", align_corners=True, padding_mode="border")
            s = s.squeeze(-1).squeeze(-1).transpose(1, 2)
            sampled.append(s)
        return torch.cat(sampled, dim=-1)

    def forward(self, v: torch.Tensor, feat_maps: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_steps):
            img_f = self._sample_feats(feat_maps, v)
            x = self.input_proj(torch.cat([v, img_f], dim=-1))
            for conv in self.convs:
                x = x + conv(x, edge_index)
            v = v + self.update_head(x)
        return v

def laplacian_smooth(v: torch.Tensor, edge_index: torch.Tensor, n: int = 1, lambd: float = 0.5) -> torch.Tensor:
    B, N, _ = v.shape
    src = edge_index[0]
    dst = edge_index[1]
    for _ in range(n):
        nbr_v = v[:, src, :]
        agg = torch.zeros_like(v)
        cnt = torch.zeros(1, N, 1, device=v.device, dtype=v.dtype)
        idx = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, 3)
        agg.scatter_add_(1, idx, nbr_v)
        cnt.scatter_add_(1, dst.unsqueeze(0).unsqueeze(-1), torch.ones(1, src.shape[0], 1, device=v.device))
        v = (1.0 - lambd) * v + lambd * agg / cnt.clamp(min=1.0)
    return v

class HarmoMeshNet(nn.Module):
    def __init__(self, in_ch: int = 1, base_f: int = 32, latent_dim: int = 512, max_sh_degree: int = 8, sphere_subdivisions: int = 4, refiner_steps: int = 3, refiner_layers: int = 3, refiner_hidden: int = 64):
        super().__init__()
        sphere_verts, sphere_faces = build_canonical_sphere(sphere_subdivisions)
        edge_index = faces_to_edge_index(sphere_faces)
        self.register_buffer("sphere_faces", torch.from_numpy(sphere_faces))
        self.register_buffer("edge_index", torch.from_numpy(edge_index))
        self.encoder = ImageEncoder3D(in_ch, base_f, latent_dim)
        self.sh_decoder = SHDecoder(latent_dim, max_sh_degree, sphere_verts, sphere_faces)
        self.refiner = GraphRefiner(n_img_feat=4 * 32, hidden_dim=refiner_hidden, n_layers=refiner_layers, n_steps=refiner_steps)
        self.last_sh_coeffs: torch.Tensor = None

    def forward(self, volume: torch.Tensor, n_smooth: int = 1, lambd: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        B = volume.shape[0]
        z, feat_maps = self.encoder(volume)
        v, coeffs = self.sh_decoder(z)
        self.last_sh_coeffs = coeffs
        v = self.refiner(v, feat_maps, self.edge_index)
        if n_smooth > 0:
            v = laplacian_smooth(v, self.edge_index, n=n_smooth, lambd=lambd)
        faces = self.sphere_faces.unsqueeze(0).expand(B, -1, -1)
        return v, faces