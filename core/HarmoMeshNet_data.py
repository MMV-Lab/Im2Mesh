import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
import vedo

MESH_EXTS = {".obj", ".stl", ".ply", ".vtk"}

def _find_pairs(image_dir: str, mesh_dir: str) -> List[Tuple[str, str, str]]:
    img_map = {Path(f).stem: f for f in os.listdir(image_dir) if Path(f).suffix.lower() in {".tif", ".tiff"}}
    mesh_map = {Path(f).stem: f for f in os.listdir(mesh_dir) if Path(f).suffix.lower() in MESH_EXTS}
    common = sorted(set(img_map) & set(mesh_map))
    return [(s, img_map[s], mesh_map[s]) for s in common]

def split_pairs(pairs: List[Tuple], val_split: float, seed: int = 42) -> Tuple[List[Tuple], List[Tuple]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs)).tolist()
    n_val = max(1, int(len(pairs) * val_split))
    val_idx = set(indices[:n_val])
    train_idx = set(indices[n_val:])
    return [pairs[i] for i in sorted(train_idx)], [pairs[i] for i in sorted(val_idx)]

def normalize_vertices(v: torch.Tensor, vol_shape: tuple) -> torch.Tensor:
    Z, Y, X = vol_shape
    center = torch.tensor([X / 2.0, Y / 2.0, Z / 2.0], dtype=torch.float32, device=v.device)
    scale = torch.tensor([X / 2.0, Y / 2.0, Z / 2.0], dtype=torch.float32, device=v.device)
    return (v - center) / scale.clamp(min=1e-8)

class CellMeshDataset(Dataset):
    def __init__(self, root_dir: str, pairs: Optional[List[Tuple]] = None):
        self.image_dir = os.path.join(root_dir, "images")
        self.mesh_dir = os.path.join(root_dir, "meshes")
        self.pairs = pairs if pairs is not None else _find_pairs(self.image_dir, self.mesh_dir)
        if not self.pairs:
            raise RuntimeError(f"No matching image/mesh pairs found in {root_dir}.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stem, img_fname, mesh_fname = self.pairs[idx]
        vol = tifffile.imread(os.path.join(self.image_dir, img_fname)).astype(np.float32)
        vol_shape = vol.shape
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        volume = torch.FloatTensor(vol).unsqueeze(0)
        
        mesh = vedo.load(os.path.join(self.mesh_dir, mesh_fname))
        mesh.triangulate()
        verts = torch.FloatTensor(np.asarray(mesh.points, dtype=np.float32))
        faces = torch.LongTensor(np.asarray(mesh.cells, dtype=np.int64))
        
        verts = normalize_vertices(verts, vol_shape)
        
        return volume, verts, faces, stem

def make_splits(train_dir: str, val_split: float = 0.2, seed: int = 42) -> Tuple[CellMeshDataset, CellMeshDataset]:
    image_dir = os.path.join(train_dir, "images")
    mesh_dir = os.path.join(train_dir, "meshes")
    all_pairs = _find_pairs(image_dir, mesh_dir)
    if not all_pairs:
        raise RuntimeError(f"No matched pairs found in {train_dir}.")
    train_pairs, val_pairs = split_pairs(all_pairs, val_split=val_split, seed=seed)
    return CellMeshDataset(train_dir, pairs=train_pairs), CellMeshDataset(train_dir, pairs=val_pairs)