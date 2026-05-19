import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile

def _find_pairs(data_dir: str) -> List[Tuple[str, str, str]]:
    if not os.path.exists(data_dir):
        return []
    
    files = os.listdir(data_dir)
    img_map = {}
    npy_map = {}
    
    for f in files:
        path = Path(f)
        stem = path.stem
        suffix = path.suffix.lower()  
        
        if suffix in {".tif", ".tiff"} and stem.endswith("_IM"):
            base_id = stem[:-3]
            img_map[base_id] = f
            
        elif suffix == ".npy" and stem.endswith("_GT"):
            base_id = stem[:-3]
            npy_map[base_id] = f
            
    common = sorted(set(img_map) & set(npy_map))

    return [(s, img_map[s], npy_map[s]) for s in common]

def split_pairs(pairs: List[Tuple], val_split: float, seed: int = 42) -> Tuple[List[Tuple], List[Tuple]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs)).tolist()
    n_val = max(1, int(len(pairs) * val_split))
    val_idx = set(indices[:n_val])
    train_idx = set(indices[n_val:])
    return [pairs[i] for i in sorted(train_idx)], [pairs[i] for i in sorted(val_idx)]

class CellSHCoeffDataset(Dataset):
    def __init__(self, root_dir: str, pairs: Optional[List[Tuple]] = None):
        self.root_dir = root_dir
        self.pairs = pairs if pairs is not None else _find_pairs(self.root_dir)
        if not self.pairs:
            raise RuntimeError(f"No matching image/npy pairs found in {root_dir}.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stem, img_fname, npy_fname = self.pairs[idx]
        
        vol = tifffile.imread(os.path.join(self.root_dir, img_fname)).astype(np.float32)
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        volume = torch.FloatTensor(vol).unsqueeze(0) 
        
        target_vector = np.load(os.path.join(self.root_dir, npy_fname)).astype(np.float32)
        target_tensor = torch.FloatTensor(target_vector)
        
        return volume, target_tensor, stem

def make_splits(train_dir: str, val_split: float = 0.2, seed: int = 42) -> Tuple[CellSHCoeffDataset, CellSHCoeffDataset]:
    all_pairs = _find_pairs(train_dir)
    if not all_pairs:
        raise RuntimeError(f"No matched pairs found in {train_dir}.")
    train_pairs, val_pairs = split_pairs(all_pairs, val_split=val_split, seed=seed)
    return CellSHCoeffDataset(train_dir, pairs=train_pairs), CellSHCoeffDataset(train_dir, pairs=val_pairs)