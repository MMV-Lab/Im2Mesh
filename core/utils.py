import numpy as np
import torch
import vedo
from scipy.spatial import cKDTree

def compute_normal(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    B, N, _ = v.shape
    f_ = f[0]
    v0 = v[:, f_[:, 0], :]
    v1 = v[:, f_[:, 1], :]
    v2 = v[:, f_[:, 2], :]
    normals = torch.cross(v1 - v0, v2 - v0, dim=2)
    vertex_normals = torch.zeros_like(v)
    vertex_normals.scatter_add_(1, f_[:, 0:1].unsqueeze(0).expand(B, -1, 3), normals)
    vertex_normals.scatter_add_(1, f_[:, 1:2].unsqueeze(0).expand(B, -1, 3), normals)
    vertex_normals.scatter_add_(1, f_[:, 2:3].unsqueeze(0).expand(B, -1, 3), normals)
    return torch.nn.functional.normalize(vertex_normals, p=2, dim=2)

def save_mesh_vedo(v: np.ndarray, f: np.ndarray, n: np.ndarray, path: str) -> None:
    mesh = vedo.Mesh([v, f])
    mesh.pointdata["Normals"] = n
    vedo.write(mesh, path)

def sample_points_numpy(v: np.ndarray, f: np.ndarray, num_samples: int) -> np.ndarray:
    """Samples points on the surface of a given mesh weighted by triangle areas."""
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) * 0.5
    
    areas_sum = np.sum(areas)
    probs = areas / areas_sum if areas_sum > 1e-8 else np.ones_like(areas) / len(areas)
    
    face_idx = np.random.choice(len(f), size=num_samples, p=probs)
    
    r1 = np.random.rand(num_samples, 1)
    r2 = np.random.rand(num_samples, 1)
    
    sqrt_r1 = np.sqrt(r1)
    w0 = 1.0 - sqrt_r1
    w1 = sqrt_r1 * (1.0 - r2)
    w2 = sqrt_r1 * r2
    
    sv0, sv1, sv2 = v[f[face_idx, 0]], v[f[face_idx, 1]], v[f[face_idx, 2]]
    
    return w0 * sv0 + w1 * sv1 + w2 * sv2

def compute_distance(v_pred, v_gt, f_pred, f_gt, n_samples=150000):
    cd = 0.0
    
    # 1. Point-to-point Chamfer Distance evaluation
    kd_pred = cKDTree(v_pred)
    cd += kd_pred.query(v_gt)[0].mean() / 2
    kd_gt = cKDTree(v_gt)
    cd += kd_gt.query(v_pred)[0].mean() / 2
    
    # 2. Dense Surface Distance calculation using sampled points
    pts_pred = sample_points_numpy(v_pred, f_pred, n_samples)
    pts_gt = sample_points_numpy(v_gt, f_gt, n_samples)
    
    kd_pred_pts = cKDTree(pts_pred)
    d_g2p, _ = kd_pred_pts.query(pts_gt)
    
    kd_gt_pts = cKDTree(pts_gt)
    d_p2g, _ = kd_gt_pts.query(pts_pred)
    
    assd = (np.sum(d_p2g) + np.sum(d_g2p)) / float(len(d_p2g) + len(d_g2p))
    hd90 = max(np.percentile(d_p2g, 90), np.percentile(d_g2p, 90))
    
    return cd, assd, hd90

def load_sh_vector(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)

def evaluate_vector_error(pred: np.ndarray, gt: np.ndarray) -> dict:
    mse = np.mean((pred - gt) ** 2)
    mae = np.mean(np.abs(pred - gt))
    
    norm_pred = pred / (np.linalg.norm(pred) + 1e-8)
    norm_gt = gt / (np.linalg.norm(gt) + 1e-8)
    cosine_sim = np.dot(norm_pred, norm_gt)
    
    return {
        "eval_mse": float(mse),
        "eval_mae": float(mae),
        "eval_cosine_similarity": float(cosine_sim)
    }