from typing import Dict, Tuple
import torch


def _safe_norm(x, dim=-1, eps=1e-8):
    return torch.sqrt(torch.sum(x**2, dim=dim) + eps)


def sample_points_from_meshes(v, f, num_samples):
    B, F_cnt, _ = f.shape
    v0 = v[:, f[0, :, 0], :]
    v1 = v[:, f[0, :, 1], :]
    v2 = v[:, f[0, :, 2], :]

    cross = torch.cross(v1 - v0, v2 - v0, dim=2)
    areas = _safe_norm(cross, dim=2) * 0.5

    areas_sum = areas.sum(dim=1, keepdim=True)
    probs = torch.where(areas_sum > 1e-8, areas / areas_sum, torch.ones_like(areas) / areas.shape[1])

    dist = torch.distributions.Categorical(probs)
    face_idx = dist.sample((num_samples,)).transpose(0, 1)

    w = torch.rand(B, num_samples, 3, device=v.device)
    w = -torch.log(w + 1e-8)
    w = w / w.sum(dim=-1, keepdim=True)

    b_idx = torch.arange(B).unsqueeze(-1).expand(-1, num_samples)
    sv0 = v[b_idx, f[b_idx, face_idx, 0], :]
    sv1 = v[b_idx, f[b_idx, face_idx, 1], :]
    sv2 = v[b_idx, f[b_idx, face_idx, 2], :]
    pts = w[:, :, 0:1] * sv0 + w[:, :, 1:2] * sv1 + w[:, :, 2:3] * sv2
    return pts


def chamfer_distance(p1, p2):
    p1_sq = p1.pow(2).sum(-1, keepdim=True)
    p2_sq = p2.pow(2).sum(-1).unsqueeze(1)
    dist = p1_sq + p2_sq - 2 * torch.bmm(p1, p2.transpose(1, 2))
    dist = dist.clamp(min=0.0)
    min1 = dist.min(dim=2)[0]
    min2 = dist.min(dim=1)[0]
    return min1.mean() + min2.mean(), None


def mesh_edge_loss(v, f):
    f_ = f[0]
    edges = torch.cat([f_[:, [0, 1]], f_[:, [1, 2]], f_[:, [2, 0]]], dim=0)
    edges = torch.sort(edges, dim=1)[0]
    edges = torch.unique(edges, dim=0)
    v0 = v[:, edges[:, 0], :]
    v1 = v[:, edges[:, 1], :]
    edge_lens = _safe_norm(v0 - v1, dim=-1)
    return (edge_lens - edge_lens.mean()).pow(2).mean()


def mesh_laplacian_smoothing(v, f):
    B, N, _ = v.shape
    f_ = f[0]
    edges = torch.cat([f_[:, [0, 1]], f_[:, [1, 2]], f_[:, [2, 0]],
                       f_[:, [1, 0]], f_[:, [2, 1]], f_[:, [0, 2]]], dim=0)
    src = edges[:, 0]
    dst = edges[:, 1]
    deg = torch.zeros(N, device=v.device, dtype=torch.float32)
    deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))

    nbr_sum = torch.zeros(B, N, 3, device=v.device)
    nbr_v = v[:, dst, :]
    for b in range(B):
        nbr_sum[b].scatter_add_(0, src.unsqueeze(-1).expand(-1, 3), nbr_v[b])

    lap = v - nbr_sum / deg.clamp(min=1).view(1, N, 1)
    return _safe_norm(lap, dim=-1).mean()


def mesh_normal_consistency(v, f):
    F_cnt = f.shape[1]
    f_ = f[0]
    edges = torch.cat([f_[:, [0, 1]], f_[:, [1, 2]], f_[:, [2, 0]]], dim=0)
    face_idx = torch.cat([torch.arange(F_cnt), torch.arange(F_cnt), torch.arange(F_cnt)]).to(f.device)

    edges_sorted, _ = torch.sort(edges, dim=1)
    _, inv_idx, counts = torch.unique(edges_sorted, dim=0, return_inverse=True, return_counts=True)
    mask = counts[inv_idx] == 2
    edges_inner = edges_sorted[mask]
    faces_inner = face_idx[mask]

    _, sort_idx = torch.sort(edges_inner[:, 0] * v.shape[1] + edges_inner[:, 1])
    faces_inner = faces_inner[sort_idx].view(-1, 2)

    f0 = faces_inner[:, 0]
    f1 = faces_inner[:, 1]

    v0 = v[:, f_[:, 0], :]
    v1 = v[:, f_[:, 1], :]
    v2 = v[:, f_[:, 2], :]

    cross = torch.cross(v1 - v0, v2 - v0, dim=2)
    normals = cross / _safe_norm(cross, dim=2).clamp(min=1e-8).unsqueeze(-1)

    n0 = normals[:, f0, :]
    n1 = normals[:, f1, :]
    dot = (n0 * n1).sum(dim=-1)
    return (1.0 - dot).mean()


def cotangent_weights(v, f):
    f_ = f[0]
    v0 = v[:, f_[:, 0], :]
    v1 = v[:, f_[:, 1], :]
    v2 = v[:, f_[:, 2], :]

    cross_norm_01_02 = _safe_norm(torch.cross(v1 - v0, v2 - v0, dim=-1)).clamp(min=1e-6)
    cross_norm_12_10 = _safe_norm(torch.cross(v2 - v1, v0 - v1, dim=-1)).clamp(min=1e-6)
    cross_norm_20_21 = _safe_norm(torch.cross(v0 - v2, v1 - v2, dim=-1)).clamp(min=1e-6)

    cot12 = ((v1 - v0) * (v2 - v0)).sum(dim=-1) / cross_norm_01_02
    cot20 = ((v2 - v1) * (v0 - v1)).sum(dim=-1) / cross_norm_12_10
    cot01 = ((v0 - v2) * (v1 - v2)).sum(dim=-1) / cross_norm_20_21

    # Clamp to prevent explosion from near-degenerate triangles
    cot12 = cot12.clamp(-5.0, 5.0)
    cot20 = cot20.clamp(-5.0, 5.0)
    cot01 = cot01.clamp(-5.0, 5.0)

    return cot12, cot20, cot01


def willmore_energy(v, f):
    B, N, _ = v.shape
    cot12, cot20, cot01 = cotangent_weights(v, f)
    L_v = torch.zeros(B, N, 3, device=v.device)
    f_ = f[0]
    for b in range(B):
        w12 = cot12[b].unsqueeze(-1)
        w20 = cot20[b].unsqueeze(-1)
        w01 = cot01[b].unsqueeze(-1)
        L_v[b].scatter_add_(0, f_[:, 0].unsqueeze(-1).expand(-1, 3),
                            (v[b, f_[:, 0]] - v[b, f_[:, 1]]) * w20 +
                            (v[b, f_[:, 0]] - v[b, f_[:, 2]]) * w12)
        L_v[b].scatter_add_(0, f_[:, 1].unsqueeze(-1).expand(-1, 3),
                            (v[b, f_[:, 1]] - v[b, f_[:, 2]]) * w01 +
                            (v[b, f_[:, 1]] - v[b, f_[:, 0]]) * w20)
        L_v[b].scatter_add_(0, f_[:, 2].unsqueeze(-1).expand(-1, 3),
                            (v[b, f_[:, 2]] - v[b, f_[:, 0]]) * w12 +
                            (v[b, f_[:, 2]] - v[b, f_[:, 1]]) * w01)
    # Clamp per-vertex curvature norm before squaring to prevent explosion
    curvature_norm = _safe_norm(L_v, dim=-1).clamp(max=50.0)
    return curvature_norm.pow(2).mean()


def sh_mode_energy(coeffs: torch.Tensor, max_degree: int) -> torch.Tensor:
    energy = coeffs.new_zeros(1)
    idx = 0
    for l in range(max_degree + 1):
        n = 2 * l + 1
        mode_contrib = l * (l + 1) * coeffs[:, :, idx:idx + n].pow(2).sum(dim=(-1, -2)).mean()
        energy = energy + mode_contrib
        idx += n
    return energy


def compute_losses(
    v_pred: torch.Tensor,
    f_pred: torch.Tensor,
    v_gt: torch.Tensor,
    f_gt: torch.Tensor,
    sh_coeffs: torch.Tensor,
    max_sh_degree: int,
    n_pts: int,
    scales: Dict[str, float],
    loss_cap: float = 50.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pts_pred = sample_points_from_meshes(v_pred, f_pred, n_pts)
    pts_gt = sample_points_from_meshes(v_gt, f_gt, n_pts)

    l_chamfer   = scales["chamfer"]      * chamfer_distance(pts_pred, pts_gt)[0]
    l_nc        = scales["normal"]       * mesh_normal_consistency(v_pred, f_pred)
    l_el        = scales["edge"]         * mesh_edge_loss(v_pred, f_pred)
    l_ls        = scales["laplacian"]    * mesh_laplacian_smoothing(v_pred, f_pred)
    l_willmore  = scales["willmore"]     * willmore_energy(v_pred, f_pred)
    l_mode      = scales["mode_energy"]  * sh_mode_energy(sh_coeffs, max_sh_degree)

    # Per-loss cap to prevent single outlier samples from corrupting the update
    l_chamfer   = l_chamfer.clamp(max=loss_cap)
    l_nc        = l_nc.clamp(max=loss_cap)
    l_el        = l_el.clamp(max=loss_cap)
    l_ls        = l_ls.clamp(max=loss_cap)
    l_willmore  = l_willmore.clamp(max=loss_cap)
    l_mode      = l_mode.clamp(max=loss_cap)

    total = l_chamfer + l_nc + l_el + l_ls + l_willmore + l_mode
    return total, {
        "chamfer":      l_chamfer,
        "normal":       l_nc,
        "edge":         l_el,
        "laplacian":    l_ls,
        "willmore":     l_willmore,
        "mode_energy":  l_mode,
    }