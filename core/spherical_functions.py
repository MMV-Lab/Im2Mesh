import pyshtools
import numpy as np
from scipy.interpolate import griddata
from vedo import spher2cart, mag, Box, Point, Points, Plotter, Mesh
import torch

def coefficients_to_gt_vector(coeffs_array: np.ndarray, center_of_mass: list = None) -> np.ndarray:
    """
    Flattens a spherical harmonics coefficient array and prepends the center of mass.
    """
    if center_of_mass is not None:
        gt_list = [float(center_of_mass[0]), float(center_of_mass[1]), float(center_of_mass[2])]
    else:
        gt_list = []

    lmax = coeffs_array.shape[1] - 1

    for l in range(lmax + 1):
        gt_list.append(coeffs_array[0, l, 0])
        for m in range(1, l + 1):
            gt_list.append(coeffs_array[0, l, m])
            gt_list.append(coeffs_array[1, l, m])

    return np.array(gt_list, dtype=np.float32)


def gt_vector_to_coefficients(vector: np.ndarray, centers: bool = True):
    """
    Reconstructs the center of mass and the SH coefficient array from a 1D vector.
    """
    if centers:
        center_of_mass = vector[0:3]
        idx = 3
        sh_length = len(vector) - idx
    else:
        center_of_mass = None
        idx = 0
        sh_length = len(vector)

    if sh_length is None or not np.isfinite(sh_length) or sh_length <= 0:
        return None , None        
    else:
        # 2. Ejecución normal
        lmax = int(np.sqrt(sh_length)) - 1
        coeffs_reconstructed = np.zeros((2, lmax + 1, lmax + 1), dtype=np.float32)

        for l in range(lmax + 1):
            coeffs_reconstructed[0, l, 0] = vector[idx]
            idx += 1
            for m in range(1, l + 1):
                coeffs_reconstructed[0, l, m] = vector[idx]
                coeffs_reconstructed[1, l, m] = vector[idx + 1]
                idx += 2

        if center_of_mass is not None:
            return center_of_mass, coeffs_reconstructed
        else:
            return coeffs_reconstructed
   




def _make_icosphere(subdivisions: int = 3):
    """
    Return (vertices, faces) for a unit icosphere at the given subdivision level.
 
    Subdivision counts:
        0 → 12 verts / 20 faces
        1 → 42 verts / 80 faces
        2 → 162 verts / 320 faces
        3 → 642 verts / 1 280 faces   ← default, good balance
        4 → 2 562 verts / 5 120 faces
    """
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
                m /= np.linalg.norm(m)          # project back onto unit sphere
                _em[key] = len(_vm)
                _vm.append(m.tolist())
            return _em[key]
 
        for a, b, c in faces:
            ab, bc, ca = get_mid(a, b), get_mid(b, c), get_mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
 
        verts, faces = np.array(vlist), new_faces
 
    return verts, np.array(faces, dtype=np.int32)


def vector_to_points(SH_coef, x0, lmax=None, n_fine=600, subdivisions= None, mode ='icosphera'):
    """
    Converts SH coefficients to a structured point grid on the surface.
    Returns the point list and the grid dimensions (n_lat, n_lon) needed
    for topology-aware mesh construction.
    """

    if mode == 'grid':
        if lmax is None:
            lmax = SH_coef.shape[1] - 1

        reconstructed_coeffs_obj = pyshtools.SHCoeffs.from_array(SH_coef)
        grid_reco = reconstructed_coeffs_obj.expand(lmax=lmax).to_array()

        ll = []
        for i, long in enumerate(np.linspace(0, 360, num=grid_reco.shape[1], endpoint=False)):
            for j, lat in enumerate(np.linspace(90, -90, num=grid_reco.shape[0], endpoint=True)):
                ll.append((lat, long))

        radii = grid_reco.T.ravel()
        n_fine_c = complex(0, n_fine)
        lnmin, lnmax_vals = np.array(ll).min(axis=0), np.array(ll).max(axis=0)

        grid_m = np.mgrid[lnmax_vals[0]:lnmin[0]:n_fine_c, lnmin[1]:lnmax_vals[1]:n_fine_c]
        grid_x, grid_y = grid_m
        grid_reco_finer = griddata(ll, radii, (grid_x, grid_y), method='cubic')

        n_lat = grid_reco_finer.shape[0]
        n_lon = grid_reco_finer.shape[1]

        lats = np.linspace(90, -90, num=n_lat, endpoint=True)
        longs = np.linspace(0, 360, num=n_lon, endpoint=False)

        pts2 = []
        for i, long_val in enumerate(longs):
            for j, lat_val in enumerate(lats):
                th = np.deg2rad(90 - lat_val)
                ph = np.deg2rad(long_val)
                r = grid_reco_finer[j, i]
                r = max(0, r) if not np.isnan(r) else 0.0
                p = spher2cart(r, th, ph)
                pts2.append(p + x0)

        return pts2, n_lat, n_lon
    else:
        if lmax is None:
            lmax = SH_coef.shape[1] - 1

        if subdivisions is None:
            # ~(2·lmax)² vertices needed; icosphere at level k has 10·4^k + 2 vertices
            n_verts_needed = max(400, (2 * lmax) ** 2)
            subdivisions = int(np.ceil(np.log(n_verts_needed / 12) / np.log(4)))
            subdivisions = np.clip(subdivisions, 2, 6)

        unit_verts, faces = _make_icosphere(subdivisions)
    
        # pyshtools convention: geographic latitude [-90, 90], longitude [0, 360)
        lat = np.degrees(np.arcsin(np.clip(unit_verts[:, 2], -1.0, 1.0)))
        lon = np.degrees(np.arctan2(unit_verts[:, 1], unit_verts[:, 0])) % 360.0
    
        coeffs_obj = pyshtools.SHCoeffs.from_array(SH_coef.astype(np.float64))
        r_vals = np.asarray(coeffs_obj.expand(lat=lat, lon=lon, lmax_calc=lmax))
        r_vals = np.maximum(r_vals, 0.0)                       # clamp negative radii
    
        pts = (unit_verts * r_vals[:, np.newaxis] + np.asarray(x0, dtype=np.float64)).tolist()
        return pts, faces

def reconstruct_scene(
    cells: list,
    subdivisions: int = None,
    padding: int = 5,
    mode: str = 'icosphera',
) -> Mesh:
    """
    Reconstruct a full scene from per-cell SH decompositions.
 
    Args:
        cells           : list of dicts, each with:
                            'sh_vector' – 1-D ndarray with CoM prepended (x, y, z)
                            'bbox'      – (x1, y1, x2, y2) in scene pixel coords
        subdivisions    : icosphere subdivision level (None = auto from lmax)
        smoothing_iters : Laplacian smoothing iterations per mesh (0 to skip)
        padding         : same padding used during single-cell cropping
    Returns:
        vedo Mesh combining all cells in scene space.
    """
    all_verts, all_faces = [], []
    vert_offset = 0
 
    for idx, cell in enumerate(cells):
        local_com, sh_coeffs = gt_vector_to_coefficients(np.asarray(cell['sh_vector']))

        if local_com is None and sh_coeffs is None:
            continue

        x1, y1, x2, y2 = cell['bbox']
        
        # Ensure we compute the exact same crop origin used in predictions_process.py
        x_start = max(0, int(x1) - padding)
        y_start = max(0, int(y1) - padding)
 
        # Correctly map Numpy (Z,Y,X) offsets to Vedo (X,Y,Z) coordinates:
        world_com = np.array([
            local_com[0],
            local_com[1] + y_start,
            local_com[2] + x_start,
        ], dtype=np.float64)

        if mode == 'grid':
            pts2, n_lat, n_lon = vector_to_points(SH_coef=sh_coeffs,x0=world_com,mode=mode)
            mesh = points_to_mesh(pts_list=pts2, n_lat=n_lat,n_lon=n_lon,mode=mode)
        else:
            pts, faces = vector_to_points(SH_coef=sh_coeffs,x0=world_com,mode=mode)
            mesh = points_to_mesh(pts_list=pts,faces=faces,mode=mode)
    
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        tris  = np.asarray(mesh.cells,    dtype=np.int32)
 
        all_verts.append(verts)
        all_faces.append(tris + vert_offset)
        vert_offset += len(verts)
    if len(all_verts)>0 and len(all_faces)>0:
        scene_mesh = Mesh([np.vstack(all_verts), np.vstack(all_faces)])
        scene_mesh.compute_normals()
    else:
       scene_mesh = Mesh([[], []])
    return scene_mesh

class LaplacianSmooth:

    def __init__(self, aggr='mean'):
        self.aggr = aggr

    def __call__(self, x: torch.Tensor, adj_matrix: torch.sparse_coo_tensor, lambd=0.5) -> torch.Tensor:
        neighbor_sum = torch.sparse.mm(adj_matrix, x)
        
        if self.aggr == 'mean':
            degrees = torch.sparse.sum(adj_matrix, dim=1).to_dense().view(-1, 1)
            neighbor_avg = neighbor_sum / degrees.clamp(min=1)
            out = (1 - lambd) * x + lambd * neighbor_avg
        else:
            out = (1 - lambd) * x + lambd * neighbor_sum
            
        return out

def points_to_mesh(pts_list, n_lat=None, n_lon=None, faces=None, smoothing_iters=2, mode='icosphera'):
    pts = np.array(pts_list)
    
    if mode == 'grid':
        mesh_faces = []
        for i in range(n_lon):
            i_next = (i + 1) % n_lon
            for j in range(n_lat - 1):
                v0 = i      * n_lat + j
                v1 = i      * n_lat + (j + 1)
                v2 = i_next * n_lat + j
                v3 = i_next * n_lat + (j + 1)
                mesh_faces.append([v0, v1, v2])
                mesh_faces.append([v1, v3, v2])
    else:
        mesh_faces = np.array(faces)
    temp_mesh = Mesh([pts, mesh_faces]).clean().fill_holes(size=10000)
    
    clean_pts = temp_mesh.vertices
    clean_faces = temp_mesh.cells  
    num_verts = len(clean_pts)
    if smoothing_iters > 0:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.tensor(clean_pts, dtype=torch.float32, device=device)
        edges_i = []
        edges_j = []
        for f in clean_faces:
            for start, end in [(0,1), (1,2), (2,0)]:
                edges_i.append(f[start])
                edges_j.append(f[end])
                edges_i.append(f[end])
                edges_j.append(f[start])
        
        indices = torch.tensor([edges_i, edges_j], device=device)
        values = torch.ones(len(edges_i), device=device)
        adj = torch.sparse_coo_tensor(indices, values, (num_verts, num_verts)).coalesce()

        smoother = LaplacianSmooth(aggr='mean')
        for _ in range(smoothing_iters):
            x = smoother(x, adj, lambd=0.5)
            
        clean_pts = x.detach().cpu().numpy()

    final_mesh = Mesh([clean_pts, clean_faces])
    final_mesh.compute_normals()
    return final_mesh

def _make_icosphere(subdivisions: int = 3):
    """
    Return (vertices, faces) for a unit icosphere at the given subdivision level.
 
    Subdivision counts:
        0 → 12 verts / 20 faces
        1 → 42 verts / 80 faces
        2 → 162 verts / 320 faces
        3 → 642 verts / 1 280 faces   ← default, good balance
        4 → 2 562 verts / 5 120 faces
    """
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
                m /= np.linalg.norm(m)          # project back onto unit sphere
                _em[key] = len(_vm)
                _vm.append(m.tolist())
            return _em[key]
 
        for a, b, c in faces:
            ab, bc, ca = get_mid(a, b), get_mid(b, c), get_mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
 
        verts, faces = np.array(vlist), new_faces
 
    return verts, np.array(faces, dtype=np.int32)
    