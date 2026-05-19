import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import os
import argparse
from pathlib import Path

import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from bioio import BioImage
import bioio_tifffile
import celldetection as cd

from HarmoMeshNet_architecture import HarmoMeshNet
from utils import compute_normal, save_mesh_vedo
from spherical_functions import LaplacianSmooth
import vedo

def Percentile_Normalization(projection: np.array, percentile_low: float = 0.00, percentile_high: float = 100.00):    
    low_val = np.percentile(projection, percentile_low)
    high_val = np.percentile(projection, percentile_high)
    if high_val <= low_val:
        max_val = np.max(projection)
        if max_val > 0:
            normalization = projection / max_val
        else:
            normalization = projection
    else:
        normalization = (projection - low_val) / (high_val - low_val)
    normalization = np.clip(normalization, 0.0, 1.0)
    return normalization

def Format_chanels(projection: np.array):
    if len(projection.shape) == 2:
        projection = np.stack([projection] * 3, axis=-1).astype(np.float32)
    if projection.shape[0] == 1:
        projection = np.repeat(projection, 3, axis=0)
    return projection

def Charge_model():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c'
    model = cd.fetch_model(model_name, check_hash=True).to(device)
    return model.eval(), device
    
def Detector(model, device, scene: np.array, mood: str = 'both', order=2, nms_thresh=0.31, score_thresh=0.90, samples=224.0, refinement=False, refinement_iterations=4.0, refinement_margin=700.0, refinement_buckets=300.0):
    if mood == 'max':
        projection = np.max(scene, axis=0).astype(np.float32)
    if mood == 'std':
        projection = np.std(scene, axis=0).astype(np.float32)
    if mood == 'both':
        projection = (np.max(scene, axis=0) + np.std(scene, axis=0)).astype(np.float32)

    projection = Percentile_Normalization(projection=projection)
    projection = Format_chanels(projection=projection)

    in_channels = projection.shape[2]
    with torch.no_grad():
        x = cd.to_tensor(projection, transpose=True, device=device, dtype=torch.float32)
        x = x[None]
        y = model(x, in_channels=in_channels, order=order, score_thresh=score_thresh, samples=samples, refinement=refinement, refinement_iterations=refinement_iterations, refinement_margin=refinement_margin, refinement_buckets=refinement_buckets)
    return projection, y

def generate_plot_detection(projection, bboxes, contours):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(projection) 
    ax.axis('off')
    cd.plot_boxes(bboxes)
    cd.plot_contours(contours)
    return fig

def check_files(im_path):
    if not os.path.exists(im_path):
        raise ValueError(f"Folder {im_path} doesn't exist")
    else:
        all_files = os.listdir(im_path)
        ids = sorted([f for f in all_files if f.endswith('.tiff') or f.endswith('.tif')])
    if not len(ids) > 0:
        raise ValueError(f"No tiff/tif files found in {im_path}")
    return ids

def select_full_cells(projection, bboxes, contours, margin=2):
    h, w = projection.shape[:2]
    is_on_edge = (contours[:, :, 0] <= margin).any(axis=1) | \
                 (contours[:, :, 1] <= margin).any(axis=1) | \
                 (contours[:, :, 0] >= w - margin).any(axis=1) | \
                 (contours[:, :, 1] >= h - margin).any(axis=1)
    mask = ~is_on_edge
    bboxes = bboxes[mask]
    contours = contours[mask]
    return bboxes, contours

def single_cell_extraction(im, bboxes, padding=5):
    max_z, max_y, max_x = im.shape
    single_cells_info = []
    for box in bboxes:
        x1, y1, x2, y2 = box
        y_start = int(np.clip(y1 - padding, 0, max_y))
        y_end   = int(np.clip(y2 + padding, 0, max_y))
        x_start = int(np.clip(x1 - padding, 0, max_x))
        x_end   = int(np.clip(x2 + padding, 0, max_x))
        single_cells_info.append((im[:, y_start:y_end, x_start:x_end], box))
    return single_cells_info

def unnormalize_vertices(v: torch.Tensor, vol_shape: tuple) -> torch.Tensor:
    Z, Y, X = vol_shape
    center = torch.tensor([X / 2.0, Y / 2.0, Z / 2.0], dtype=torch.float32, device=v.device)
    scale = torch.tensor([ X / 2.0, Y / 2.0, Z / 2.0], dtype=torch.float32, device=v.device)
    return v * scale + center

class InferenceDataset(Dataset):
    def __init__(self, image_dir: str):
        self.image_dir = image_dir
        self.images = check_files(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_fname = self.images[idx]
        stem = Path(img_fname).stem
        vol = tifffile.imread(os.path.join(self.image_dir, img_fname)).astype(np.float32)
        vol_shape = vol.shape
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        volume = torch.FloatTensor(vol).unsqueeze(0)
        return volume, vol_shape, stem

def apply_laplacian_smooth(v_tensor, f_tensor, iters, lambd):
    if iters <= 0:
        return v_tensor
    device = v_tensor.device
    v = v_tensor[0]
    f = f_tensor[0].cpu().numpy()
    edges_i = []
    edges_j = []
    for face in f:
        for start, end in [(0,1), (1,2), (2,0)]:
            edges_i.append(face[start])
            edges_j.append(face[end])
            edges_i.append(face[end])
            edges_j.append(face[start])
    indices = torch.tensor([edges_i, edges_j], device=device)
    values = torch.ones(len(edges_i), device=device)
    adj = torch.sparse_coo_tensor(indices, values, (len(v), len(v))).coalesce()
    smoother = LaplacianSmooth(aggr='mean')
    for _ in range(iters):
        v = smoother(v, adj, lambd=lambd)
    return v.unsqueeze(0)

def parse_args():
    parser = argparse.ArgumentParser(description="HarmoMeshNet Inference")
    parser.add_argument("--inference_mode", type=str, required=True, 
                        choices=["multicell2multicell", "multicell2singlecell", "singlecell2singlecell"])
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False, default='./HarmoMeshNet_Inference')
    
    parser.add_argument("--save_detection", action="store_true")
    parser.add_argument("--only_full_cells", action="store_true")
    parser.add_argument("--save_extracted_cells", action="store_true")
    parser.add_argument("--margin_exclusion", type=float, default=2)
    parser.add_argument("--padding", type=int, default=5)
    
    parser.add_argument("--n_start_filters", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--max_sh_degree", type=int, default=8)
    parser.add_argument("--sphere_subdivisions", type=int, default=4)
    parser.add_argument("--refiner_steps", type=int, default=3)
    parser.add_argument("--refiner_layers", type=int, default=3)
    parser.add_argument("--refiner_hidden", type=int, default=64)
    parser.add_argument("--n_smooth", type=int, default=1)
    parser.add_argument("--lambd", type=float, default=0.5)
    parser.add_argument("--post_smoothing_iters", type=int, default=0)
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup subdirectories
    if args.inference_mode in ["multicell2multicell", "multicell2singlecell"]:
        if args.save_extracted_cells:
            os.makedirs(os.path.join(args.output_dir, "extracted_cells"), exist_ok=True)
        if args.save_detection:
            os.makedirs(os.path.join(args.output_dir, "cell_detections_png"), exist_ok=True)
    
    model = HarmoMeshNet(
        in_ch=1, base_f=args.n_start_filters, latent_dim=args.latent_dim,
        max_sh_degree=args.max_sh_degree, sphere_subdivisions=args.sphere_subdivisions,
        refiner_steps=args.refiner_steps, refiner_layers=args.refiner_layers,
        refiner_hidden=args.refiner_hidden,
    ).to(device)
    
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()

    if args.inference_mode == "singlecell2singlecell":
        dataset = InferenceDataset(args.input_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        with torch.no_grad():
            for volume, vol_shape, stem_tuple in tqdm(dataloader, desc="Inference"):
                stem = stem_tuple[0]
                volume = volume.to(device)
                v_shape = (vol_shape[0].item(), vol_shape[1].item(), vol_shape[2].item())
                
                v_out, f_out = model(volume, n_smooth=args.n_smooth, lambd=args.lambd)
                v_out = unnormalize_vertices(v_out, v_shape)
                v_out = apply_laplacian_smooth(v_out, f_out, args.post_smoothing_iters, args.lambd)
                normal = compute_normal(v_out, f_out)
                
                save_mesh_vedo(
                    v_out[0].cpu().numpy(), f_out[0].cpu().numpy(), normal[0].cpu().numpy(),
                    os.path.join(args.output_dir, f"pred_{stem}.obj")
                )

    elif args.inference_mode in ["multicell2singlecell", "multicell2multicell"]:
        files = check_files(args.input_dir)
        det_model, det_device = Charge_model()

        for file in tqdm(files, desc='Process image'):
            fpath = os.path.join(args.input_dir, file)
            fname = Path(file).stem
            
            im = BioImage(fpath, reader=bioio_tifffile.Reader)
            im_data = im.get_image_data("ZYX", T=0)

            projection, prediction = Detector(model=det_model, device=det_device, scene=im_data)
            bboxes = prediction['boxes'][0].cpu().numpy()
            contours = prediction['contours'][0].cpu().numpy()

            if args.only_full_cells and len(contours) > 0:
                bboxes, contours = select_full_cells(projection, bboxes, contours, args.margin_exclusion)

            if args.save_detection:
                png_out = os.path.join(args.output_dir, 'cell_detections_png')
                fig = generate_plot_detection(projection, bboxes, contours)
                fig.savefig(os.path.join(png_out, f'{fname}_detection.png'), dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            cell_info = single_cell_extraction(im_data, bboxes, args.padding)
            

            scene_vertices = []
            scene_faces = []
            vert_offset = 0

            with torch.no_grad():
                for i, (cell, bbox) in enumerate(tqdm(cell_info, desc='Cell prediction', leave=False), start=1):
                    if args.save_extracted_cells:
                        tifffile.imwrite(os.path.join(args.output_dir, "extracted_cells", f"{fname}_cell{i}.tiff"), cell)
                    
                    vol_shape = cell.shape
                    vol = (cell.astype(np.float32) - cell.min()) / (cell.max() - cell.min() + 1e-8)
                    volume = torch.FloatTensor(vol).unsqueeze(0).unsqueeze(0).to(device)
                    
                    v_out, f_out = model(volume, n_smooth=args.n_smooth, lambd=args.lambd)
                    v_out = unnormalize_vertices(v_out, vol_shape)
                    v_out = apply_laplacian_smooth(v_out, f_out, args.post_smoothing_iters, args.lambd)
                    

                    v_np = v_out[0].cpu().numpy()
                    f_np = f_out[0].cpu().numpy()

                    if args.inference_mode == "multicell2singlecell":
                        normal = compute_normal(v_out, f_out)
                        save_mesh_vedo(
                            v_np, f_np, normal[0].cpu().numpy(),
                            os.path.join(args.output_dir, f"pred_{fname}_cell{i}.obj")
                        )
                    else:

                        x_start = max(0, int(bbox[0]) - args.padding)
                        y_start = max(0, int(bbox[1]) - args.padding)

                        v_np[:, 2] += x_start  
                        v_np[:, 1] += y_start  

                        scene_vertices.append(v_np)
                        scene_faces.append(f_np + vert_offset)
                        vert_offset += len(v_np)


            if args.inference_mode == "multicell2multicell" and scene_vertices:
                final_v = np.vstack(scene_vertices)
                final_f = np.vstack(scene_faces)
                scene_mesh = vedo.Mesh([final_v, final_f])
                scene_mesh.compute_normals()
                scene_mesh.write(os.path.join(args.output_dir, f'{fname}.obj'))

if __name__ == "__main__":
    main()