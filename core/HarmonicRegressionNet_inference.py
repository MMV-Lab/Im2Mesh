import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from bioio import BioImage
import os
import numpy as np 
import bioio_tifffile
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import celldetection as cd
import torch
import argparse
from pathlib import Path
import skimage.io
from bioio.writers import OmeTiffWriter
from skimage.draw import polygon
import shutil
import random
import math
from spherical_functions import reconstruct_scene
from vector2Mesh import vec2mesh
from HarmonicRegressionNet_architecture import HarmonicRegressionNet

def Percentile_Normalization(projection:np.array, percentile_low: float =0.00,percentile_high: float =100.00):    
    
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

def Format_chanels(projection:np.array):
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
    
    
def Detector(model, device, scene : np.array, mood:str = 'both',order=2,nms_thresh = 0.31,score_thresh = 0.90,samples = 224.0,refinement = False,refinement_iterations = 4.0,refinement_margin = 700.0,refinement_buckets =  300.0):
    
    if mood == 'max':
        projection = np.max(scene, axis=0).astype(np.float32)
    if mood == 'std':
        projection = np.std(scene, axis=0).astype(np.float32)
    if mood == 'both':
        projection = (np.max(scene, axis=0)+np.std(scene, axis=0)).astype(np.float32)

    projection = Percentile_Normalization(projection=projection)
    
    projection = Format_chanels(projection=projection)

    in_channels = projection.shape[2]
    with torch.no_grad():
        x = cd.to_tensor(projection, transpose=True, device=device, dtype=torch.float32)
        x = x[None]
        y = model(x, in_channels=in_channels, order=order, score_thresh=score_thresh, samples=samples, refinement=refinement, refinement_iterations=refinement_iterations, refinement_margin=refinement_margin, refinement_buckets=refinement_buckets)
    return projection , y

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
        ids = set(f for f in all_files if f.endswith('.tiff') or f.endswith('.tif') )
    
    if not len(ids)>0:
        raise ValueError(f"No tiff/tif files found in {im_path}")
    
    return ids

def select_full_cells(projection,bboxes,contours,margin=2):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image")

    parser.add_argument('--input_folder', type=str, required=True, help='Path to input folder')
    parser.add_argument('--out_folder', type=str,  required=False, default='HarmonicRegressionNet_Inference' , help='Path to output folder')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to HarmonicRegressionNet checkpoint')
    parser.add_argument('--n_start_filters', type=int, default=32, help='Base filters for HarmonicRegressionNet')
    parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension for HarmonicRegressionNet')
    parser.add_argument('--target_dim', type=int, default=2503, help='Target dimension for HarmonicRegressionNet')
    
    parser.add_argument('--mode_prediction', type=str, required=True, help='Prediction mode multicell2multicell/multicell2singlecell/singlecell2singlecell')
    parser.add_argument('--save_detection', action='store_true', help='Pass this flag to save the png cell detection results')
    parser.add_argument('--save_single_shd', action='store_true', help='Pass this flag to save the single cell descomposition on multicell mode')
    parser.add_argument('--only_full_cells', action='store_true', help='Discard cells that are actually cut by image edges based on contours')
    parser.add_argument('--margin_exclusion',type=float, required=False, default=2, help='Touch margin treshold')
    parser.add_argument('--padding',type=int, required=False, default=5, help='Padding for single cell cropping')
    parser.add_argument('--grid_generator', action='store_true', help='Build meshes with grid (worst quality) instead of icosphere (best quality)')
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    out_folder = Path(args.out_folder)
    checkpoint_path = args.checkpoint
    n_start_filters = args.n_start_filters
    latent_dim = args.latent_dim
    target_dim = args.target_dim

    mode_prediction = args.mode_prediction
    save_detection = args.save_detection
    only_full_cells = args.only_full_cells
    margin_exclusion = args.margin_exclusion
    save_single = args.save_single_shd
    padding = args.padding
    grid_generator = args.grid_generator

    if grid_generator:
        modeG = 'grid'
    else:
        modeG='icosphera'
    
    modes = ["multicell2multicell","multicell2singlecell","singlecell2singlecell"]
    if mode_prediction not in modes:
        raise ValueError(f" --mode_prediction should be one of the following options : {modes} but {mode_prediction} was given")

    files = check_files(str(input_folder))

    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    
    if mode_prediction != modes[0]:
        os.makedirs(out_folder/'SH_decomposition', exist_ok=True)
    else:
        if save_single:
            os.makedirs(out_folder/'SH_singlecell_decomposition', exist_ok=True)


    os.makedirs(out_folder/'Generated_meshes', exist_ok=True)
    
    if save_detection and mode_prediction != modes[2]:
        png_out = os.path.join(out_folder,'cell_detections_png')
        if not os.path.exists(png_out):
            os.makedirs(png_out, exist_ok=True)

    
    print(f'############################# {len(files)} Files found for process #############################')
    print(f'############################# {mode_prediction} mode execution #############################')

    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

    regression_model = HarmonicRegressionNet(
        in_ch=1,
        base_f=n_start_filters,
        latent_dim=latent_dim,
        target_dim=target_dim
    ).to(device)

    regression_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    regression_model.eval()

    if mode_prediction == modes[2]:
        print(f"############################# Prediction on {len(files)} single cells ###########################################################")
        
        for file in tqdm(files, desc='Process single cell'):
            fpath = input_folder/file
            fname = Path(file).stem
            im = BioImage(fpath, reader=bioio_tifffile.Reader)
            im = im.get_image_data("ZYX", T=0).astype(np.float32)

            vol = (im - im.min()) / (im.max() - im.min() + 1e-8)
            volume = torch.FloatTensor(vol).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_vector = regression_model(volume)
                SHD = pred_vector.cpu().numpy()[0]

            np.save(out_folder / 'SH_decomposition' / f"{fname}.npy", SHD)
            del SHD
            torch.cuda.empty_cache()

    else:
        
        model, device_cd = Charge_model()
        open(out_folder/'cell_detection.txt', 'w').close()

        for file in tqdm(files,desc='Process image'):
            fpath = input_folder/file
            fname = Path(file).stem
            im = BioImage(fpath, reader=bioio_tifffile.Reader)
            im = im.get_image_data("ZYX", T=0)

            projection,prediction = Detector(model=model,device=device_cd,scene = im)
            bboxes = prediction['boxes'][0].cpu().numpy()
            contours = prediction['contours'][0].cpu().numpy()

            if only_full_cells and len(contours) > 0:
                bboxes, contours = select_full_cells(projection,bboxes,contours,margin_exclusion)

            if save_detection: 
                figure = generate_plot_detection(projection, bboxes, contours)
                figure.savefig(f'{png_out}/{fname}_detection.png', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close(figure)

            cell_info = single_cell_extraction(im,bboxes,padding)
            
            cells_for_scene = []

            with torch.no_grad():
                with open(out_folder / 'cell_detection.txt', 'a') as f:
                    f.write(f"### Scene_name: {fname} | Scene_shape: {im.shape} | Detected_cells: {len(cell_info)} ###\n")
                    for i, (cell, bbox) in enumerate(tqdm(cell_info, desc='Cell prediction', leave=False), start=1):
                        
                        cell = cell.astype(np.float32)
                        vol = (cell - cell.min()) / (cell.max() - cell.min() + 1e-8)
                        volume = torch.FloatTensor(vol).unsqueeze(0).unsqueeze(0).to(device)

                        pred_vector = regression_model(volume)
                        SHD = pred_vector.cpu().numpy()[0]

                        if mode_prediction != modes[0]:
                            np.save(out_folder / 'SH_decomposition' / f"{fname}_cell{i}.npy", SHD)
                        else:
                            if save_single:
                                np.save(out_folder / 'SH_singlecell_decomposition' / f"{fname}_cell{i}.npy", SHD)
                            
                            cells_for_scene.append({
                                'sh_vector': SHD,
                                'bbox': bbox,
                            })

                        del SHD
                        torch.cuda.empty_cache()
                        f.write(f" Cell: {fname}_cell{i} | bbox: {bbox}\n")

            if mode_prediction == modes[0] and len(cells_for_scene) > 0:
                scene_mesh = reconstruct_scene(cells=cells_for_scene, padding=padding,mode=modeG)
                scene_mesh.write(str(out_folder / 'Generated_meshes' / f'{fname}.obj'))

    if mode_prediction != modes[0] :
        vec2mesh(str(out_folder/'SH_decomposition'), str(out_folder/'Generated_meshes'), modeG)