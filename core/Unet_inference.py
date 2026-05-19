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
from mmv_im2im.configs.config_base import (
    ProgramConfig,
    parse_adaptor,
    configuration_validation,
)
from vector2Mesh import vec2mesh
from mmv_im2im import ProjectTester

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
    
    '''
    order: Contour order. The higher, the more complex contours can be proposed. order=1 restricts the CPN to propose ellipses, order=3 allows for non-convex rough outlines, order=8 allows even finer detail.
    nms_thresh: IoU threshold for non-maximum suppression (NMS). NMS considers all objects with iou > nms_thresh to be identical.
    score_thresh: Score threshold. For binary classification problems (object vs. background) an object must have score > score_thresh to be proposed as a result.
    samples: Sampling points  Number of samples. This sets the number of coordinates with which a contour is defined. This setting can be changed on the fly, e.g. small for training and large for inference. Small settings reduces computational costs, while larger settings capture more detail.
    refinement: Whether to use local refinement or not. Local refinement generally improves pixel precision of the proposed contours.
    refinement_iterations: Refinement step  Number of refinement iterations.
    refinement_margin: Tile size Maximum refinement margin (step size) per iteration.
    refinement_buckets: Overlap size Number of refinement buckets. Bucketed refinement is especially recommended for data with overlapping objects.
    
    '''


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

    parser.add_argument('--yaml_inference', type=str, required=True, help='Path to .yaml inference config file')
    parser.add_argument('--mode_prediction', type=str, required=True, help='Prediction mode multicell2multicell/multicell2singlecell/singlecell2singlecell')
    parser.add_argument('--save_detection', action='store_true', help='Pass this flag to save the png cell detection results')
    parser.add_argument('--save_single_shd', action='store_true', help='Pass this flag to save the single cell descomposition on multicell mode')
    parser.add_argument('--only_full_cells', action='store_true', help='Discard cells that are actually cut by image edges based on contours')
    parser.add_argument('--margin_exclusion',type=float, required=False, default=2, help='Touch margin treshold')
    parser.add_argument('--padding',type=int, required=False, default=5, help='Padding for single cell cropping')
    parser.add_argument('--grid_generator', action='store_true', help='Build meshes with grid (worst quality) instead of icosphere (best quality)')
    
    
    args = parser.parse_args()
    
    yaml_inference = args.yaml_inference
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


    cfg = parse_adaptor(config_class=ProgramConfig, config=yaml_inference, args=[])
    cfg = configuration_validation(cfg)

    out_folder = cfg.data.inference_output.path
    input_folder = cfg.data.inference_input.dir

    files = check_files(input_folder)

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
    if mode_prediction == modes[2]:
        print(f"############################# Prediction on {len(files)} single cells ###########################################################")
        cfg.data.inference_output.path = out_folder/'SH_decomposition'
        exe = ProjectTester(cfg)
        exe.run_inference()
    else:
        
        model, device = Charge_model()
        open(out_folder/'cell_detection.txt', 'w').close()

        if mode_prediction != modes[2]:
            executor = ProjectTester(cfg)
            executor.setup_model()
            executor.setup_data_processing()

        for file in tqdm(files,desc='Process image'):
            # Image and file readings
            fpath = input_folder/file
            fname = Path(file).stem
            im = BioImage(fpath, reader=bioio_tifffile.Reader)
            im = im.get_image_data("ZYX", T=0)

            # Cell detection
            projection,prediction = Detector(model=model,device=device,scene = im)
            bboxes = prediction['boxes'][0].cpu().numpy()
            contours = prediction['contours'][0].cpu().numpy()

            # Full cells 
            if only_full_cells and len(contours) > 0:
                bboxes, contours = select_full_cells(projection,bboxes,contours,margin_exclusion)

            # Save detection output
            if save_detection: 
                figure = generate_plot_detection(projection, bboxes, contours)
                figure.savefig(f'{png_out}/{fname}_detection.png', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close(figure)

            if mode_prediction != modes[2]:
                cell_info = single_cell_extraction(im,bboxes,padding)
                tqdm.write(f"############################# {len(cell_info)} Single cells detected #############################")

                # Accumulate cell data for scene reconstruction (modes[0] only)
                cells_for_scene = []
    
                with torch.no_grad():
                    with open(out_folder / 'cell_detection.txt', 'a') as f:
                        f.write(f"### Scene_name: {fname} | Scene_shape: {im.shape} | Detected_cells: {len(cell_info)} ###\n")
                        for i, (cell, bbox) in enumerate(tqdm(cell_info, desc='Cell prediction', leave=False), start=1):
                            
                            SHD = executor.process_one_image(cell)
    
                            if mode_prediction != modes[0]:
                                # multicell2singlecell: save each SH decomposition
                                np.save(out_folder / 'SH_decomposition' / f"{fname}_cell{i}.npy", SHD)
                            else:
                                # multicell2multicell: accumulate for scene reconstruction
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
        vec2mesh(out_folder/'SH_decomposition',out_folder/'Generated_meshes',modeG)



        
             
                
                



            


                

            
                



    

    



 