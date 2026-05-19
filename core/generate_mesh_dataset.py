import argparse
import os
import re
from pathlib import Path
import numpy as np
from scipy.ndimage import find_objects, center_of_mass
from skimage.morphology import remove_small_objects, remove_small_holes
import tifffile
from tqdm import tqdm
import pyshtools
from scipy.interpolate import griddata
import random
import shutil
from vedo import spher2cart, mag, Box, Point, Points, Plotter, Mesh, Volume, write
from spherical_functions import coefficients_to_gt_vector

def process_log_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found, please check the existence/name")

    # Global tracking variables for Sizes Info
    global_zmax = float('-inf')
    global_ymax = float('-inf')
    global_xmax = float('-inf')
    
    global_zmin = float('inf')
    global_ymin = float('inf')
    global_xmin = float('inf')
    
    global_max_vol = float('-inf')
    global_max_shape = ""
    global_im_M = ""
    
    global_min_vol = float('inf')
    global_min_shape = ""
    global_im_m = ""

    # Variables for calculating mean cell shape
    total_z = 0
    total_y = 0
    total_x = 0
    cell_count = 0

    # Updated regex to capture shape and the "cuted_by_axis" status
    # It continues to work even if new fields are added at the end (like gv_shape)
    cell_pattern = re.compile(r"cell_original_shape:\s*\((\d+),\s*(\d+),\s*(\d+)\)\|\s*cuted_by_axis:\s*([^|]+)")

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Extract cell info and apply filtering logic for mean calculation
            cell_match = cell_pattern.search(line)
            if cell_match:
                z, y, x, cut_status = cell_match.groups()
                z, y, x = float(z), float(y), float(x)
                cut_status = cut_status.strip()

                # Logic: Include if 'No computed' OR if explicitly 'False'
                # If it's 'True', it's excluded from the mean.
                if cut_status == "False" or cut_status == "No computed":
                    total_z += z
                    total_y += y
                    total_x += x
                    cell_count += 1

            # Extract Sizes Info line 
            if line.startswith("| Zmax:"):
                parts = [p.strip() for p in line.split('|') if p.strip()]
                info = {}
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        info[key.strip()] = value.strip()
                
                # We use float() instead of int() to handle cases where the log outputs 'inf'
                if 'Zmax' in info: global_zmax = max(global_zmax, float(info['Zmax']))
                if 'Ymax' in info: global_ymax = max(global_ymax, float(info['Ymax']))
                if 'Xmax' in info: global_xmax = max(global_xmax, float(info['Xmax']))
                
                if 'Zmin' in info: global_zmin = min(global_zmin, float(info['Zmin']))
                if 'Ymin' in info: global_ymin = min(global_ymin, float(info['Ymin']))
                if 'Xmin' in info: global_xmin = min(global_xmin, float(info['Xmin']))
                
                if 'max_vol' in info:
                    current_max_vol = float(info['max_vol'])
                    if current_max_vol > global_max_vol:
                        global_max_vol = current_max_vol
                        global_max_shape = info.get('max_shape', '')
                        global_im_M = info.get('im_M', '')
                
                if 'min_vol' in info:
                    current_min_vol = float(info['min_vol'])
                    if current_min_vol < global_min_vol:
                        global_min_vol = current_min_vol
                        global_min_shape = info.get('min_shape', '')
                        global_im_m = info.get('im_m', '')

    # Calculate the mean cell shape
    mean_z = total_z / cell_count if cell_count > 0 else 0
    mean_y = total_y / cell_count if cell_count > 0 else 0
    mean_x = total_x / cell_count if cell_count > 0 else 0

    # Determine the output path in the same directory as the analyzed file
    output_dir = os.path.dirname(os.path.abspath(file_path))
    output_file_path = os.path.join(output_dir, "global_info_dataset.txt")

    # Save the results to the txt file
    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        out_file.write("----- Global Sizes Info -----\n")
        out_file.write(f"Zmax: {global_zmax}, Ymax: {global_ymax}, Xmax: {global_xmax}\n")
        out_file.write(f"Zmin: {global_zmin}, Ymin: {global_ymin}, Xmin: {global_xmin}\n")
        out_file.write(f"Max Volume: {global_max_vol}\n")
        out_file.write(f"Max Shape: {global_max_shape}\n")
        out_file.write(f"Image with Max Volume (im_M): {global_im_M}\n")
        out_file.write("-" * 29 + "\n")
        out_file.write(f"Min Volume: {global_min_vol}\n")
        out_file.write(f"Min Shape: {global_min_shape}\n")
        out_file.write(f"Image with Min Volume (im_m): {global_im_m}\n\n")
        
        out_file.write("----- Mean Cell Shape (Filtered) -----\n")
        out_file.write(f"Total valid cells (False or No computed): {cell_count}\n")
        out_file.write(f"Mean Shape (Z, Y, X): ({mean_z:.2f}, {mean_y:.2f}, {mean_x:.2f})\n")


def generate_test_set(full_train_set,p_testset=0.2):
    out = Path(full_train_set).parent / 'test_dataset'

    if not os.path.exists(full_train_set):
        raise ValueError(f"Folder {full_train_set} doesn't exist")    

    if not os.path.exists(out):
        os.makedirs(out)

    all_files = os.listdir(full_train_set)
    ids = set(f.replace('_IM.tiff','') for f in all_files if f.endswith('.tiff'))

    num_to_move = max(1, int(len(ids) * p_testset))

    ids_to_move = random.sample(list(ids), num_to_move)

    print(f"Total files found: {len(ids)}")
    print(f"Move {num_to_move} files (approx. {p_testset*100}%)...")

    for file_id in tqdm(ids_to_move,desc='movig files'):
        img_name = f"{file_id}_IM.tiff"
        gt_name = f"{file_id}_GT.npy"
        for name in [img_name, gt_name]:
            src_path = os.path.join(full_train_set , name)
            dst_path = os.path.join(out, name)
            shutil.move(src_path, dst_path)



def extract_and_clean_single_cells(image_volume,
    segmentation_volume,
    only_centred=False,
    correct_seg=True,
    save_seg=False,
    save_mesh=False,
    output='./im2mesh_dataset',
    name='',
    binarization=False,
    N=100,
    min_shape_constrain=[0,0,0],
    max_shape_constrain=[0,0,0],
    padding=5,
    margin=2):

    unique_labels = np.unique(segmentation_volume)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        return []

    max_label = int(np.max(segmentation_volume))
    slices = find_objects(segmentation_volume, max_label)
    
    vol_z, vol_y, vol_x = segmentation_volume.shape
    extracted_cells = []
    
    with open(os.path.join(output,"dataset_creation_log.txt"), "a") as f:
        f.write(f"################################################# Scene {name} | shape {image_volume.shape} | discarted_cuted {only_centred}  ################################################# \n")
        
        max_z, max_y, max_x = 0, 0, 0
        min_z, min_y, min_x = float('inf'), float('inf'), float('inf')
        shape_max = None
        shape_min = None
        vol_max = 0
        vol_min = float('inf')
        cellM = ''
        cellm = ''

        for label_idx in tqdm(unique_labels,desc='processing single cells', leave=False):
            
            slc = slices[label_idx - 1]
            
            if slc is None:
                continue
            
            ############# all the volume z size ######################
            z_start = 0 
            z_stop = vol_z


            ############## only contain cell in z #################

            # z_start = max(0, slc[0].start - 1)  
            # z_stop = min(vol_z, slc[0].stop + 1)
            
            y_start = max(0, slc[1].start - padding)
            y_stop = min(vol_y, slc[1].stop + padding)
            
            x_start = max(0, slc[2].start - padding)
            x_stop = min(vol_x, slc[2].stop + padding)
            
            expanded_slc = (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))

            sub_image = image_volume[expanded_slc].copy()
            sub_seg = segmentation_volume[expanded_slc].copy()

            if min_shape_constrain != [0,0,0]:
                zo,yo,xo = sub_image.shape
                if zo < min_shape_constrain[0] or yo < min_shape_constrain[1] or xo < min_shape_constrain[2]:
                    continue
            
            if max_shape_constrain != [0,0,0]:
                zo,yo,xo = sub_image.shape
                if zo > max_shape_constrain[0] or yo > max_shape_constrain[1] or xo > max_shape_constrain[2]:
                    continue

            if np.sum(sub_seg == label_idx) < 10: 
                continue

            # Clean the mask: isolate the dominant cell in the crop
            labels_in_sub, counts = np.unique(sub_seg, return_counts=True)
            non_zero_mask = labels_in_sub != 0
            labels_in_sub = labels_in_sub[non_zero_mask]
            counts = counts[non_zero_mask]

            if len(labels_in_sub) > 0:
                largest_label = labels_in_sub[np.argmax(counts)]
                cleaned_sub_seg = np.where(sub_seg == largest_label, largest_label, 0)
            else:
                cleaned_sub_seg = sub_seg
                largest_label = label_idx

            if correct_seg:
                mask = (cleaned_sub_seg == largest_label)
                mask = remove_small_objects(mask, max_size=5)
                mask = remove_small_holes(mask, max_size=5)
                if not np.any(mask):
                    continue
                temp_cleaned = np.zeros_like(cleaned_sub_seg)
                temp_cleaned[mask] = largest_label
                cleaned_sub_seg = temp_cleaned.astype(np.uint8)



            Line = f"Cell {str(int(largest_label)).zfill(2)} | cell_original_shape: {sub_image.shape}"

            # Check if the cell touches the global volume boundaries
            if only_centred:
                # Add a margin tolerance to match the explicit_model_prediction logic
                # This catches cells that are visually cut but mathematically 1-2 pixels away from the absolute edge
                is_cut = False
                
                # Check Z axis boundaries (slc.stop is exclusive, so we compare with vol - margin)
                if slc[0].start <= margin or slc[0].stop >= vol_z - margin:
                    is_cut = True
                # Check Y axis boundaries
                elif slc[1].start <= margin or slc[1].stop >= vol_y - margin:
                    is_cut = True
                # Check X axis boundaries
                elif slc[2].start <= margin or slc[2].stop >= vol_x - margin:
                    is_cut = True

                # Discard the cell if its bounding box falls within the edge margin
                if is_cut:
                    f.write(Line + f"| cuted_by_axis: {is_cut} | gv_shape: (nana,) -> (3 + (lmax)**2,)\n")
                    continue
            else:
                is_cut = 'No computed'


            z, y, x = sub_image.shape
            vol_actual = z * y * x
            # max
            if z > max_z: max_z = z
            if y > max_y: max_y = y
            if x > max_x: max_x = x
            if vol_actual > vol_max:
                vol_max = vol_actual
                shape_max = sub_image.shape
                cellM = f"{Path(name).stem}_{int(largest_label)}"
        
            # min
            if z < min_z: min_z = z
            if y < min_y: min_y = y
            if x < min_x: min_x = x
            if vol_actual < vol_min:
                vol_min = vol_actual
                shape_min = sub_image.shape
                cellm= f"{Path(name).stem}_{int(largest_label)}"

            Line = Line + f"| cuted_by_axis: {is_cut}"

            im_out = os.path.join(output,'training_set')
            if not os.path.exists(im_out):
                os.makedirs(im_out)
            
            tifffile.imwrite(f"{im_out}/{Path(name).stem}_{int(largest_label)}_IM.tiff",sub_image)

            if save_seg:
                seg_out = os.path.join(output,'segmentations')
                if not os.path.exists(seg_out):
                    os.makedirs(seg_out)

                if binarization:
                    cleaned_sub_seg = (cleaned_sub_seg > 0).astype(np.uint8)

                tifffile.imwrite(f"{seg_out}/{Path(name).stem}_{int(largest_label)}.tiff",cleaned_sub_seg)
           
            surface = Volume(cleaned_sub_seg).isosurface(value=0.5)

            if is_cut == 'No computed': 
                surface =  surface.clean().fill_holes(size=1e10)
            if save_mesh:
                mesh_out = os.path.join(output,'meshes')
                if not os.path.exists(mesh_out):
                    os.makedirs(mesh_out)
                write(surface,f"{mesh_out}/{Path(name).stem}_{int(largest_label)}.obj")
            
            x0 = surface.center_of_mass()
            rmax = surface.diagonal_size() 
            # #####################################
            # lmax_obj = 20
            # N = 2 * lmax_obj+ 2
            # ###############################
            agrid = []
            for th in np.linspace(0, np.pi, N, endpoint=True):
                longs = []
                for ph in np.linspace(0, 2*np.pi, N, endpoint=False):
                    # Origin with spheric bundle coordinates
                    p_dest = spher2cart(rmax, th, ph)
                    # Intersection among x0 and bundle
                    intersections = surface.intersect_with_line(x0, x0 + p_dest)
                    
                    if len(intersections):
                        # last intersection
                        dist = mag(intersections[-1] - x0)
                        longs.append(dist)
                    else:
                        # if no intersec radius 0
                        longs.append(0)
                agrid.append(longs)
            agrid = np.array(agrid)

            # Spheric armonic expansion 
            grid = pyshtools.SHGrid.from_array(agrid)
            clm = grid.expand()
            
            gt_vector = coefficients_to_gt_vector(clm.to_array(),x0)

            np.save(f"{im_out}/{Path(name).stem}_{int(largest_label)}_GT.npy", gt_vector)
            
            Line = Line + f"| gv_shape: {gt_vector.shape} -> (3 + (lmax)**2,)\n"
            
            f.write(Line)  
        f.write("################################################# Sizes Info ################################################# \n")
        f.write(f"| Zmax:{max_z}| Ymax:{max_y} | Xmax:{max_x} | Zmin:{min_z}| Ymin:{min_y} | Xmin:{min_x} | max_shape:{shape_max}| min_shape:{shape_min} | max_vol:{vol_max}| min_vol:{vol_min} | im_M:{cellM} | im_m:{cellm} | \n")
            

             
def check_files(im_path,gt_path):
    paths = [im_path,gt_path]
    n = [0,0]
    for i , path in enumerate(paths):
        if not os.path.exists(path):
            raise ValueError(f"Folder {path} doesn't exist")
        else:
            n[i] = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

    if n[0] != n[1]:
        raise ValueError(f"{n[0]} images and {n[1]} segmentations were found, please check missmatch")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate im2mesh dataset for training")

    parser.add_argument('--image_files', type=str, required=True, help='Path to raw Allan dataset images')
    parser.add_argument('--segentation_files', type=str, required=True, help='Path to GT segmentation masks')
    parser.add_argument('--only_full_cell', action='store_true', help='Pass this flag to discart the cells on the image edge')
    parser.add_argument('--save_seg', action='store_true', help='Pass this flag to save the correspondig segmentation')
    parser.add_argument('--save_mesh', action='store_true', help='Pass this flag to save the correspondig mesh')
    parser.add_argument('--correct_seg', action='store_true', help='Pass this flag to close posible holes and delete one pixel segmentations')
    parser.add_argument('--binarization', action='store_true', help='Pass this flag to binarize single cell')
    parser.add_argument('--N_grid', type=int, required=False, default=100, help='Number of grid intervals on the unit sphere')
    parser.add_argument('--p_testset', type=float, required=False, default=None, help='Percentage of data reserved for the test dataset')
    parser.add_argument('--min_shape_constrain', type=int, nargs='+', required=False, default=[0,0,0], help='Min shape to keep a single cell z , y, x (e.g., 128 128 128)')
    parser.add_argument('--max_shape_constrain', type=int, nargs='+', required=False, default=[0,0,0], help='Max shape to keep a single cell z , y, x (e.g., 128 128 128)')
    parser.add_argument('--padding', type=int, required=False, default=5, help='Padding for single cell cropping in 3D')
    parser.add_argument('--margin', type=int, required=False, default=2, help='Discrimination Margin')


    args = parser.parse_args()
    
    image_files = args.image_files
    segentation_files = args.segentation_files
    only_full_cell = args.only_full_cell
    save_seg = args.save_seg
    save_mesh = args.save_mesh
    correct_seg = args.correct_seg
    binarization = args.binarization
    N_grid = args.N_grid
    p_testset = args.p_testset
    min_shape_constrain = args.min_shape_constrain
    max_shape_constrain = args.max_shape_constrain
    padding = args.padding
    margin = args.margin



    print("###################################################################################################### Starting process ######################################################################################################")

    check_files(image_files,segentation_files)

    out_root = os.path.join(os.getcwd(),'im2mesh_dataset')
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    
    open(os.path.join(out_root, "dataset_creation_log.txt"), 'w').close()
    files = [file for file in os.listdir(image_files) if os.path.isfile(os.path.join(image_files, file))]
    for file in tqdm(files,desc='Processing volumes', leave=True):
        vol_path = os.path.join(image_files,file)
        seg_path = os.path.join(segentation_files,file)

        vol = tifffile.imread(vol_path).astype(np.float32)
        seg = tifffile.imread(seg_path).astype(np.uint8)
     
        extract_and_clean_single_cells(vol,seg,
            only_full_cell,
            correct_seg,
            save_seg,save_mesh,
            out_root,
            file,
            binarization,
            N_grid,
            min_shape_constrain,
            max_shape_constrain,
            padding,
            margin)
       
    
    process_log_file(os.path.join(out_root,'dataset_creation_log.txt'))

    if p_testset is not None:
        generate_test_set(os.path.join(out_root, 'training_set'),p_testset)

    print("###################################################################################################### Process Ready ######################################################################################################")
    print(f"Dataset results saved in -> {out_root}")