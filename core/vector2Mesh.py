
import pyshtools
import numpy as np
from scipy.interpolate import griddata
from vedo import spher2cart, mag, Box, Point, Points, Plotter, Mesh, write
import os
import argparse
from spherical_functions import gt_vector_to_coefficients, points_to_mesh, vector_to_points
from tqdm import tqdm


def vec2mesh(input_files,out_files='./meshes_generated',mode='icosphera'):

    if not os.path.exists(input_files):
        raise ValueError(f"Folder {input_files} doesn't exist")
    
    if not os.path.exists(out_files):
        os.makedirs(out_files)
    
    all_files = os.listdir(input_files)
    ids = set(f for f in all_files if f.endswith('.npy'))

    for file_id in tqdm(ids,desc='genereting meshes'):
        path_file = os.path.join(input_files,file_id)
        SHD_vector = np.load(path_file)
        x0 , coeffs_reconstructed = gt_vector_to_coefficients(SHD_vector)
        if mode == 'grid':
            pts2, n_lat, n_lon = vector_to_points(SH_coef=coeffs_reconstructed,x0=x0,mode=mode)
            reconstructed_points = Points(pts2, r=3).c("blue5").alpha(1)
            mesh = points_to_mesh(pts_list=pts2, n_lat=n_lat, n_lon=n_lon,mode=mode)
        else:
            pts, faces = vector_to_points(SH_coef=coeffs_reconstructed,x0=x0,mode=mode)
            mesh = points_to_mesh(pts_list=pts,faces=faces,mode=mode)
        write(mesh,os.path.join(out_files,f"{file_id.replace('.npy','')}.obj"))
    
    print(f"Generated meshes saven in -> {out_files}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform .npy files into .obj files")

    parser.add_argument('--input_files', type=str, required=True, help='Path to .npy files')
    parser.add_argument('--out_folder', type=str, required=False, default='./meshes_generated', help='Path for output files')
    parser.add_argument('--grid_generator', action='store_true', help='Build meshes with grid (worst quality) instead of icosphere (best quality)')
    
    
    args = parser.parse_args()
    
    input_files = args.input_files
    out_folder = args.out_folder
    grid_generator = args.grid_generator

    if grid_generator:
        mode = 'grid'
    else:
        mode='icosphera'   
    
    vec2mesh(input_files,out_folder,mode)





