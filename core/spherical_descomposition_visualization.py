import pyshtools
import numpy as np
from scipy.interpolate import griddata
from vedo import spher2cart, mag, Box, Point, Points, Plotter, Mesh
import os
import argparse
from spherical_functions import gt_vector_to_coefficients, points_to_mesh, vector_to_points


def plot_SHD_mesh(path_file,mode='icosphera'):

    SHD_vector = np.load(path_file)

    x0, coeffs_reconstructed = gt_vector_to_coefficients(SHD_vector, centers=True)

    lmax = coeffs_reconstructed.shape[1] - 1

    if mode == 'grid':
        pts2, n_lat, n_lon = vector_to_points(SH_coef=coeffs_reconstructed,x0=x0,mode=mode)
        reconstructed_points = Points(pts2, r=3).c("blue5").alpha(1)
        mesh = points_to_mesh(pts_list=pts2, n_lat=n_lat, n_lon=n_lon,mode=mode)
    else:
        pts, faces = vector_to_points(SH_coef=coeffs_reconstructed,x0=x0,mode=mode)
        mesh = points_to_mesh(pts_list=pts,faces=faces,mode=mode)

    points_shd = [reconstructed_points, Point(x0, c='black')]

    mesh_reconstructed = [
        mesh.color('grey').alpha(0.8),
        Point(x0, c='black')
    ]

    plt = Plotter(shape=(1, 2), axes=1, sharecam=False)
    plt.at(0).show(points_shd, f"SHD Points (lmax={lmax})")
    plt.at(1).show(mesh_reconstructed, "Reconstructed Mesh")
    plt.interactive().close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizing Spherical Harmonics Decomposition and Mesh Reconstruction")

    parser.add_argument('--path_file', type=str, required=True, help='Path to .npy file')
    parser.add_argument('--grid_generator', action='store_true', help='Build meshes with grid (worst quality) instead of icosphere (best quality)')

    args = parser.parse_args()

    grid_generator = args.grid_generator

    if grid_generator:
        mode = 'grid'
    else:
        mode='icosphera' 

    plot_SHD_mesh(args.path_file,mode)