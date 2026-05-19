import quilt3 as q3
import os
import numpy as np
import argparse
import tifffile
from pathlib import Path
from tqdm import tqdm
def get_data(input_path):

    path = os.path.join(os.getcwd(),'Allancell_Data')
    if not os.path.exists(path):
        os.makedirs(path)

    subfolder = os.path.join(path,'volum_model_segmentations')
    if not os.path.exists(path):
        os.makedirs(subfolder)
    
    dataset_train = os.path.join(path,'cellpose_train_dataset')
    if not os.path.exists(path):
        os.makedirs(dataset_train)
    

    b = q3.Bucket("s3://allencell") 
    print("##################################### Download model segmentations in ########################################")
    print(f"###### {subfolder} ######")
    b.fetch(
        f"aics/nuc-morph-dataset/hipsc_nuclei_image_datasets_for_training_deep_learning_models/segmentation_decoder_training_fov_dataset/model_segmentation_100x/",
        f"{subfolder}/"
    )

    print("################################# Download complete ################################################")
    print("################################# Starting datset parsing ################################################")
    files = [f for f in os.listdir(subfolder)]

    for file in tqdm(files,desc='processing volumes'):

        volume_path = os.path.join(input_path, file)
        segmentation_path = os.path.join(subfolder, file)

        volume_stack = tifffile.imread(volume_path)
        segmentation_stack = tifffile.imread(segmentation_path)

        n_planes = min(volume_stack.shape[0], segmentation_stack.shape[0])

        for z in range(n_planes): 
            if np.any(segmentation_stack[z] > 0) and z>0:
                slice_name = f"{Path(file).stem}_z{z:03d}"
                tifffile.imwrite(f"{dataset_train}/{slice_name}.tif", volume_stack[z])
                tifffile.imwrite(f"{dataset_train}/{slice_name}_seg.tif", segmentation_stack[z])

    print("################################# Dataset ready ################################################")
    print(f"Datased saved -> {dataset_train} ######")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for cellpose training")

    parser.add_argument('--path_files', type=str, required=True, help='Path to raw Allan dataset images')
    
    args = parser.parse_args()
    
    path_files = args.path_files

    get_data(path_files)