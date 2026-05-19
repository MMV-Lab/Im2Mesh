import quilt3 as q3
import os
import argparse

def get_data(only_images=False,only_watershed=False):

    if only_images:
        datasets = ['high_res_100x']
    elif only_watershed:      
        datasets = ['watershed_segmentation_100x']
    else:
        datasets = ['high_res_100x', 'watershed_segmentation_100x']

    path = os.path.join(os.getcwd(),'Allancell_Data')
    
    if not os.path.exists(path):
        os.makedirs(path)
  
    for dataset in datasets:
        subfolder = os.path.join(path,dataset)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        b = q3.Bucket("s3://allencell") 
        print(f"Download {dataset} in: {subfolder}")
        b.fetch(
            f"aics/nuc-morph-dataset/hipsc_nuclei_image_datasets_for_training_deep_learning_models/segmentation_decoder_training_fov_dataset/{dataset}/",
            f"{subfolder}/"
        )
    print("Download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Donwload Allencell Dataset")

    parser.add_argument('--only_images', action='store_true', help='Pass this flag to only download raw images')
    parser.add_argument('--only_watershed', action='store_true', help='Pass this flag to only download watershed segmentation')
    args = parser.parse_args()
    
    only_images = args.only_images
    only_watershed = args.only_watershed

    get_data(only_images,only_watershed)