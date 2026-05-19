import argparse
import os
from pathlib import Path
from tqdm import tqdm
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate harmmesh dataset for training")

    parser.add_argument('--dataset_path', type=str, required=True, help='Path to genertated im2mesh_dataset')
    parser.add_argument('--output_path', type=str, required=False,default="./harmmesh_dataset", help='Path to dataset output')

    args = parser.parse_args()
    
    dataset_file = Path(args.dataset_path)
    output_path = Path(args.output_path)
    
    if not os.path.exists(dataset_file):
        raise ValueError(f"No dataset folder found : {dataset_file}")
    elif not os.path.exists(dataset_file/'meshes'):
        raise ValueError(f"No meshe folder found: {dataset_file/'meshes'}")
    elif not os.path.exists(dataset_file/'training_set'):
        raise ValueError(f"No taining folder found: {dataset_file/'training_set'}")
    else:
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(output_path/'train'/'images', exist_ok=True)
        os.makedirs(output_path/'train'/'meshes', exist_ok=True)
    
    if not os.path.exists(dataset_file/'test_dataset'):
        is_test = False
    else:
        is_test = True 
        test_files = set(f for f in os.listdir(dataset_file/'test_dataset') if f.endswith('.tiff') or f.endswith('.tif'))
        os.makedirs(output_path/'test'/'images', exist_ok=True)
        os.makedirs(output_path/'test'/'meshes', exist_ok=True)
    
    files = set(f for f in os.listdir(dataset_file/'training_set') if f.endswith('.tiff') or f.endswith('.tif') )

    for file in tqdm(files,desc='Moving'):
        name = Path(file).stem.replace('_IM','')
        oriI = dataset_file/'training_set'/file
        oriM = dataset_file/'meshes'/f'{name}.obj'
        if not os.path.exists(oriM):
            print(f'Warning: mesh not found for {name}, skipping.')
            continue
        destI = output_path/'train'/'images'/f'{name}.tiff'
        destM = output_path/'train'/'meshes'/f'{name}.obj'
        shutil.copy2(oriI,destI)
        shutil.copy2(oriM,destM)
    print(f'Train set generated in ->{output_path}')
    
    if is_test:
        for file in tqdm(test_files,desc='Moving'):
            name = Path(file).stem.replace('_IM','')
            oriI = dataset_file/'test_dataset'/file
            oriM = dataset_file/'meshes'/f'{name}.obj'
            if not os.path.exists(oriM):
                print(f'Warning: mesh not found for {name}, skipping.')
                continue
            destI = output_path/'test'/'images'/f'{name}.tiff'
            destM = output_path/'test'/'meshes'/f'{name}.obj'
            shutil.copy2(oriI,destI)
            shutil.copy2(oriM,destM)
        print(f'Test set generated in ->{output_path}')
    else:
        print(f"No test dataset found {dataset_file/'test_dataset'}, no test dataset generated.")