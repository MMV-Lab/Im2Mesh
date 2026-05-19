import numpy as np
from cellpose import models, core, io, plot, utils
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import argparse

def cellpose_pred(input_path,out_path=None,trained_model=None):

  print("############################################# Starting Predictions ##################################################")
  dir = input_path  #"/mnt/eternus/users/Jair/Im2Mesh_Myversion/100x_multicell/"
  
  if out_path is not None:
    out = out_path #"/mnt/eternus/users/Jair/Im2Mesh_Myversion/100x_multicell_segmentations_cellpose/"
  else:
    out = os.path.join(os.getcwd(),'Cellpose_segmentation_prediction')


  if not os.path.exists(out):
    os.makedirs(out) 


  io.logger_setup() # run this to get printing of progress

  #Check if colab notebook instance has GPU access
  if core.use_gpu()==False:
    raise ImportError("No GPU access, change your runtime")

  
  if trained_model is not None:
    model = models.CellposeModel(gpu=True,pretrained_model=trained_model)
  else:
    model = models.CellposeModel(gpu=True)
  dir = Path(dir)
  if not dir.exists():
    raise FileNotFoundError("directory does not exist")

  # *** change to your image extension ***
  image_ext = ".tif"

  # list all files
  files = natsorted([f for f in dir.glob("*"+image_ext) if "_masks" not in f.name and "_flows" not in f.name])

  if(len(files)==0):
    raise FileNotFoundError("no image files found, did you specify the correct folder and extension?")
  else:
    print(f"{len(files)} images in folder:")

  
  masks_ext = ".png" if image_ext == ".png" else ".tif"
  for i in trange(len(files)):
    f = files[i]
    img = io.imread(f)
    masks, flows, styles = model.eval(img, z_axis=0, channel_axis=3,batch_size=32,do_3D=True, flow3D_smooth=2)
    io.imsave( os.path.join(out,f.stem + masks_ext), masks)
  print("############################################# Predictions Ready ##################################################")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run Cellpose segmentation")

  parser.add_argument('--input_path', type=str, required=True, help='Path to image for prediction')
  parser.add_argument('--out_path', type=str, required=False, default=None, help='Path to prediction output')
  parser.add_argument('--trained_model', type=str, required=False, default=None, help='Path to trained model')
  args = parser.parse_args()

  input_path = args.input_path
  out_path = args.out_path
  trained_model = args.trained_model
  cellpose_pred(input_path,out_path,trained_model)


