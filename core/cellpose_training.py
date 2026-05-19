import numpy as np
from cellpose import models, core, io, plot, utils, train
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import argparse

def cellpose_training(train_dir,test_dir=None,masks_ext="_seg",n_epochs=500,learning_rate =1e-5,weight_decay = 0.1,batch_size = 1,model_name='multicell_seg',min_train_masks=5):

  print("############################################# Starting Training ##################################################")

  io.logger_setup() # run this to get printing of progress

  #Check if colab notebook instance has GPU access
  if core.use_gpu()==False:
    raise ImportError("No GPU access, change your runtime")

  model = models.CellposeModel(gpu=True)

  if not Path(train_dir).exists():
    raise FileNotFoundError("directory does not exist")

  # list all files
  files = [f for f in Path(train_dir).glob("*") if "_masks" not in f.name and "_flows" not in f.name and "_seg" not in f.name]

  if(len(files)==0):
    raise FileNotFoundError("no files found, did you specify the correct folder and extension?")
  else:
    print(f"{len(files)} files in folder:")

  # get files
  output = io.load_train_test_data(train_dir, test_dir, mask_filter=masks_ext)

  train_data, train_labels, _, test_data, test_labels, _ = output

  new_model_path, train_losses, test_losses = train.train_seg(model.net,
                                                              train_data=train_data,
                                                              train_labels=train_labels,
                                                              batch_size=batch_size,
                                                              n_epochs=n_epochs,
                                                              learning_rate=learning_rate,
                                                              weight_decay=weight_decay,
                                                              nimg_per_epoch=max(2, len(train_data)), 
                                                              model_name=model_name,
                                                              min_train_masks=min_train_masks)


  print("############################################# Training Finish ##################################################")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run Cellpose training")

  parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset')
  parser.add_argument('--test_dir', type=str, required=False,default=None, help='Path to test dataset')
  parser.add_argument('--masks_ext', type=str, required=False,default="_seg", help='mask extention')
  parser.add_argument('--n_epochs', type=int, required=False,default=500, help='Epoch number')
  parser.add_argument('--learning_rate', type=float, required=False,default=1e-5, help='Learning rate')
  parser.add_argument('--weight_decay', type=float, required=False,default=0.1, help='Weight decay')
  parser.add_argument('--batch_size', type=int, required=False,default=8, help='Batch size')
  parser.add_argument('--model_name', type=str, required=False,default='multicell_seg', help='Model name')
  parser.add_argument('--min_train_masks', type=int, required=False,default=1, help='Minimun of instances on GT')

  args = parser.parse_args()

  train_dir = args.train_dir
  test_dir = args.test_dir
  masks_ext = args.masks_ext
  n_epochs = args.n_epochs
  learning_rate = args.learning_rate
  weight_decay = args.weight_decay
  batch_size = args.batch_size
  model_name = args.model_name
  min_train_masks = args.min_train_masks

  cellpose_training(train_dir,test_dir,masks_ext,n_epochs,learning_rate,weight_decay,batch_size,model_name,min_train_masks)


