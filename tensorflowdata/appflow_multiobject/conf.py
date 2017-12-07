import os
current_dir = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append("/home/brandon/dynamic_multiview_3d/dyn_multi_view/multi_view_model")

from multiobject_appflow import MultiObjectAppFlow

# tf record data location:
import dyn_mult_view
DATA_DIR = '/home/febert/Documents/courses/dynamic_multiview_3d/trainingdata/multicardataset/train'

# local output directory
OUT_DIR = current_dir + '/modeldata'

configuration = {
'experiment_name': 'appflow_multiobject',
'data_dir': DATA_DIR,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir,   #'directory for writing summary.' ,
'num_iterations': 200000,   #'number of training iterations.' ,
'batch_size':64,
'learning_rate':1e-4,
'train_val_split':0.95,
'model':MultiObjectAppFlow
}
