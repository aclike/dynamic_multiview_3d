#!/usr/bin/env python

import argparse
import pickle
import cv2
import os
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

def load_data(data_filepath, outfolder):
  with open(data_filepath, 'rb') as f:
    data = pickle.load(f)
    for model_name, imgs in data.items():
      for img_data in imgs:
        img = img_data['img']
        rot = img_data['rot']

        bridge = CvBridge()
        img.step = img.width * 3
        try:
          cv_img = bridge.imgmsg_to_cv2(img, 'rgb8')
        except CvBridgeError as e:
          print(e)
        img = np.asarray(cv_img).astype(np.float32)
        outpath = os.path.join(outfolder, '%s_%s.png' % (model_name, str(rot)))
        cv2.imwrite(outpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data_filepath', type=str)
  parser.add_argument('outfolder', type=str)
  args = parser.parse_args()

  load_data(args.data_filepath, args.outfolder)
