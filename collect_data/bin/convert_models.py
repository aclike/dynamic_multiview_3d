"""
convert_models.py

Converts ShapeNet models from a given directory
into Gazebo models in another given directory.
"""

import argparse
import os, sys
try:
  from ..scripts import utils
except ValueError:
  sys.path.append('/home/owen/ros/dynamic_multiview_3d/collect_data/scripts')
  import utils
import const
from subprocess import call

def generate_gazebo_model_structure(gazebo_folder, model_name):
  utils.make_dir(os.path.join(gazebo_folder, model_name, 'meshes'))
  with open(os.path.join(gazebo_folder, model_name, 'model.config'), 'w') as f:
    f.write(const.MODEL_CONFIG_CONTENT.format(model_name).replace('\t', '  '))
  with open(os.path.join(gazebo_folder, model_name, 'model.sdf'), 'w') as f:
    ip0, ip1, ip2, ip3, ip4, ip5 = 0, 0, 0, 1.5, 0, 0
    cp0, cp1, cp2, cp3, cp4, cp5 = 0, 0, 0, 1.5, 0, 0
    vp0, vp1, vp2, vp3, vp4, vp5 = 0, 0, 0, 1.5, 0, 0
    ixx, iyy, izz = 0.013, 0.011, 0.0037
    mass, max_vel, min_depth = 1.0, 0.1, 0.001
    scale0, scale1, scale2 = 1.0, 1.0, 1.0
    uri = 'model://%s/meshes/model.dae' % model_name
    f.write(const.MODEL_SDF_CONTENT.format(
      model_name, ip0, ip1, ip2, ip3, ip4, ip5, ixx, iyy, izz, mass, cp0, cp1,
      cp2, cp3, cp4, cp5, uri, max_vel, min_depth, vp0, vp1, vp2, vp3, vp4, vp5,
      uri, scale0, scale1, scale2
    ).replace('\t', '  '))

def main(args):
  for i, dirname in enumerate(os.listdir(args.shapenet_folder)):
    dirpath = os.path.join(args.shapenet_folder, dirname)
    if not os.path.isdir(dirpath) or 'models' not in set(os.listdir(dirpath)):
      continue
    if 'images' in set(os.listdir(dirpath)):
      print('Omitting %s because it requires images.' % dirpath)
      continue
    model_name = args.synset_name + str(i)
    generate_gazebo_model_structure(args.gazebo_folder, model_name)
    shapenet_model_path = os.path.join(dirpath, 'models', 'model_normalized.obj')
    dae_model_path = os.path.join(args.gazebo_folder, model_name, 'meshes', 'model.dae')
    call([
      'meshtool',
      '--load_obj', shapenet_model_path,
      '--save_collada', dae_model_path,
    ])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('synset_name', type=str)
  parser.add_argument('shapenet_folder', type=str)
  parser.add_argument('gazebo_folder', type=str)
  args = parser.parse_args()
  main(args)
