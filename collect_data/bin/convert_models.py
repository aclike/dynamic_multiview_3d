"""
convert_models.py

Converts ShapeNet models from a given directory
into Gazebo models in another given directory.

Usage:
  convert_models.py <synset_name> <shapenet_dir> <gazebo_dir> [--force]
e.g.
  convert_models.py house ~/Downloads/ShapeNetCore.v2/02843684 ~/.gazebo/models --force
"""

import argparse
import os, sys
try:
  from ..scripts import utils
except ValueError:
  sys.path.append('/home/owen/ros/dynamic_multiview_3d/collect_data/scripts')
  import utils
import const
from subprocess import call, check_output
import shutil
from resize_models import define_scale

def generate_gazebo_model_structure(gazebo_dir, model_name, meshtool_output):
  utils.make_dir(os.path.join(gazebo_dir, model_name, 'meshes'))
  utils.make_dir(os.path.join(gazebo_dir, model_name, 'images'))
  with open(os.path.join(gazebo_dir, model_name, 'model.config'), 'w') as f:
    f.write(const.MODEL_CONFIG_CONTENT.format(model_name).replace('\t', '  '))
  with open(os.path.join(gazebo_dir, model_name, 'model.sdf'), 'w') as f:
    ip0, ip1, ip2, ip3, ip4, ip5 = 0, 0, 0, 1.5, 0, 0
    cp0, cp1, cp2, cp3, cp4, cp5 = 0, 0, 0, 1.5, 0, 0
    vp0, vp1, vp2, vp3, vp4, vp5 = 0, 0, 0, 1.5, 0, 0
    ixx, iyy, izz = 0.013, 0.011, 0.0037
    mass, max_vel, min_depth = 1.0, 0.1, 0.001
    scale0, scale1, scale2 = define_scale(meshtool_output)
    uri = 'model://%s/meshes/model.dae' % model_name
    f.write(const.MODEL_SDF_CONTENT.format(
      model_name, ip0, ip1, ip2, ip3, ip4, ip5, ixx, iyy, izz, mass, cp0, cp1,
      cp2, cp3, cp4, cp5, uri, max_vel, min_depth, vp0, vp1, vp2, vp3, vp4, vp5,
      uri, scale0, scale1, scale2
    ).replace('\t', '  '))

def main(synset_name, shapenet_dir, gazebo_dir, force=False):
  for i, dirname in enumerate(os.listdir(shapenet_dir)):
    dirpath = os.path.join(shapenet_dir, dirname)
    if not os.path.isdir(dirpath) or 'models' not in set(os.listdir(dirpath)):
      continue
    model_name = synset_name + str(i)
    shapenet_model_path = os.path.join(dirpath, 'models', 'model_normalized.obj')
    meshtool_output = check_output([
      'meshtool',
      '--load_obj', shapenet_model_path,
      '--print_bounds'
    ])
    generate_gazebo_model_structure(gazebo_dir, model_name, meshtool_output)
    dae_model_path = os.path.join(gazebo_dir, model_name, 'meshes', 'model.dae')
    if force and os.path.isfile(dae_model_path):
      os.remove(dae_model_path)
    call([
      'meshtool',
      '--load_obj', shapenet_model_path,
      '--save_collada', dae_model_path,
    ])
    if 'images' in set(os.listdir(dirpath)):
      for filename in os.listdir(os.path.join(dirpath, 'images')):
        shutil.copy2(os.path.join(dirpath, 'images', filename), os.path.join(gazebo_dir, model_name, 'images'))
      # Fix path references in DAE file
      with open(dae_model_path, 'r+') as f:
        dae_content = f.read()
        f.seek(0)
        f.write(dae_content.replace('./images', '../images/'))
        f.truncate()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('synset_name', type=str)
  parser.add_argument('shapenet_dir', type=str)
  parser.add_argument('gazebo_dir', type=str)
  parser.add_argument('--force', action='store_true')
  args = parser.parse_args()
  main(args.synset_name, args.shapenet_dir, args.gazebo_dir, args.force)
