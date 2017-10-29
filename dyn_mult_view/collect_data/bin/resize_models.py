"""
resize_models.py

Extracts bounds information from meshes
and resizes the corresponding Gazebo SDFs.

Usage:
  resize_models.py <synset_name> <gazebo_dir>
"""

import argparse
import os, sys
from dyn_mult_view.collect_data.scripts import utils
import subprocess
import re

BOUNDS_PATTERN = r'[\s\S]*Bounds: <<(.*), (.*), (.*)>, <(.*), (.*), ' \
                 r'(.*)>>[\s\S]*'
CENTER_PATTERN = r'[\s\S]*Center: <(.*), (.*), (.*)>[\s\S]*'
FPOINT_PATTERN = r'[\s\S]*Point farthest from center: <(.*), (.*), (.*)> ' \
                 r'at distance of (.*)[\s\S]*'

def define_scale(meshtool_output):
  """Given some information about the model bounds,
  returns an appropriate (x, y, z) scale for the object.
  """
  _, _, fpoint = extract_bounds_info(meshtool_output)
  fdist = fpoint[-1]  # maximal distance from the center
  return (0.65 / fdist + 1e-9,) * 3

def extract_bounds_info(meshtool_output):
  bounds_match = re.match(BOUNDS_PATTERN, meshtool_output)
  center_match = re.match(CENTER_PATTERN, meshtool_output)
  fpoint_match = re.match(FPOINT_PATTERN, meshtool_output)

  if None in (bounds_match, center_match, fpoint_match):
    print('warning: invalid output: %s' % meshtool_output)
    print('the program is about to crash')

  bounds = [
    [float(bounds_match.group(i)) for i in range(1, 4)],
    [float(bounds_match.group(j)) for j in range(4, 7)]
  ]
  center = [float(center_match.group(i)) for i in range(1, 4)]
  fpoint = [float(fpoint_match.group(i)) for i in range(1, 5)]

  return bounds, center, fpoint

def main(synset_name, gazebo_dir):
  for model_name in os.listdir(gazebo_dir):
    if model_name.startswith(synset_name):
      dae_model_path = os.path.join(gazebo_dir, model_name, 'meshes', 'model.dae')
      meshtool_output = subprocess.check_output([
        'meshtool',
        '--load_collada', dae_model_path,
        '--print_bounds'
      ])
      bounds, center, fpoint = extract_bounds_info(meshtool_output)
      print(bounds, center, fpoint)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('synset_name', type=str)
  parser.add_argument('gazebo_dir', type=str)
  args = parser.parse_args()
  main(args.synset_name, args.gazebo_dir)
