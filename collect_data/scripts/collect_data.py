"""
collect_data.py

Loads synsets of objects in Gazebo. For each object, runs through multiple
rotations in front of the camera and saves an image of each one.
"""

import rospy
import roslib
roslib.load_manifest('collect_data')
import collect_data.srv as collect_srv
import sensor_msgs.msg as sensor_msg

import os
import argparse
import utils
import pickle

GAZEBO_DIR = '/home/owen/.gazebo/models'
MODEL_SDF_BASE = '/home/owen/.gazebo/models/{}/model.sdf'

class DataCollector(object):
  def __init__(self):
    rospy.init_node('data_collector')
    rospy.sleep(2)

    self._service_proxies = {}
    self._call_spawn_object = utils.persistent_service_proxy(
      '/manage_objects/spawn_object', collect_srv.SpawnObject, self._service_proxies)
    self._call_set_object_rotation = utils.persistent_service_proxy(
      '/manage_objects/set_object_rotation', collect_srv.SetObjectRotation, self._service_proxies)

    self.latest_img = None
    def img_callback(msg):
      self.latest_img = msg
    rospy.Subscriber('/distorted_camera/image_raw', sensor_msg.Image, img_callback)

  def collect_data(self, synset_name, outfolder='.'):
    all_imgs = []
    for model_name in os.listdir(GAZEBO_DIR):
      if model_name.startswith(synset_name):
        # Spawn the object
        model_sdf_file = MODEL_SDF_BASE.format(model_name)
        pos = [0, 0, 0]  # (x, y, z)
        rot = [1, 0, 0, 0]  # (w, x, y, z)
        self._call_spawn_object(model_name, model_sdf_file, *(pos + rot))

        # Run through a series of rotations, saving an image for each
        imgs = []
        rotations = self.make_rotations()
        for r, p, y in rotations:
          success = self._call_set_object_rotation(model_name, r, p, y)
          if not success:
            print('WARNING: rotation update unsuccessful!')
          rospy.sleep(0.1)
          imgs.append({'img': self.latest_img, 'meta': [model_name, r, p, y]})
        all_imgs.append(imgs)
    with open(os.path.join(outfolder, '%s.pkl' % synset_name), 'wb') as f:
      pickle.dump(all_imgs, f)

  def make_rotations(self):
    """
    Creates a series of rotations, formatted as a list of 3-tuples.
    Each 3-tuple should be in the form (roll, pitch, yaw).
    """
    return [(0, 0, 0), (0, 0, 0)]

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('synset_name', type=str)
  parser.add_argument('outfolder', type=str)
  args = parser.parse_args()

  # Run data collection
  collector = DataCollector()
  collector.collect_data(args.synset_name, args.outfolder)
