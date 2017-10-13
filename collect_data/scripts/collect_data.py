#!/usr/bin/env python

"""
collect_data.py

Loads synsets of objects in Gazebo. For each object, runs through multiple
orientations in front of the camera and saves an image of each one.

Usage:
  collect_data.py <num_images> <synset_name> <outfolder>
e.g.
  collect_data.py house 10 /home/owen/ros/dynamic_multiview_3d/collect_data/scripts/test
"""

import rospy
import roslib
roslib.load_manifest('collect_data')
roslib.load_manifest('gazebo_msgs')
roslib.load_manifest('tf')
import collect_data.srv as collect_srv
import sensor_msgs.msg as sensor_msg
import gazebo_msgs.srv as gazebo_srv

import os
import argparse
import utils
import pickle
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tf
import time

GAZEBO_DIR = '/home/owen/.gazebo/models'
MODEL_SDF_BASE = '/home/owen/.gazebo/models/{}/model.sdf'

class DataCollector(object):
  def __init__(self):
    rospy.init_node('data_collector')
    rospy.sleep(2)

    self._service_proxies = {}
    self._call_spawn_object = utils.persistent_service_proxy(
      '/manage_objects/spawn_object', collect_srv.SpawnObject, self._service_proxies)
    self._call_rotate_object = utils.persistent_service_proxy(
      '/manage_objects/rotate_object', collect_srv.RotateObject, self._service_proxies)
    self._call_delete_model = utils.persistent_service_proxy(
      'gazebo/delete_model', gazebo_srv.DeleteModel, self._service_proxies)

    self.latest_img = None
    def img_callback(msg):
      self.latest_img = msg
    rospy.Subscriber('/distorted_camera/distorted_camera/image_raw', sensor_msg.Image, img_callback)

  def collect_data(self, synset_name, num_images=5, outfolder='.', pkl=False):
    start_time, img_count = time.time(), 0
    all_imgs = {}
    for model_name in os.listdir(GAZEBO_DIR):
      if model_name.startswith(synset_name):
        # Set initial properties
        model_sdf_file = MODEL_SDF_BASE.format(model_name)
        pos = [0, 0, 5]  # (x, y, z)
        base_orientation = [1, 0, 0, 0]  # (w, x, y, z)
        orientation, rotation = base_orientation, [0, 0, 0]

        """
        (?)
        Elevation - rotation around y-axis
        Azimuth - rotation around z-axis

        * 360 deg = 6.28 rad
        """

        # Run through a series of orientations, saving an image for each
        imgs = []
        for i in range(num_images):
          # Spawn the object
          self._call_spawn_object(model_name, model_sdf_file, *(pos + orientation))
          rospy.sleep(0.1)
          if pkl:
            imgs.append({'img': self.latest_img, 'orientation': orientation, 'rotation': rotation})
          else:
            self.save_as_img(model_name, self.latest_img, rotation, outfolder)
          img_count += 1

          # Delete the object
          self._call_delete_model(model_name=model_name)

          # Define next orientation
          rotation = [0] + (np.random.rand(2) * 6.28).tolist()
          orientation = self.rotated(base_orientation, rotation)

        all_imgs[model_name] = imgs
    if pkl:
      with open(os.path.join(outfolder, '%s.pkl' % synset_name), 'wb') as f:
        pickle.dump(all_imgs, f)
    time_elapsed_s = time.time() - start_time
    print('[o] time elapsed: %s seconds' % time_elapsed_s)
    print('[o] images collected: %d (avg %.2f s / img)' %
          (img_count, float(time_elapsed_s) / img_count))

  def make_orientations(self):
    """
    Creates a series of orientations, formatted as a list of 3-tuples.
    Each 3-tuple should be in the form (roll, pitch, yaw).
    I think the units are radians.
    """
    return [(0, 0, 0), (0, 0, 0.3), (0, 0, 0.6), (0, 0, 0.9)]  # test values

  def rotated(self, curr_orientation, rotation):
    """
    CURR_ORIENTATION - a (w, x, y, z) list representing the current orientation
    ROTATION - a (r, p, y) list representing the rotation to apply

    returns: a (w, x, y, z) list representing the rotated orientation
    """
    Rq = tf.transformations.quaternion_from_euler(*rotation)
    rotated = tf.transformations.quaternion_multiply(Rq, np.array(
      curr_orientation[1:] + curr_orientation[:1]))
    return list(rotated[-1:]) + list(rotated[:-1])

  def save_as_img(self, model_name, img, rotation, outfolder):
    bridge = CvBridge()
    img.step = img.width * 3
    try:
      cv_img = bridge.imgmsg_to_cv2(img, 'rgb8')
    except CvBridgeError as e:
      print(e); return
    img = np.asarray(cv_img).astype(np.float32)
    outpath = os.path.join(outfolder, '%s_%.1f_%.1f.png' %
                           (model_name, rotation[1], rotation[2]))
    cv2.imwrite(outpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('synset_name', type=str)
  parser.add_argument('num_images', type=int)
  parser.add_argument('outfolder', type=str)
  parser.add_argument('--pkl', action='store_true')
  args = parser.parse_args()

  if not os.path.exists(args.outfolder):
    os.makedirs(args.outfolder)

  # Run data collection
  collector = DataCollector()
  collector.collect_data(args.synset_name, args.num_images, args.outfolder, args.pkl)
