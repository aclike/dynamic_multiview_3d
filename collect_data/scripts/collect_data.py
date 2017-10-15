#!/usr/bin/env python

"""
collect_data.py

Loads synsets of objects in Gazebo. For each object, runs through multiple
orientations in front of the camera and saves an image of each one.

Usage:
  collect_data.py <num_images> <synset_name> <outfolder> [--itr] [--tfr] [--save_depth]
e.g.
  collect_data.py house 10 house_dataset --tfr --save_depth
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
import numpy as np
import tf as xf
import time
import tensorflow as tf
from cv_bridge import CvBridge
import Image

GAZEBO_DIR = '/home/owen/.gazebo/models'
MODEL_SDF_BASE = '/home/owen/.gazebo/models/{}/model.sdf'

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

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

    self.latest_img, self.latest_depth = None, None
    def img_callback(msg):
      self.latest_img = msg
    rospy.Subscriber('/camera/image_raw', sensor_msg.Image, img_callback)
    def depth_callback(msg):
      self.latest_depth = msg
    rospy.Subscriber('/camera/depth/image_raw', sensor_msg.Image, depth_callback)

  def collect_data(self, synset_name, num_images=5, outfolder='.', pkl=False, tfr=False, save_depth=False):
    writer = None
    if tfr:
      writer = tf.python_io.TFRecordWriter(os.path.join(outfolder, '%s.tfrecords' % synset_name))
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
          img, depth = self.latest_img, self.latest_depth
          if pkl:
            _info = {'img': img, 'orientation': orientation, 'rotation': rotation}
            if save_depth:
              _info.update({'depth': depth})
            imgs.append(_info)
          else:
            self.save_img(model_name, img, rotation, outfolder)
            if save_depth:
              self.save_img(model_name, depth, rotation, outfolder, depth=True)
          if tfr:
            _img_np = utils.from_sensor_msgs_img(img)
            _depth_np = utils.from_sensor_msgs_img(depth, depth=True)
            example = tf.train.Example(features=tf.train.Features(feature={
              'image': _bytes_feature(tf.compat.as_bytes(_img_np.tostring())),
              'depth': _bytes_feature(tf.compat.as_bytes(_depth_np.tostring())),
              'elevation': _float_feature(rotation[1]),
              'azimuth': _float_feature(rotation[2]),
            }))
            writer.write(example.SerializeToString())
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
    if tfr:
      writer.close()
    time_elapsed_s = time.time() - start_time
    print('[o] time elapsed: %s seconds' % time_elapsed_s)
    print('[o] images collected: %d (avg %.2f s / img)' %
          (img_count, float(time_elapsed_s) / img_count))

  def rotated(self, curr_orientation, rotation):
    """
    CURR_ORIENTATION - a (w, x, y, z) list representing the current orientation
    ROTATION - a (r, p, y) list representing the rotation to apply

    returns: a (w, x, y, z) list representing the rotated orientation
    """
    Rq = xf.transformations.quaternion_from_euler(*rotation)
    rotated = xf.transformations.quaternion_multiply(Rq, np.array(
      curr_orientation[1:] + curr_orientation[:1]))
    return list(rotated[-1:]) + list(rotated[:-1])

  def save_img(self, model_name, img, rotation, outfolder, depth=False):
    outpath = os.path.join(outfolder, '%s%s_%.1f_%.1f.png' %
                           ('depth_' if depth else '', model_name, rotation[1], rotation[2]))
    if depth:
      _img = np.copy(CvBridge().imgmsg_to_cv2(img))
      _img[np.isnan(_img)] = 0
      _img = (255.0 / _img.max() * (_img - _img.min())).astype(np.uint8)
      Image.fromarray(_img).save(outpath)
    else:
      img = utils.from_sensor_msgs_img(img, depth=False)
      cv2.imwrite(outpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('synset_name', type=str)
  parser.add_argument('num_images', type=int)
  parser.add_argument('outfolder', type=str)
  parser.add_argument('--pkl', action='store_true')
  parser.add_argument('--tfr', action='store_true')
  parser.add_argument('--save_depth', action='store_true')
  args = parser.parse_args()

  if not os.path.exists(args.outfolder):
    os.makedirs(args.outfolder)

  # Run data collection
  collector = DataCollector()
  collector.collect_data(args.synset_name, args.num_images, args.outfolder, args.pkl, args.tfr, args.save_depth)
