#!/usr/bin/env python

"""
collect_data.py

Loads synsets of objects in Gazebo. For each object, runs through multiple
orientations in front of the camera and saves an image of each one.

Usage:
  collect_data_node.py <synset_name> <num_models> <pairs_per_model> <infolder> <outfolder> [--tfr] [--save_depth] [--save_rate RATE] [--save_images]
e.g.
  collect_data_node.py plane 10000 40 ~/.gazebo/models plane_dataset --tfr --save_depth --save_rate 300
"""

import rospy
import roslib
roslib.load_manifest('collect_data')
roslib.load_manifest('gazebo_msgs')
roslib.load_manifest('geometry_msgs')
roslib.load_manifest('tf')

# from collect_data.srv import RotateObject, SetOrientation, SpawnObject
import collect_data.srv as collect_srv

import gazebo_msgs.msg as gazebo_msg
import geometry_msgs.msg as geometry_msg
import sensor_msgs.msg as sensor_msg
import gazebo_msgs.srv as gazebo_srv
import roslaunch

import pdb
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
import re
import warnings
import random
from subprocess import call

import dyn_mult_view
PLATFORM_LAUNCH = os.path.dirname(dyn_mult_view.__file__) + '/collect_data/launch/platform.launch'
GAZEBO_STARTSTOP_FREQ = 500

def _bytes_feature(value):
    if not isinstance(value, (np.ndarray, list, tuple)):
      value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    if not isinstance(value, (np.ndarray, list, tuple)):
      value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _sorted(dir_contents, synset_name):
  return sorted(dir_contents,
                key=lambda x: int(re.match(r'%s([0-9]+)' % synset_name, x).group(1))
                    if re.match(r'%s([0-9]+)' % synset_name, x) else -10)

class DataCollector(object):
  def __init__(self):
    rospy.init_node('collect_data_node')
    rospy.sleep(2)

    self.launch = None
    self.gazebo_stopped = True
    self.start_gazebo()

    self._service_proxies = {}
    self._call_spawn_object = utils.persistent_service_proxy(
      '/manage_objects/spawn_object', collect_srv.SpawnObject, self._service_proxies)
    self._call_rotate_object = utils.persistent_service_proxy(
      '/manage_objects/rotate_object', collect_srv.RotateObject, self._service_proxies)
    self._call_delete_model = utils.persistent_service_proxy(
      'gazebo/delete_model', gazebo_srv.DeleteModel, self._service_proxies)
    self._call_set_model_state = utils.persistent_service_proxy(
      'gazebo/set_model_state', gazebo_srv.SetModelState, self._service_proxies)
    self._call_get_model_state = utils.persistent_service_proxy(
      'gazebo/get_model_state', gazebo_srv.GetModelState, self._service_proxies)

    self.latest_img, self.latest_depth = None, None
    def img_callback(msg):
      self.latest_img = msg
    rospy.Subscriber('/camera/image_raw', sensor_msg.Image, img_callback)
    def depth_callback(msg):
      self.latest_depth = msg
    rospy.Subscriber('/camera/depth/image_raw', sensor_msg.Image, depth_callback)

  def start_gazebo(self):
    print('Starting Gazebo...')
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    self.launch = roslaunch.parent.ROSLaunchParent(uuid, [PLATFORM_LAUNCH])
    self.launch.start()
    self.gazebo_stopped = False
    rospy.sleep(10)

  def stop_gazebo(self):
    print('Stopping Gazebo...')
    call(['killall', 'gzserver'])
    call(['killall', 'gzclient'])
    time.sleep(10)
    self.launch.shutdown()
    self.gazebo_stopped = True
    rospy.sleep(10)

  def collect_data(self, synset_name, num_models=5, pairs_per_model=5, infolder='.', outfolder='.', pkl=False,
                   tfr=False, save_depth=False, save_rate=None, save_images=False):
    train_dir = os.path.join(outfolder, 'train')
    test_dir = os.path.join(outfolder, 'test')
    validation_dir = os.path.join(outfolder, 'validation')
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    os.makedirs(validation_dir)

    curr_outfolder = test_dir

    writer = None
    curr_tfr_path = os.path.join(curr_outfolder, '%s_0.tfrecords' % synset_name)
    if tfr:
      writer = tf.python_io.TFRecordWriter(curr_tfr_path)
    start_time, pair_count = time.time(), 0
    all_imgs = {}

    gazebo_dir = infolder
    model_sdf_base = gazebo_dir + '/{}/model.sdf'

    model_names = [mn for mn in _sorted(os.listdir(gazebo_dir), synset_name)
                   if mn.startswith(synset_name)]

    for model_index in range(num_models):
      model_name = random.choice(model_names)

      if self.gazebo_stopped:
        self.start_gazebo()

      # Set initial properties
      model_sdf_file = model_sdf_base.format(model_name)
      pos = [0, 0, 5]  # (x, y, z)
      base_orientation = [1, 0, 0, 0]  # (w, x, y, z)
      rotation = [0] + [np.random.rand(), np.random.rand() * 6.28]
      orientation = self.rotated(base_orientation, rotation)

      """
      Elevation - rotation around y-axis
      Azimuth - rotation around z-axis

      * 360 deg = 6.28 rad
      """

      # Spawn the object
      print("spawning {}".format(model_sdf_file))
      self._call_spawn_object(model_name, model_sdf_file, *(pos + orientation))
      rospy.sleep(1.1)
      time.sleep(1.9)  # TODO: all this sleeping necessary?

      imgs = []
      prev_rotation, prev_img, prev_depth = None, None, None
      for i in range(2 * pairs_per_model):
        img, depth = self.latest_img, self.latest_depth
        # if pkl:
        #   warnings.warn('pickling is deprecated', DeprecationWarning)
        #   _info = {'img': img, 'orientation': orientation, 'rotation': rotation}
        #   if save_depth:
        #     _info.update({'depth': depth})
        #   imgs.append(_info)
        if save_images:
          self.save_img(model_name, img, rotation, curr_outfolder)
          if save_depth:
            self.save_img(model_name, depth, rotation, curr_outfolder, depth=True)
        if i % 2 == 1:
          if tfr:
            _prev_img_np = utils.from_sensor_msgs_img(prev_img)
            _img_np = utils.from_sensor_msgs_img(img)
            _prev_depth_np = utils.from_sensor_msgs_img(prev_depth, depth=True)
            _depth_np = utils.from_sensor_msgs_img(depth, depth=True)
            _displacement_np = np.array(rotation[1:]) - np.array(prev_rotation[1:])
            example = tf.train.Example(features=tf.train.Features(feature={
              'image0': _bytes_feature(tf.compat.as_bytes(_prev_img_np.tostring())),
              'image1': _bytes_feature(tf.compat.as_bytes(_img_np.tostring())),
              'depth0': _bytes_feature(tf.compat.as_bytes(_prev_depth_np.tostring())),
              'depth1': _bytes_feature(tf.compat.as_bytes(_depth_np.tostring())),
              'displacement': _float_feature(_displacement_np),  # (elevation, azimuth)
            }))
            writer.write(example.SerializeToString())
          pair_count += 1

        prev_img = img
        prev_depth = depth
        prev_rotation = rotation

        # Define next orientation
        rotation = [0] + [np.random.rand(), np.random.rand() * 6.28]
        orientation = self.rotated(base_orientation, rotation)

        # Rotate the object
        self.set_orientation(model_name, *orientation)
        time.sleep(0.9)
        rospy.sleep(1.1)  # TODO: all this sleeping necessary?

      # Delete the object
      self._call_delete_model(model_name=model_name)
      time.sleep(0.9)
      rospy.sleep(1.1)  # TODO: all this sleeping necessary?

      """
      The tfrecords files, in order, will contain indices [0, save_rate - 1],
      [save_rate, 2 * save_rate - 1], ... and so on so forth.
      """

      if save_rate and (model_index + 1) % save_rate == 0:
        writer.close()
        print('Wrote tfrecords through model %d to %s.' % (model_index, curr_tfr_path))
        if curr_outfolder == test_dir:
          curr_outfolder = validation_dir
        elif curr_outfolder == validation_dir:
          curr_outfolder = train_dir
        curr_tfr_path = os.path.join(curr_outfolder, '%s_%d.tfrecords' %
                                     (synset_name, (model_index + 1) // save_rate))
        writer = tf.python_io.TFRecordWriter(curr_tfr_path)

      all_imgs[model_name] = imgs

      if (model_index + 1) % GAZEBO_STARTSTOP_FREQ == 0:
        self.stop_gazebo()

    # if pkl:
    #   with open(os.path.join(outfolder, '%s.pkl' % synset_name), 'wb') as f:
    #     pickle.dump(all_imgs, f)
    if tfr:
      writer.close()
    time_elapsed_s = time.time() - start_time
    print('[o] time elapsed: %s seconds' % time_elapsed_s)
    print('[o] image pairs collected: %d (avg %.2f s / img)' %
          (pair_count, float(time_elapsed_s) / pair_count))

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

  def set_orientation(self, model_name, w, x, y, z):
    model_state_raw = self._call_get_model_state(model_name=model_name)
    model_state = gazebo_msg.ModelState(
      model_name=model_name, pose=model_state_raw.pose, twist=model_state_raw.twist)
    model_state.pose.orientation = geometry_msg.Quaternion(x, y, z, w)
    self._call_set_model_state(model_state=model_state)
    return True

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
  parser.add_argument('num_models', type=int)
  parser.add_argument('pairs_per_model', type=int)
  parser.add_argument('infolder', type=str)
  parser.add_argument('outfolder', type=str)
  parser.add_argument('--pkl', action='store_true')
  parser.add_argument('--tfr', action='store_true')
  parser.add_argument('--save_depth', action='store_true')
  parser.add_argument('--save_rate', type=int)
  parser.add_argument('--save_images', action='store_true')
  args = parser.parse_args()

  if not os.path.exists(args.outfolder):
    os.makedirs(args.outfolder)

  # Save the command used to run everything
  bin_dir = os.path.join(args.outfolder, 'bin')
  os.makedirs(bin_dir)
  with open(os.path.join(bin_dir, 'run.sh'), 'w') as file:
    file.write(
      'python collect_data_node.py {synset_name} {num_models} {pairs_per_model} {infolder} {outfolder}{pkl}{tfr}{save_depth}{save_rate}{save_images}'.format(
        synset_name=args.synset_name, num_models=args.num_models, pairs_per_model=args.pairs_per_model, infolder=args.infolder, outfolder=args.outfolder,
        pkl=' --pkl' if args.pkl else '', tfr=' --tfr' if args.tfr else '', save_depth=' --save_depth' if args.save_depth else '',
        save_rate=' --save_rate %d' % args.save_rate if args.save_rate else '', save_images=' --save_images' if args.save_images else ''
      )
    )

  # Run data collection
  collector = DataCollector()
  collector.collect_data(args.synset_name, args.num_models, args.pairs_per_model,
                         args.infolder, args.outfolder, args.pkl, args.tfr,
                         args.save_depth, args.save_rate, args.save_images)
