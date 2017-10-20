import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

import pdb

from PIL import Image
import imp

import cPickle


def build_tfrecord_input(conf, training=True):
  """Create input tfrecord tensors.

  Args:
    training: training or validation data_files.
    conf: A dictionary containing the configuration for the experiment
  Returns:
    list of tensors corresponding to images, depth-images azimuth and displacement
  Raises:
    RuntimeError: if no files found.
  """

  filenames = gfile.Glob(os.path.join(conf['data_dir'], '*'))
  if not filenames:
    raise RuntimeError('No data_files files found.')

  index = int(np.floor(conf['train_val_split'] * len(filenames)))
  if training:
    filenames = filenames[:index]
  else:
    filenames = filenames[index:]

  if conf['visualize']:
    filenames = gfile.Glob(os.path.join(conf['data_dir'], '*'))
    print 'using input file', filenames
    shuffle = False
  else:
    shuffle = True

  filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  image_name = 'image'
  depthimage_name = 'depth'
  displacement_name = 'displacement'

  features = {
    image_name: tf.FixedLenFeature([1], tf.string),
    depthimage_name: tf.FixedLenFeature([1], tf.string),
    displacement_name: tf.FixedLenFeature([1], tf.float32),
  }

  features = tf.parse_single_example(serialized_example, features=features)

  image = process_image(features, image_name)
  depth_image = process_image(features, image_name, depth_image=True)

  if conf['visualize']:
    num_threads = 1
  else: num_threads = np.min((conf['batch_size'], 10))

  displacement = features[displacement_name]

  [image_batch, depth_image_batch, displacement_batch] = tf.train.batch(
    [image, depth_image, displacement],
    conf['batch_size'],
    num_threads=num_threads,
    capacity=100 * conf['batch_size'])
  return image_batch, depth_image_batch, displacement_batch


def process_image(features, image_name, depth_image=False):
  if depth_image:
    COLOR_CHAN = 1
  else:
    COLOR_CHAN = 3
  ORIGINAL_WIDTH = 128
  ORIGINAL_HEIGHT = 128
  # can optionally specify smaller image dimensions here:
  IMG_WIDTH = 128
  IMG_HEIGHT = 128
  image = tf.decode_raw(features[image_name], tf.uint8)
  image = tf.reshape(image, shape=[1, ORIGINAL_HEIGHT * ORIGINAL_WIDTH * COLOR_CHAN])
  image = tf.reshape(image, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
  if IMG_HEIGHT != IMG_WIDTH:
    raise ValueError('Unequal height and width unsupported')
  crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
  image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
  image = tf.reshape(image, [1, crop_size, crop_size, COLOR_CHAN])
  image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
  image = tf.cast(image, tf.float32) / 255.0
  return image


if __name__ == '__main__':
  # for debugging only:
  conf = {}
  import dyn_mult_view
  DATA_DIR = '/'.join(str.split(dyn_mult_view.__file__, '/')[:-2]) + '/trainingdata/plane_dataset/tfrecs/train'
  conf['schedsamp_k'] = -1  # don't feed ground truth
  conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
  conf['train_val_split'] = 0.95
  conf['batch_size'] = 10
  conf['visualize'] = False

  print '-------------------------------------------------------------------'
  print 'verify current settings!! '
  for key in conf.keys():
    print key, ': ', conf[key]
  print '-------------------------------------------------------------------'

  print 'testing the reader'

  image_batch, depth_image_batch, displacement_batch = build_tfrecord_input(conf, training=True)

  sess = tf.InteractiveSession()
  tf.train.start_queue_runners(sess)
  sess.run(tf.global_variables_initializer())

  from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import comp_single_video

  for i in range(2):
    print 'run number ', i

    image, depth_image, displacement = sess.run([image_batch, depth_image_batch, displacement_batch])

    file_path = '/'.join(str.split(DATA_DIR, '/')[:-1] + ['preview'])
    comp_single_video(file_path, image)

    # show some frames
    for i in range(10):

      print 'displacement'
      print displacement[i]

      image = np.squeeze(image)
      img = np.uint8(255. * image[i, 0])
      img = Image.fromarray(img, 'RGB')
      # img.save(file_path,'PNG')
      img.show()
      print i

      pdb.set_trace()
