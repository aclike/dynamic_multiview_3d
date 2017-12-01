import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

import pdb
import matplotlib.pyplot as plt
from PIL import Image
import imp

import cPickle


class Build_tfrecord_input():
    def __init__(self, conf, training=True):
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

        if 'test_mode' in conf:
            filenames = gfile.Glob(os.path.join(conf['data_dir'], '*'))
            print 'using input file', filenames
            shuffle = False
        else:
            shuffle = True

        print 'shuffle',shuffle

        filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        image0_name = 'image0'
        image0_mask0_name = 'image0_mask0'
        image0_mask1_name = 'image0_mask1'
        image1_name = 'image1'
        image1_only0_name = 'image1_only0'
        image1_only1_name = 'image1_only1'
        image1_mask0_name = 'image1_mask0'
        image1_mask1_name = 'image1_mask1'
        depth0_name = 'depth0'
        depth1_name = 'depth1'
        depth1_only0_name = 'depth1_only0'
        depth1_only1_name = 'depth1_only1'
        displacement_name = 'displacement'

        features = {
            image0_name: tf.FixedLenFeature([1], tf.string),
            image0_mask0_name: tf.FixedLenFeature([1], tf.string),
            image0_mask1_name: tf.FixedLenFeature([1], tf.string),
            image1_name: tf.FixedLenFeature([1], tf.string),
            image1_only0_name: tf.FixedLenFeature([1], tf.string),
            image1_only1_name: tf.FixedLenFeature([1], tf.string),
            image1_mask0_name: tf.FixedLenFeature([1], tf.string),
            image1_mask1_name: tf.FixedLenFeature([1], tf.string),
            depth0_name: tf.FixedLenFeature([1], tf.string),
            depth1_name: tf.FixedLenFeature([1], tf.string),
            depth1_only0_name: tf.FixedLenFeature([1], tf.string),
            depth1_only1_name: tf.FixedLenFeature([1], tf.string),

            displacement_name: tf.FixedLenFeature([2], tf.float32),
        }

        features = tf.parse_single_example(serialized_example, features=features)

        image0 = process_image(features, image0_name)
        image0_mask0 = process_image(features, image0_mask0_name, single_channel=True)
        image0_mask1 = process_image(features, image0_mask1_name, single_channel=True)
        image1 = process_image(features, image1_name)
        image1_only0 = process_image(features, image1_only0_name)
        image1_only1 = process_image(features, image1_only1_name)
        image1_mask0 = process_image(features, image1_mask0_name, single_channel=True)
        image1_mask1 = process_image(features, image1_mask1_name, single_channel=True)
        depth0 = process_image(features, depth0_name, single_channel=True)
        depth1 = process_image(features, depth1_name, single_channel=True)
        depth1_only0 = process_image(features, depth1_only0_name, single_channel=True)
        depth1_only1 = process_image(features, depth1_only1_name, single_channel=True)

        displacement = features[displacement_name]

        if 'test_mode' in conf:
            num_threads = 1
        else: num_threads = np.min((conf['batch_size'], 10))
        print 'using {} threads'.format(num_threads)

        [self.image0,
         self.image0_mask0,
         self.image0_mask1,
         self.image1,
         self.image1_only0,
         self.image1_only1,
         self.image1_mask0,
         self.image1_mask1,
         self.depth0,
         self.depth1,
         self.depth1_only0,
         self.depth1_only1,
         self.displacement
         ] = tf.train.batch([image0,
                             image0_mask0,
                             image0_mask1,
                             image1,
                             image1_only0,
                             image1_only1,
                             image1_mask0,
                             image1_mask1,
                             depth0,
                             depth1,
                             depth1_only0,
                             depth1_only1,
                             displacement],
                            conf['batch_size'],
                            num_threads=num_threads,
                            capacity=100 * conf['batch_size'])

def process_image(features, image_name, single_channel=False):
    if single_channel:
        COLOR_CHAN = 1
    else:
        COLOR_CHAN = 3

    ORIGINAL_WIDTH = 128
    ORIGINAL_HEIGHT = 128

    IMG_WIDTH = 128
    IMG_HEIGHT = 128

    # can optionally specify smaller image dimensions here:
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
    image = tf.reshape(image, [crop_size, crop_size, COLOR_CHAN])
    return image


def check_tfrecs():
    # for debugging only:
    conf = {}
    import dyn_mult_view
    # DATA_DIR = '/'.join(str.split(dyn_mult_view.__file__, '/')[:-2]) + '/trainingdata/plane_dataset2/tfrecords/train'
    DATA_DIR = "/home/frederik/Documents/catkin_ws/src/dynamic_multiview_3d/trainingdata/multicardataset/train"

    conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
    conf['train_val_split'] = 0.95
    conf['batch_size'] = 64
    conf['visualize'] = False

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    print 'testing the reader'

    b = Build_tfrecord_input(conf, training=True)

    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    for i in range(1):
        print 'run number ', i
        image0, \
        image0_mask0, \
        image0_mask1, \
        image1, \
        image1_only0, \
        image1_only1, \
        image1_mask0, \
        image1_mask1, \
        depth0, \
        depth1, \
        depth1_only0, \
        depth1_only1, \
        displacement = sess.run([
            b.image0,
            b.image0_mask0,
            b.image0_mask1,
            b.image1,
            b.image1_only0,
            b.image1_only1,
            b.image1_mask0,
            b.image1_mask1,
            b.depth0,
            b.depth1,
            b.depth1_only0,
            b.depth1_only1,
            b.displacement
        ])

        image0 = np.squeeze(image0)
        image0_mask0 = np.squeeze(image0_mask0)
        image0_mask1 = np.squeeze(image0_mask1)
        image1 = np.squeeze(image1)
        image1_only0 = np.squeeze(image1_only0)
        image1_only1 = np.squeeze(image1_only1)
        image1_mask0 = np.squeeze(image1_mask0)
        image1_mask1 = np.squeeze(image1_mask1)
        depth0 = np.squeeze(depth0)
        depth1 = np.squeeze(depth1)
        depth1_only0 = np.squeeze(depth1_only0)
        depth1_only1 = np.squeeze(depth1_only1)

            # show some frames
        for b in range(10):
            print 'batchind', b
            f = plt.figure()

            iplt = 0
            iplt +=1
            plt.subplot(2, 4, iplt)
            plt.imshow(image0[b])
            plt.axis('off')

            iplt += 1
            plt.subplot(2, 4, iplt)
            plt.imshow(image1[b])
            plt.axis('off')

            iplt += 1
            plt.subplot(2, 4, iplt)
            plt.imshow(image1_only0[b])
            plt.axis('off')

            iplt += 1
            plt.subplot(2, 4, iplt)
            plt.imshow(image1_only1[b])
            plt.axis('off')


            ## depth
            iplt += 1
            plt.subplot(2, 4, iplt)
            plt.imshow(depth0[b])
            plt.axis('off')

            iplt += 1
            plt.subplot(2, 4, iplt)
            plt.imshow(depth1[b])
            plt.axis('off')

            iplt += 1
            plt.subplot(2, 4, iplt)
            plt.imshow(depth1_only0[b])
            plt.axis('off')

            iplt += 1
            plt.subplot(2, 4, iplt)
            plt.imshow(depth1_only1[b])
            plt.axis('off')

            plt.draw()

            f = plt.figure()
            iplt = 0
            iplt += 1
            plt.subplot(1, 4, iplt)
            plt.imshow(image0_mask0[b])
            plt.axis('off')

            iplt += 1
            plt.subplot(1, 4, iplt)
            plt.imshow(image0_mask1[b])
            plt.axis('off')

            iplt += 1
            plt.subplot(1, 4, iplt)
            plt.imshow(image1_mask0[b])
            plt.axis('off')

            iplt += 1
            plt.subplot(1, 4, iplt)
            plt.imshow(image1_mask1[b])
            plt.axis('off')

            plt.show()


if __name__ == '__main__':
    check_tfrecs()
