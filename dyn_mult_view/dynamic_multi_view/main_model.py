import time
import os
import sys

import dyn_mult_view.mv3d.utils.realtime_renderer as rtr
from dyn_mult_view.mv3d.utils.tf_utils import *
from dyn_mult_view.dynamic_multi_view.utils.read_tf_record_wristrot import build_tfrecord_input

class Base_Prediction_Model():
  def __init__(self,
               conf,
               load_tfrec=True):

    self.conf = conf
    self.batch_size = 64
    self.image_shape = [128, 128, 3]
    self.max_iter = 1000000
    self.start_iter = 0

    self.log_folder = "logs/nobg_nodm"
    self.train_samples_folder = "samples/nobg_nodm/train"
    self.test_samples_folder = "samples/nobg_nodm/test"
    self.snapshots_folder = "snapshots/nobg_nodm"

    self.rend = rtr.RealTimeRenderer(self.batch_size)
    self.rend.load_model_names("data/cars_training.txt")

    self.test_images1, self.test_images2, \
    self.test_dm2, self.test_labels = load_test_set(True, False)



    self.train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")

    if not load_tfrec:
      # pairs of images: the first one is the starting image the second is the image which
      # shall be inferred
      self.images = tf.placeholder(tf.float32,
                                   [self.batch_size, 2] + self.image_shape,
                                   name='input_images')

      self.depth_images = tf.placeholder(tf.float32,
                                   [self.batch_size, 2] + self.image_shape,
                                   name='input_images')

      self.labels = tf.placeholder(tf.float32, [self.batch_size, 5],
                                   name='labels')

    else:
      train_image, train_depth_image, train_disp = build_tfrecord_input(conf, training=True)
      test_image, test_depth_image, test_disp = build_tfrecord_input(conf, training=False)

      self.train_image, self.depth_image, self.disp = tf.cond(self.train_cond > 0,  # if 1 use trainigbatch else validation batch
                                      lambda: [train_image, train_depth_image, train_disp],
                                      lambda: [test_image, test_depth_image, test_disp])

    self.buildModel()

  def buildModel(self):

    image0 = self.images[:,0]
    gtruth_image = self.images[:,1]

    # convolutional encoder
    e0 = lrelu(conv2d_msra(image0, 32, 5, 5, 2, 2, "e0"))
    e0_0 = lrelu(conv2d_msra(e0, 32, 5, 5, 1, 1, "e0_0"))
    e1 = lrelu(conv2d_msra(e0_0, 32, 5, 5, 2, 2, "e1"))
    e1_0 = lrelu(conv2d_msra(e1, 32, 5, 5, 1, 1, "e1_0"))
    e2 = lrelu(conv2d_msra(e1_0, 64, 5, 5, 2, 2, "e2"))
    e2_0 = lrelu(conv2d_msra(e2, 64, 5, 5, 1, 1, "e2_0"))
    e3 = lrelu(conv2d_msra(e2_0, 128, 3, 3, 2, 2, "e3"))
    e3_0 = lrelu(conv2d_msra(e3, 128, 3, 3, 1, 1, "e3_0"))
    e4 = lrelu(conv2d_msra(e3_0, 256, 3, 3, 2, 2, "e4"))
    e4_0 = lrelu(conv2d_msra(e4, 256, 3, 3, 1, 1, "e4_0"))
    e4r = tf.reshape(e4_0, [self.batch_size, 4096])
    e5 = lrelu(linear_msra(e4r, 4096, "fc1"))

    # angle processing
    a0 = lrelu(linear_msra(self.labels, 64, "a0"))
    a1 = lrelu(linear_msra(a0, 64, "a1"))
    a2 = lrelu(linear_msra(a1, 64, "a2"))

    concated = tf.concat(axis=1, values=[e5, a2])

    # joint processing
    a3 = lrelu(linear_msra(concated, 4096, "a3"))
    a4 = lrelu(linear_msra(a3, 4096, "a4"))
    a5 = lrelu(linear_msra(a4, 4096, "a5"))
    a5r = tf.reshape(a5, [self.batch_size, 4, 4, 256])

    # convolutional decoder
    d4 = lrelu(deconv2d_msra(a5r, [self.batch_size, 8, 8, 128],
                             3, 3, 2, 2, "d4"))
    d4_0 = lrelu(conv2d_msra(d4, 128, 3, 3, 1, 1, "d4_0"))
    d3 = lrelu(deconv2d_msra(d4_0, [self.batch_size, 16, 16, 64],
                             3, 3, 2, 2, "d3"))
    d3_0 = lrelu(conv2d_msra(d3, 64, 5, 5, 1, 1, "d3_0"))
    d2 = lrelu(deconv2d_msra(d3_0, [self.batch_size, 32, 32, 32],
                             5, 5, 2, 2, "d2"))
    d2_0 = lrelu(conv2d_msra(d2, 64, 5, 5, 1, 1, "d2_0"))
    d1 = lrelu(deconv2d_msra(d2_0, [self.batch_size, 64, 64, 32],
                             5, 5, 2, 2, "d1"))
    d1_0 = lrelu(conv2d_msra(d1, 32, 5, 5, 1, 1, "d1_0"))
    self.gen = tf.nn.tanh(deconv2d_msra(d1_0,
                                        [self.batch_size, 128, 128, 3],
                                        5, 5, 2, 2, "d0"))

    self.loss = euclidean_loss(self.gen, gtruth_image)
    self.training_summ = tf.summary.scalar("training_loss", self.loss)

    self.t_vars = tf.trainable_variables()
    self.saver = tf.train.Saver(max_to_keep=20)

  def build_loss(self):

    summaries = []
    self.loss = euclidean_loss(self.gen, self.images2)
    summaries.append(tf.summary.scalar("training_loss", self.loss))

    self.train_op = tf.train.AdamOptimizer(self.conf['learning_rate']).minimize(self.loss)

    self.summ_op = tf.summary.merge(summaries)

  def generate_sample_set(self, path, im1, im2, gen, iter_num):
    save_images(gen, [8, 8], path + "/output_%s.png" % (iter_num))
    save_images(np.array(im2), [8, 8],
                path + '/tr_gt_%s.png' % (iter_num))
    save_images(np.array(im1), [8, 8],
                path + '/tr_input_%s.png' % (iter_num))


  def visualize(self, sess, global_iter):
    self.test_iter = 19
    sm_path = os.path.join(self.test_samples_folder,
                           str(global_iter).zfill(8))
    if not os.path.exists(sm_path):
      os.mkdir(sm_path)
    local_loss = 0.0
    for i in range(0, self.test_iter):
      batch_images1 = self.test_images1[i * self.batch_size:
      (i + 1) * self.batch_size]
      batch_images2 = self.test_images2[i * self.batch_size:
      (i + 1) * self.batch_size]
      batch_labels = self.test_labels[i * self.batch_size:
      (i + 1) * self.batch_size]
      output = sess.run([self.gen, self.loss],
                             feed_dict={
                               self.images: batch_images1,
                               self.images2: batch_images2,
                               self.labels: batch_labels})

      self.generate_sample_set(sm_path, batch_images1,
                               batch_images2, output[0], i)
      local_loss += float(output[1])

    total_loss = local_loss / self.test_iter
    print("[i: %s] [test loss: %.6f]" %
          (global_iter, total_loss))

    # if self.writer is not None:
    #   log_value(self.writer, total_loss, 'test_loss', global_iter)


global_start_time = time.time()
