import time
import os
import sys

import dyn_mult_view.mv3d.utils.realtime_renderer as rtr
from dyn_mult_view.mv3d.utils.tf_utils import *
from dyn_mult_view.multi_view_model.utils.read_tf_records import build_tfrecord_input

import pdb
import matplotlib.pyplot as plt

from utils.read_tf_records_multobj import Build_tfrecord_input

class MultiObjectAppFlow():
    def __init__(self,
                 conf,
                 load_tfrec=True,
                 build_loss = True):
        self.conf = conf
        self.batch_size = conf['batch_size']
        self.image_shape = [128, 128, 3]
        self.scalar_imshape = [128, 128, 1]

        self.max_iter = 1000000
        self.start_iter = 0

        self.train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")

        train_b = Build_tfrecord_input(conf, training=True)
        val_b = Build_tfrecord_input(conf, training=False)

        self.image0       ,\
        self.image0_mask0 ,\
        self.image0_mask1 ,\
        self.image1       ,\
        self.image1_only0 ,\
        self.image1_only1 ,\
        self.image1_mask0 ,\
        self.image1_mask1 ,\
        self.depth0       ,\
        self.depth1       ,\
        self.depth1_only0 ,\
        self.depth1_only1 ,\
        self.displacement  \
        = tf.cond(self.train_cond > 0,  # if 1 use trainigbatch else validation batch
                    lambda:
                    [train_b.image0,
                    train_b.image0_mask0,
                    train_b.image0_mask1,
                    train_b.image1,
                    train_b.image1_only0,
                    train_b.image1_only1,
                    train_b.image1_mask0,
                    train_b.image1_mask1,
                    train_b.depth0,
                    train_b.depth1,
                    train_b.depth1_only0,
                    train_b.depth1_only1,
                    train_b.displacement],
                    lambda:
                  [val_b.image0,
                   val_b.image0_mask0,
                   val_b.image0_mask1,
                   val_b.image1,
                   val_b.image1_only0,
                   val_b.image1_only1,
                   val_b.image1_mask0,
                   val_b.image1_mask1,
                   val_b.depth0,
                   val_b.depth1,
                   val_b.depth1_only0,
                   val_b.depth1_only1,
                   val_b.displacement])

        self.buildModel()

        if build_loss:
            self.build_loss()

    def image_preprocessing(self, input, scope):
        with tf.variable_scope(scope):
            e0 = lrelu(conv2d_msra(input, 32, 5, 5, 2, 2, "e0"))  # 64x64
            e0_0 = lrelu(conv2d_msra(e0, 32, 5, 5, 1, 1, "e0_0"))
            e1 = lrelu(conv2d_msra(e0_0, 32, 5, 5, 2, 2, "e1"))  # 32x32
            e1_0 = lrelu(conv2d_msra(e1, 32, 5, 5, 1, 1, "e1_0"))
            e2 = lrelu(conv2d_msra(e1_0, 64, 5, 5, 2, 2, "e2"))  # 16x16

        return e2

    def decode_flow(self, src_img, input, scope):
        flow_field_channels = 2
        with tf.variable_scope(scope):
            d2 = lrelu(deconv2d_msra(input, [self.batch_size, 32, 32, 32],  # 32x32
                                     5, 5, 2, 2, "d2"))
            d2_0 = lrelu(conv2d_msra(d2, 64, 5, 5, 1, 1, "d2_0"))
            d1 = lrelu(deconv2d_msra(d2_0, [self.batch_size, 64, 64, 32],  # 64x64
                                     5, 5, 2, 2, "d1"))
            d1_0 = lrelu(conv2d_msra(d1, 32, 5, 5, 1, 1, "d1_0"))

            flow_field = deconv2d_msra(d1_0, [self.batch_size, 128, 128, flow_field_channels], 5, 5, 2, 2, "d0")
            warp_pts = warp_pts_layer(flow_field)
            gen = resample_layer(src_img, warp_pts)

        return gen

    def decode_direct(self, input, scope, num_outputs=1):
        channels = 1
        with tf.variable_scope(scope):
            d2 = lrelu(deconv2d_msra(input, [self.batch_size, 32, 32, 32],  # 32x32
                                     5, 5, 2, 2, "d2"))
            d2_0 = lrelu(conv2d_msra(d2, 64, 5, 5, 1, 1, "d2_0"))
            d1 = lrelu(deconv2d_msra(d2_0, [self.batch_size, 64, 64, 32],  # 64x64
                                     5, 5, 2, 2, "d1"))
            d1_0 = lrelu(conv2d_msra(d1, 32, 5, 5, 1, 1, "d1_0"))

            pre_tanh = deconv2d_msra(d1_0, [self.batch_size, 128, 128, channels], 5, 5, 2, 2, "d0")  # 128x128

            gen = tf.nn.tanh(pre_tanh)

        return gen


    def buildModel(self):
        # convolutional encoder
        concat_list = []
        if 'use_color' in self.conf:
            concat_list.append(self.image_preprocessing(self.image0, 'pre_image0_f'))
        if 'use_depth' in self.conf:
            concat_list.append(self.image_preprocessing(self.depth0, 'pre_dimage0_f'))

        concat_list.append(self.image_preprocessing(self.image0_mask0, 'pre_mask0_ob0'))
        concat_list.append(self.image_preprocessing(self.image0_mask1, 'pre_mask0_ob1'))

        comb_enc = tf.concat(axis=3, values=concat_list)

        e2_0 = lrelu(conv2d_msra(comb_enc, 64, 5, 5, 1, 1, "e2_0"))
        e3 = lrelu(conv2d_msra(e2_0, 128, 3, 3, 2, 2, "e3"))  #8x8
        e3_0 = lrelu(conv2d_msra(e3, 128, 3, 3, 1, 1, "e3_0"))
        e4 = lrelu(conv2d_msra(e3_0, 256, 3, 3, 2, 2, "e4"))  #4x4
        e4_0 = lrelu(conv2d_msra(e4, 256, 3, 3, 1, 1, "e4_0"))

        # angle processing
        a0 = lrelu(linear_msra(self.displacement, 64, "a0"))
        a1 = lrelu(linear_msra(a0, 64, "a1"))
        a2 = lrelu(linear_msra(a1, 64, "a2"))

        if 'fully_conv' in self.conf:
            smear = tf.reshape(a2, [int(self.batch_size), 1, 1, int(a2.get_shape()[1])])
            smear = tf.tile(smear, [1, int(e4_0.get_shape()[1]), int(e4_0.get_shape()[2]), 1])
            concated = tf.concat(axis=3, values=[e4_0, smear])

            e4_1 = lrelu(conv2d_msra(concated, 256, 3, 3, 1, 1, "e4_1"))
            a5r = lrelu(conv2d_msra(e4_1, 256, 3, 3, 1, 1, "e4_2"))
        else:
            e4r = tf.reshape(e4_0, [self.batch_size, 4096])
            e5 = lrelu(linear_msra(e4r, 4096, "fc1"))

            concated = tf.concat(axis=1, values=[e5, a2])

            # joint processing
            a3 = lrelu(linear_msra(concated, 4096, "a3"))
            a4 = lrelu(linear_msra(a3, 4096, "a4"))
            a5 = lrelu(linear_msra(a4, 4096, "a5"))
            a5r = tf.reshape(a5, [self.batch_size, 4, 4, 256])

        # joint convolutional decoder
        d4 = lrelu(deconv2d_msra(a5r, [self.batch_size, 8, 8, 128],   #8x8
                                 3, 3, 2, 2, "d4"))
        d4_0 = lrelu(conv2d_msra(d4, 128, 3, 3, 1, 1, "d4_0"))
        d3 = lrelu(deconv2d_msra(d4_0, [self.batch_size, 16, 16, 64],  #16x16
                                 3, 3, 2, 2, "d3"))
        num_decode = 0
        if 'use_color' in self.conf:
            if 'combination_image' in self.conf:
                num_decode += 1
            if 'gen_sep_images' in self.conf:
                num_decode += 2
        if 'use_depth' in self.conf:
            if 'combination_image' in self.conf:
                num_decode += 1
            if 'gen_sep_images' in self.conf:
                num_decode += 2
        if 'predict_target_masks' in self.conf:
            num_decode += 2

        d3_0 = lrelu(conv2d_msra(d3, 64*num_decode, 5, 5, 1, 1, "d3_0"))

        #splitting up the representation
        split_list = tf.split(d3_0, num_decode, axis=3)

        if 'use_color' in self.conf:
            if 'combination_image' in self.conf:
                self.gen_image1 = self.decode_flow(self.image0, split_list.pop(), 'dec_image1')
            else:
                self.gen_image1 = tf.constant([0])
            if 'gen_sep_images' in self.conf:
                self.gen_image1_only0 = self.decode_flow(self.image0, split_list.pop(), 'dec_image1_only0')
                self.gen_image1_only1 = self.decode_flow(self.image0, split_list.pop(), 'dec_image1_only1')
            else: self.gen_image1_only0, self.gen_image1_only1 = tf.constant([0]), tf.constant([0])

        if 'use_depth' in self.conf:
            if 'combination_image' in self.conf:
                self.gen_depth1 = self.decode_direct(split_list.pop(), 'dec_dimage1_f', num_outputs=1)
            else:
                self.gen_depth1 = tf.constant([0])  #dummy value
            if 'gen_sep_images' in self.conf:
                self.gen_depth1_only0 = self.decode_direct(split_list.pop(), 'dec_depth1_only0', num_outputs=1)
                self.gen_depth1_only1 = self.decode_direct(split_list.pop(), 'dec_depth1_only1', num_outputs=1)
            else: self.gen_depth1_only0, self.gen_depth1_only1 = tf.constant([0]), tf.constant([0])

        if 'predict_target_masks' in self.conf:
            self.gen_image1_mask0 = self.decode_direct(split_list.pop(), 'dec_image1_mask0', num_outputs=1)
            self.gen_image1_mask1 = self.decode_direct(split_list.pop(), 'dec_image1_mask1', num_outputs=1)
        else:
            self.gen_image1_mask0 = tf.constant([0])  #dummy value
            self.gen_image1_mask1 = tf.constant([0])  # dummy value

        assert split_list == []

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(max_to_keep=20)

    def build_loss(self):

        train_summaries = []
        val_summaries = []

        self.loss = 0
        if 'use_color' in self.conf:
            print 'using color loss'
            colorloss = 0.
            if 'combination_image' in self.conf:
                colorloss += euclidean_loss(self.gen_image1, self.image1)

            if 'gen_sep_images' in self.conf:
                if 'masked_image_loss' in self.conf:
                    diff = (self.gen_image1_only0 - self.image1_only0) * self.image1_mask0
                    colorloss += tf.reduce_mean(tf.reduce_sum(tf.pow(diff, 2), 3))
                    diff = (self.gen_image1_only1 - self.image1_only1) * self.image1_mask1
                    colorloss += tf.reduce_mean(tf.reduce_sum(tf.pow(diff, 2), 3))
                else:
                    colorloss += euclidean_loss(self.gen_image1_only0, self.image1_only0)
                    colorloss += euclidean_loss(self.gen_image1_only1, self.image1_only1)

            train_summaries.append(tf.summary.scalar("colorloss", colorloss))
            val_summaries.append(tf.summary.scalar("val_colorloss", colorloss))
            self.loss += colorloss

        if 'use_depth' in self.conf:
            print 'using depth loss'
            depthloss = 0.
            depth_factor = self.conf['use_depth']
            if 'combination_image' in self.conf:
                depthloss += euclidean_loss(self.gen_depth1, self.depth1)

            if 'gen_sep_images' in self.conf:
                if 'masked_image_loss' in self.conf:
                    diff = (self.gen_depth1_only0 - self.depth1_only0) * self.image1_mask0
                    depthloss += tf.reduce_mean(tf.reduce_sum(tf.pow(diff, 2), 3))*depth_factor
                    diff = (self.gen_depth1_only1 - self.depth1_only1) * self.image1_mask1
                    depthloss += tf.reduce_mean(tf.reduce_sum(tf.pow(diff, 2), 3))*depth_factor
                else:
                    depthloss += euclidean_loss(self.gen_depth1_only0, self.depth1_only0)*depth_factor
                    depthloss += euclidean_loss(self.gen_depth1_only1, self.depth1_only1)*depth_factor

            train_summaries.append(tf.summary.scalar("depthloss", depthloss))
            val_summaries.append(tf.summary.scalar("val_depthloss", depthloss))
            self.loss += depthloss

        if 'predict_target_masks' in self.conf:
            print 'using target mask loss'
            mask_factor = self.conf['predict_target_masks']
            mask_loss = 0.
            mask_loss += euclidean_loss(self.gen_image1_mask0, self.image1_mask0)*mask_factor
            mask_loss += euclidean_loss(self.gen_image1_mask1, self.image1_mask1)*mask_factor
            train_summaries.append(tf.summary.scalar("mask_loss", mask_loss))
            val_summaries.append(tf.summary.scalar("val_mask_loss", mask_loss))
            self.loss += mask_loss

        train_summaries.append(tf.summary.scalar("total_training_loss", self.loss))
        val_summaries.append(tf.summary.scalar("total_val_loss", self.loss))

        self.train_op = tf.train.AdamOptimizer(self.conf['learning_rate']).minimize(self.loss)

        self.train_summ_op = tf.summary.merge(train_summaries)
        self.val_summ_op = tf.summary.merge(val_summaries)


    def visualize(self, sess):
        image0, \
        image0_mask0,\
        image0_mask1,\
        image1,\
        image1_only0,\
        image1_only1,\
        image1_mask0,\
        image1_mask1,\
        depth0,\
        depth1,\
        depth1_only0,\
        depth1_only1,\
        displacement,\
        gen_image1,\
        gen_image1_only0,\
        gen_image1_only1,\
        gen_image1_mask0,\
        gen_image1_mask1,\
        gen_depth1,\
        gen_depth1_only0,\
        gen_depth1_only1 = sess.run([
                          ### inputs
                          self.image0       ,
                          self.image0_mask0 ,
                          self.image0_mask1 ,
                          self.image1       ,
                          self.image1_only0 ,
                          self.image1_only1 ,
                          self.image1_mask0 ,
                          self.image1_mask1 ,
                          self.depth0       ,
                          self.depth1       ,
                          self.depth1_only0 ,
                          self.depth1_only1 ,
                          self.displacement,
                          ### outputs
                          self.gen_image1,
                          self.gen_image1_only0,
                          self.gen_image1_only1,
                          self.gen_image1_mask0,
                          self.gen_image1_mask1,
                          self.gen_depth1,
                          self.gen_depth1_only0,
                          self.gen_depth1_only1,
                          ],
                         feed_dict={self.train_cond: 0})

        # print 'input'
        # plt.imshow(image0[0])
        # plt.show()
        # print 'gtruth'
        # plt.imshow(image1[0])
        # plt.show()
        #
        # print 'gen_image'
        # plt.imshow(gen[0])
        # plt.show()
        for b in range(10):
            print 'input', b
            fig1 = plt.figure()

            plt.subplot(3, 3, 1)
            plt.imshow(image0[b])
            plt.axis('off')

            plt.subplot(3, 3, 4)
            plt.imshow(image1[b])
            plt.axis('off')

            plt.subplot(3, 3, 5)
            plt.imshow(image1_only0[b])
            plt.axis('off')

            plt.subplot(3, 3, 6)
            plt.imshow(image1_only1[b])
            plt.axis('off')

            if 'combination_image' in self.conf:
                plt.subplot(3, 3, 7)
                plt.imshow(gen_image1[b])
                plt.axis('off')

            plt.subplot(3, 3, 8)
            plt.imshow(gen_image1_only0[b])
            plt.axis('off')

            plt.subplot(3, 3, 9)
            plt.imshow(gen_image1_only1[b])
            plt.axis('off')

            plt.draw()
            fig2 = plt.figure()

            ## depth
            plt.subplot(3, 3, 1)
            plt.imshow(np.squeeze(depth0[b]))
            plt.axis('off')

            plt.subplot(3, 3, 4)
            plt.imshow(np.squeeze(depth1[b]))
            plt.axis('off')

            plt.subplot(3, 3, 5)
            plt.imshow(np.squeeze(depth1_only0[b]))
            plt.axis('off')

            plt.subplot(3, 3, 6)
            plt.imshow(np.squeeze(depth1_only1[b]))
            plt.axis('off')

            if 'combination_image' in self.conf:
                plt.subplot(3, 3, 7)
                plt.imshow(np.squeeze(gen_depth1[b]))
                plt.axis('off')

            plt.subplot(3, 3, 8)
            plt.imshow(np.squeeze(gen_depth1_only0[b]))
            plt.axis('off')

            plt.subplot(3, 3, 9)
            plt.imshow(np.squeeze(gen_depth1_only1[b]))
            plt.axis('off')

            plt.draw()

            if 'predict_target_masks' in self.conf:
                rows = 2
            else:
                rows = 1

            fig3 = plt.figure()
            iplt = 0
            iplt += 1
            plt.subplot(rows, 4, iplt)
            plt.imshow(np.squeeze(image0_mask0[b]))
            plt.axis('off')

            iplt += 1
            plt.subplot(rows, 4, iplt)
            plt.imshow(np.squeeze(image0_mask1[b]))
            plt.axis('off')

            iplt += 1
            plt.subplot(rows, 4, iplt)
            plt.imshow(np.squeeze(image1_mask0[b]))
            plt.axis('off')

            iplt += 1
            plt.subplot(rows, 4, iplt)
            plt.imshow(np.squeeze(image1_mask1[b]))
            plt.axis('off')

            if 'predict_target_masks' in self.conf:
                plt.subplot(rows, 4, 7)
                plt.imshow(np.squeeze(image1_mask0[b]))
                plt.axis('off')

                plt.subplot(rows, 4, 8)
                plt.imshow(np.squeeze(image1_mask1[b]))
                plt.axis('off')

            save_images = True
            if save_images:
                fig1.savefig(self.conf['output_dir']+'/img_exp{}'.format(b))
                fig2.savefig(self.conf['output_dir'] + '/depth_exp{}'.format(b))
                fig3.savefig(self.conf['output_dir'] + '/masks_exp{}'.format(b))
            else:
                plt.show()

        iter_num = re.match('.*?([0-9]+)$', self.conf['visualize']).group(1)

        return



