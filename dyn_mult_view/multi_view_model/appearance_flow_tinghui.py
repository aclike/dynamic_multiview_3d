from appearance_flow_model import AppearanceFlowModel
from dyn_mult_view.mv3d.utils.tf_utils import *
import tensorflow as tf

class AppearanceFlowTinghui(AppearanceFlowModel):

    def decodeAngle(self):
        a0 = lrelu(linear_msra(self.disp, 64, "a0"))
        a1 = lrelu(linear_msra(a0, 64, "a1"))
        a2 = lrelu(linear_msra(a1, 64, "a2"))
        return a2

    def buildModel(self):
        image0 = self.image0

        # convolutional encoder
        e0 = relu(conv2d_msra(image0, 16, 3, 3, 2, 2, "e0"))
        e1 = relu(conv2d_msra(e0, 32, 3, 3, 2, 2, "e1"))
        e2 = relu(conv2d_msra(e1, 64, 3, 3, 2, 2, "e2"))
        e3 = relu(conv2d_msra(e2, 128, 3, 3, 2, 2, "e3"))
        e4 = relu(conv2d_msra(e3, 256, 3, 3, 2, 2, "e4"))
        # e5 = relu(conv2d_msra(e4, 512, 3, 3, 1, 1, "e5"))
        e4r = tf.reshape(e4, [self.batch_size, 4096])
        e_fc0 = relu(linear_msra(e4r, 2048, "e_fc0"))
        e_fc1 = relu(linear_msra(e_fc0, 2048, "e_fc1"))

        # angle processing
        concated = tf.concat(axis=1, values=[e_fc1, self.decodeAngle()])

        # joint processing
        d_fc0 = relu(linear_msra(concated, 2048, "a3"))
        d_fc1 = relu(linear_msra(d_fc0, 2048, "a4"))
        dr = tf.reshape(d_fc1, [self.batch_size, 8, 8, 32])

        # convolutional decoder
        d3 = relu(deconv2d_msra(dr, [self.batch_size, 16, 16, 128], 3, 3, 2, 2, "d3"))
        d2 = relu(deconv2d_msra(d3, [self.batch_size, 32, 32 ,64], 3, 3, 2, 2, "d2"))
        d1 = relu(deconv2d_msra(d2, [self.batch_size, 64, 64, 32],
                                 3, 3, 2, 2, "d1"))
        d0 = relu(deconv2d_msra(d1, [self.batch_size, 128, 128, 16],
                                 3, 3, 2, 2, "d0"))

        # Appearance flow layers. We can maybe change these parameters at some point. The appearance flow paper has this deconv layer as kernel 3, stride 1, pad 1.
        self.flow_field = deconv2d_msra(d0, [self.batch_size, 128, 128, 2], 3, 3, 1, 1, "flow_field")

        with tf.variable_scope("warp_pts"):
            img_shape = tf.shape(image0)
            self.warp_pts = self.flow_field + coords(img_shape[1], img_shape[2], self.batch_size)
        self.gen = tf.contrib.resampler.resampler(image0, self.warp_pts)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(max_to_keep=20)
