from main_model import Base_Prediction_Model
from dyn_mult_view.mv3d.utils.tf_utils import *
import tensorflow as tf

class AppearanceFlowModel(Base_Prediction_Model):

    def buildModel(self):
        image0 = self.image0
        gtruth_image = self.image1

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
        a0 = lrelu(linear_msra(self.disp, 64, "a0"))
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

        # Appearance flow layers. We can maybe change these parameters at some point. The appearance flow paper has this deconv layer as kernel 3, stride 1, pad 1.
        self.pre_resampler = deconv2d_msra(d1_0, [self.batch_size, 128, 128, 2], 5, 5, 2, 2, "warp_pts")
        self.gen = tf.contrib.resampler.resampler(image0, self.pre_resampler)

        self.loss = euclidean_loss(self.gen, gtruth_image)
        self.training_summ = tf.summary.scalar("training_loss", self.loss)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(max_to_keep=20)
