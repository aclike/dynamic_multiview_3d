from dyn_mult_view.mv3d.utils.tf_utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

class AppearanceFlowModel():

    def __init__(self,
               conf,
               load_tfrec=True,
               build_loss=True):

        self.conf = conf
        self.batch_size = conf['batch_size']
        self.image_shape = [128, 128, 3]
        self.max_iter = 1000000
        self.start_iter = 0

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

            self.disp = tf.placeholder(tf.float32, [self.batch_size, 5],
                                       name='labels')

        else:
            train_image0, train_image1, train_depth_image0, train_depth_image1, train_disp = build_tfrecord_input(conf, training=True)
            test_image0, test_image1, test_depth_image0, test_depth_image1, test_disp = build_tfrecord_input(conf, training=False)

            self.image0, self.image1, self.depth_image0, self.depth_image1, self.disp = tf.cond(self.train_cond > 0,  # if 1 use trainigbatch else validation batch
                                          lambda: [train_image0, train_image1, train_depth_image0, train_depth_image1, train_disp],
                                          lambda: [test_image0, test_image1, test_depth_image0, test_depth_image1, test_disp])
            self.image0 = tf.reshape(self.image0, [conf['batch_size'], 128, 128, 3])
            self.image1 = tf.reshape(self.image1, [conf['batch_size'], 128, 128, 3])
            self.depth_image0 = tf.reshape(self.depth_image0, [conf['batch_size'], 128, 128, 1])
            self.depth_image1 = tf.reshape(self.depth_image1, [conf['batch_size'], 128, 128, 1])

            self.buildModel()
            if build_loss:
                self.build_loss()


    def decodeAngle(self):
        a0 = lrelu(linear_msra(self.disp, 64, "a0"))
        a1 = lrelu(linear_msra(a0, 64, "a1"))
        return lrelu(linear_msra(a1, 64, "a2"))

    def build_loss(self):

        train_summaries = []
        val_summaries = []

        self.loss = euclidean_loss(self.gen, self.image1)
        train_summaries.append(tf.summary.scalar("training_loss", self.loss))
        val_summaries.append(tf.summary.scalar("val_loss", self.loss))

        self.train_op = tf.train.AdamOptimizer(self.conf['learning_rate']).minimize(self.loss)

        self.train_summ_op = tf.summary.merge(train_summaries)
        self.val_summ_op = tf.summary.merge(val_summaries)


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
        concated = tf.concat(axis=1, values=[e5, self.decodeAngle()])

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
        self.flow_field = deconv2d_msra(d1_0, [self.batch_size, 128, 128, 2], 5, 5, 2, 2, "flow_field")
        with tf.variable_scope("warp_pts"):
            img_shape = tf.shape(image0)
            self.warp_pts = self.flow_field + coords(img_shape[0], img_shape[1], self.batch_size)
        self.gen = tf.contrib.resampler.resampler(image0, self.warp_pts)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(max_to_keep=20)

    def visualize(self, sess):

        image0, image1, gen, loss, disp, warp_pts = sess.run([self.image0, self.image1,
                                                    self.gen, self.loss, self.disp,
                                                    self.warp_pts],
                                 feed_dict={self.train_cond: 0})

        print 'loss', loss
        print warp_pts
        print 'max resample coord:', np.max(warp_pts)
        iter_num = re.match('.*?([0-9]+)$', self.conf['visualize']).group(1)

        path = self.conf['output_dir']
        save_images(gen, [8, 8], path + "/output_%s.png" % (iter_num))
        save_images(np.array(image1), [8, 8],
                        path + '/tr_gt_%s.png' % (iter_num))
        save_images(np.array(image0), [8, 8],
                        path + '/tr_input_%s.png' % (iter_num))

        plt.figure()
        plt.axes([0, 0.025, 0.95, 0.95])
        plt.quiver(warp_pts[0,:,:,0],warp_pts[0,:,:,1])
        plt.savefig(path + '/quiver_%s.pdf' % (iter_num))
    
        plt.figure()
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        ax1.imshow(image0[0])
        ax2.imshow(gen[0])

        coordsA = "data"
        coordsB = "data"
        # random pts 
        num_samples = 6
        pts_output = np.random.randint(50, 78, size=(num_samples,2))
        for pt_output in pts_output:
            sampled_location = warp_pts[0,pt_output[0],pt_output[1],:].astype('uint32')
            print pt_output, sampled_location
            con = ConnectionPatch(xyA=np.flip(pt_output,0), xyB=np.flip(sampled_location,0), coordsA=coordsA, coordsB=coordsB,
                         axesA=ax2, axesB=ax1,
                         arrowstyle="<->",shrinkB=5)
            ax2.add_artist(con)
        ax1.set_xlim(0, 128)
        ax1.set_ylim(0, 128)
        ax2.set_xlim(0, 128)
        ax2.set_ylim(0, 128)
        plt.draw()
        plt.savefig(path + '/corr_plot_%s.pdf' % (iter_num))
