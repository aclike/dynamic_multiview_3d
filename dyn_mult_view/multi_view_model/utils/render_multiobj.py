from renderer import Renderer
import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
import random
import os
import sys
import Image
import copy
import tensorflow as tf
import cPickle
import mmap
from scipy.spatial.distance import cdist
import math

from multiprocessing import Pool

import gc

import matplotlib.pyplot as plt
import threading
from collections import OrderedDict

# bam_path = "../obj_cars3"

# bam_path = "/mnt/sda1/shapenet/shapenetcore_v2/ShapeNetCore.v2/02958343"

output_path = "../shapenet_output"
save_images_as_pngs = False  # possibly only for debugging

MASK_APPROACH = 'red_green'  # one of ('red_green', 'closer')

def _bytes_feature(value):
    if not isinstance(value, (np.ndarray, list, tuple)):
      value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    if not isinstance(value, (np.ndarray, list, tuple)):
      value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class Render_thread(threading.Thread):
    def __init__(self, conf, model_names_subset, placeholders, enqueue_op, sess):

        """
        :param conf:
        :param bam_path:
        :param placeholders:  a dictionary with pairs 'name_string': tf.placeholder
        :param enqueue_op:
        :param sess:
        :param write_tf_recs:
        :return:
        """
        self.model_names_subset = model_names_subset
        print 'starting render thread...'

        self.enqueue_op = enqueue_op
        self.sess = sess

        self.bam_path = conf['data_dir'] + '/bam_path'

        self.placeholders = placeholders
        self.mean_abs_displacement = (5,5)

        self.lock = threading.RLock()
        super(Render_thread, self).__init__()

    def set_mean_displacement(self, mean):
        with self.lock:
            self.mean_abs_displacement = mean

    def run(self):
        _i = -1
        num_load_models = 30
        while True:
            _i += 1
            print '_i', _i

            if _i % num_load_models*2 == 0:
                print 'loading models...'
                model_inds = np.random.randint(0, len(self.model_names_subset), num_load_models)

                model_names = [self.model_names_subset[ind] for ind in list(model_inds)]

                if _i !=0:
                    rend.delete()
                rend = Renderer(True, False)
                rend.loadModels(model_names, self.bam_path)

            ind0, ind1 = random.randint(0, num_load_models - 1), random.randint(0, num_load_models - 1)
            while ind0 == ind1:
                ind1 = random.randint(0, num_load_models - 1)
            pos0, yaw0 = rend.showModel(ind0, debug=False)
            pos1, yaw1 = rend.showModel(ind1, debug=False)

            num_lights = random.randint(2, 4)
            lights = []
            for nl in range(num_lights):
                light_pos = [random.random() * 2. + 2.5,
                             random.randint(-90, 90),
                             random.randint(0, 360),
                             random.randint(10, 15)]
                lights.append(light_pos)

            rad = 2.5

            # Two models in the same scene occluding each other
            # -----------------------------------------------
            el0 = int(random.random() * 50 - 10)
            az0 = int(random.random() * 20) + 90  # reduced in order to better guarantee occlusion
            blur0 = random.random() * 0.4 + 0.2
            blending0 = random.random() * 0.3 + 1

            im0, dm0 = rend.renderView([rad, el0, az0], lights, blur0, blending0, default_bg_setting=False)

            if save_images_as_pngs:
                output_name = os.path.join(output_path, 'rgb_%d_0.png' % _i)
                scipy.misc.toimage(im0, cmin=0, cmax=255).save(output_name)
                output_name = os.path.join(output_path, 'depth_%d_0.png' % _i)
                scipy.misc.toimage(dm0, cmin=0, cmax=65535, low=0, high=65535, mode='I').save(output_name)

            # Per-object masks of the models (part 1)
            # ---------------------------------------
            image0_mask0, image0_mask1 = None, None
            if MASK_APPROACH == 'red_green':
                rend.hideModel(ind0)
                rend.hideModel(ind1)
                rend.showModel(ind0, pos=pos0, yaw=yaw0, color='red')
                rend.showModel(ind1, pos=pos1, yaw=yaw1, color='green')
                _im0, _ = rend.renderView([rad, el0, az0], lights, blur0, blending0, default_bg_setting=False, reuse_camera_target=True)

                image0_mask0 = copy.deepcopy(_im0)[:, :, 0]
                image0_mask0[image0_mask0 >= 128] = 255
                image0_mask0[image0_mask0 < 128] = 0
                image0_mask0 = (255.0 / image0_mask0.max() * (image0_mask0 - image0_mask0.min())).astype(np.uint8)
                image0_mask1 = copy.deepcopy(_im0)[:, :, 1]
                image0_mask1[image0_mask1 >= 128] = 255
                image0_mask1[image0_mask1 <  128] = 0
                image0_mask1 = (255.0 / image0_mask1.max() * (image0_mask1 - image0_mask1.min())).astype(np.uint8)

                rend.hideModel(ind0)
                rend.hideModel(ind1)
                rend.showModel(ind0, pos=pos0, yaw=yaw0)
                rend.showModel(ind1, pos=pos1, yaw=yaw1)
            elif MASK_APPROACH == 'closer':
                az0_rad = math.radians(az0)
                el0_rad = math.radians(el0)
                camera_pos = np.array([
                    rad * math.cos(el0_rad) * math.cos(az0_rad),
                    rad * math.cos(el0_rad) * math.sin(az0_rad),
                    rad * math.sin(el0_rad)
                ])
                closer_idx = int(cdist(camera_pos, pos1 + (0,), 'euclidean') < cdist(camera_pos, pos0 + (0,), 'euclidean'))
                raise NotImplementedError("didn't finish implementing this because I think I found the original error")

            if save_images_as_pngs:
                mask0_v0_path = os.path.join(output_path, 'image0_mask0.png')
                mask1_v0_path = os.path.join(output_path, 'image0_mask1.png')
                Image.fromarray(image0_mask0).save(mask0_v0_path)
                Image.fromarray(image0_mask1).save(mask1_v0_path)

            # A randomly rotated view of the same two models
            # --------------------------------------------
            multiplier = (-1, 1)

            with self.lock:
                el1 = el0 + random.choice(multiplier) * np.random.normal(self.mean_abs_displacement[0], self.mean_abs_displacement[0])
                az1 = az0 + random.choice(multiplier) * np.random.normal(self.mean_abs_displacement[1], self.mean_abs_displacement[1])

            blur1 = random.random() * 0.4 + 0.2
            blending1 = random.random() * 0.3 + 1

            im1, dm1 = rend.renderView([rad, el1, az1], lights, blur1, blending1, default_bg_setting=False, reuse_camera_target=True)

            if save_images_as_pngs:
                output_name = os.path.join(output_path, 'rgb_%d_1.png' % _i)
                scipy.misc.toimage(im1, cmin=0, cmax=255).save(output_name)
                output_name = os.path.join(output_path, 'depth_%d_1.png' % _i)
                scipy.misc.toimage(dm1, cmin=0, cmax=65535, low=0, high=65535, mode='I').save(output_name)

            # Two separate images of each model from the same new viewpoint (no occlusions)
            # ------------------------------------------------------------------------------
            # (just model 0)
            rend.hideModel(ind1)
            im2, dm2 = rend.renderView([rad, el1, az1], lights, blur1, blending1, default_bg_setting=False, reuse_camera_target=True)

            if save_images_as_pngs:
                output_name = os.path.join(output_path, 'rgb_%d_2.png' % _i)
                scipy.misc.toimage(im2, cmin=0, cmax=255).save(output_name)
                output_name = os.path.join(output_path, 'depth_%d_2.png' % _i)
                scipy.misc.toimage(dm2, cmin=0, cmax=65535, low=0, high=65535, mode='I').save(output_name)

            # (just model 1)
            rend.hideModel(ind0)
            rend.showModel(ind1, pos=pos1, yaw=yaw1)
            im3, dm3 = rend.renderView([rad, el1, az1], lights, blur1, blending1, default_bg_setting=False, reuse_camera_target=True)

            if save_images_as_pngs:
                output_name = os.path.join(output_path, 'rgb_%d_3.png' % _i)
                scipy.misc.toimage(im3, cmin=0, cmax=255).save(output_name)
                output_name = os.path.join(output_path, 'depth_%d_3.png' % _i)
                scipy.misc.toimage(dm3, cmin=0, cmax=65535, low=0, high=65535, mode='I').save(output_name)

            # Per-object masks of the models (part 2)
            # ---------------------------------------
            image1_mask0, image1_mask1 = None, None
            if MASK_APPROACH == 'red_green':
                rend.hideModel(ind1)
                rend.showModel(ind0, pos=pos0, yaw=yaw0, color='red')
                rend.showModel(ind1, pos=pos1, yaw=yaw1, color='green')
                _im1, _ = rend.renderView([rad, el1, az1], lights, blur1, blending1, default_bg_setting=False, reuse_camera_target=True)

                image1_mask0 = copy.deepcopy(_im1)[:, :, 0]
                image1_mask0[image1_mask0 >= 128] = 255
                image1_mask0[image1_mask0 <  128] = 0
                image1_mask0 = (255.0 / image1_mask0.max() * (image1_mask0 - image1_mask0.min())).astype(np.uint8)
                image1_mask1 = copy.deepcopy(_im1)[:, :, 1]
                image1_mask1[image1_mask1 >= 128] = 255
                image1_mask1[image1_mask1 <  128] = 0
                image1_mask1 = (255.0 / image1_mask1.max() * (image1_mask1 - image1_mask1.min())).astype(np.uint8)
            else:
                raise NotImplementedError("didn't implement this")

            if save_images_as_pngs:
                mask0_v1_path = os.path.join(output_path, 'mask0_v1_view1.png')
                mask1_v1_path = os.path.join(output_path, 'mask1_v1_view1.png')
                Image.fromarray(image1_mask0).save(mask0_v1_path)
                Image.fromarray(image1_mask1).save(mask1_v1_path)

                # for debugging
                output_name = os.path.join(output_path, 'mask_original.png')
                scipy.misc.toimage(_im1, cmin=0, cmax=255).save(output_name)

            rend.hideModel(ind0)
            rend.hideModel(ind1)

            _displacement_np = np.array([el1, az1]) - np.array([el0, az0])

            max_16bit_val = 65535
            placeholders = self.placeholders

            feed_dict = {
            placeholders['image0']: im0.astype(np.float32)/255.,
            placeholders['image0_mask0']: image0_mask0.astype(np.float32) / 255.,
            placeholders['image0_mask1']: image0_mask0.astype(np.float32) / 255.,
            placeholders['image1']: im1.astype(np.float32)/255.,
            placeholders['image1_only0']: im2.astype(np.float32)/255.,
            placeholders['image1_only1']: im3.astype(np.float32)/255.,
            placeholders['image1_mask0']: image1_mask0.astype(np.float32)/255.,
            placeholders['image1_mask1']: image1_mask1.astype(np.float32)/255.,
            placeholders['depth0']:dm0.astype(np.float32)/max_16bit_val,
            placeholders['depth1']:dm1.astype(np.float32)/max_16bit_val,
            placeholders['depth1_only0']:dm2.astype(np.float32)/max_16bit_val,
            placeholders['depth1_only1']:dm3.astype(np.float32)/max_16bit_val,
            placeholders['displacement']:_displacement_np.astype(np.float32)}   # (elevation, azimuth)

            self.sess.run(self.enqueue_op, feed_dict=feed_dict)

            # if write_tf_recs:
            #     # Write everything to TFRecords
            #     _displacement_np = np.array([el1, az1]) - np.array([el0, az0])
            #     example = tf.train.Example(features=tf.train.Features(feature={
            #         'image0': _bytes_feature(tf.compat.as_bytes(im0.tostring())),
            #         'image0_mask0': _bytes_feature(tf.compat.as_bytes(mask0_v0.tostring())),
            #         'image0_mask1': _bytes_feature(tf.compat.as_bytes(image0_mask1.tostring())),
            #         'image1': _bytes_feature(tf.compat.as_bytes(im1.tostring())),
            #         'image1_only0': _bytes_feature(tf.compat.as_bytes(im2.tostring())),
            #         'image1_only1': _bytes_feature(tf.compat.as_bytes(im3.tostring())),
            #         'image1_mask0': _bytes_feature(tf.compat.as_bytes(image1_mask0.tostring())),
            #         'image1_mask1': _bytes_feature(tf.compat.as_bytes(image1_mask1.tostring())),
            #         'depth0': _bytes_feature(tf.compat.as_bytes(dm0.tostring())),
            #         'depth1': _bytes_feature(tf.compat.as_bytes(dm1.tostring())),
            #         'depth1_only0': _bytes_feature(tf.compat.as_bytes(dm2.tostring())),
            #         'depth1_only1': _bytes_feature(tf.compat.as_bytes(dm3.tostring())),
            #         'displacement': _float_feature(_displacement_np),  # (elevation, azimuth)
            #     }))
            #     writer.write(example.SerializeToString())
        # writer.close()


class OnlineRenderer(object):
    def __init__(self, mode, conf, sess):
        self.conf = conf

        split_file = conf['data_dir'] + '/train_val_split.pkl'
        if os.path.isfile(split_file):
            print 'Loading splitfile from: ',split_file
            file_split_dict = cPickle.load(open(split_file, "rb"))
        else:
            bam_path = conf['data_dir'] + '/bam_path'
            all_model_names = [mn for mn in os.listdir(bam_path) if mn.endswith('bam')]
            #discard files over 40 MB for speed
            model_names = [mn for mn in all_model_names if os.path.getsize(os.path.join(bam_path,mn))/1024.**2 < 40]
            model_names = [mn for mn in model_names if no_textures(bam_path, mn)]

            train_frac = .8
            val_frac = .15
            test_frac = .05
            model_count = len(model_names)

            file_split_dict = {}
            file_split_dict['train'] = model_names[:int(model_count*train_frac)]
            rem_models = model_names[int(model_count*train_frac):]
            file_split_dict['val'] = rem_models[:int(model_count*val_frac)]
            rem_models = rem_models[int(model_count*val_frac):]
            file_split_dict['test'] = rem_models
            cPickle.dump(file_split_dict, open(split_file, 'wb'))

        self.sel_files = file_split_dict[mode]

        self.sess = sess
        self.batch_size = conf['batch_size']
        rgb_shape = (128,128,3)
        single_channel = (128,128)
        placeholders = OrderedDict()

        placeholders['image0'] = tf.placeholder(tf.float32, name='image0', shape=rgb_shape)
        placeholders['image0_mask0'] = tf.placeholder(tf.float32, name='image0_mask0', shape=single_channel)
        placeholders['image0_mask1'] = tf.placeholder(tf.float32, name='image0_mask1', shape=single_channel)
        placeholders['image1'] = tf.placeholder(tf.float32, name='image1', shape=rgb_shape)
        placeholders['image1_only0'] = tf.placeholder(tf.float32, name='image1_only0', shape=rgb_shape)
        placeholders['image1_only1'] = tf.placeholder(tf.float32, name='image1_only1', shape=rgb_shape)
        placeholders['image1_mask0'] = tf.placeholder(tf.float32, name='image1_mask0', shape=single_channel)
        placeholders['image1_mask1'] = tf.placeholder(tf.float32, name='image1_mask1', shape=single_channel)
        placeholders['depth0'] = tf.placeholder(tf.float32, name='depth0', shape=single_channel)
        placeholders['depth1'] = tf.placeholder(tf.float32, name='depth1', shape=single_channel)
        placeholders['depth1_only0'] = tf.placeholder(tf.float32, name='depth1_only0', shape=single_channel)
        placeholders['depth1_only1'] = tf.placeholder(tf.float32, name='depth1_only1', shape=single_channel)
        placeholders['displacement'] = tf.placeholder(tf.float32, name='displacement', shape=[2])

        self.num_threads = 1

        shapes = [placeholders[key].get_shape().as_list() for key in placeholders.keys()]
        dtypes = [placeholders[key].dtype for key in placeholders.keys()]
        self.q = tf.FIFOQueue(100, dtypes, shapes=shapes)

        placeholder_list = [placeholders[key] for key in placeholders.keys()]
        self.enqueue_op = self.q.enqueue(placeholder_list)

        self.placeholders = placeholders
        self.start_threads()

        self.image0, self.image0_mask0, self.image0_mask1, self.image1, self.image1_only0, \
        self.image1_only1, self.image1_mask0, self.image1_mask1, self.depth0, self.depth1, \
        self.depth1_only0, self.depth1_only1, self.displacement = self.q.dequeue_many(self.batch_size)

    def start_threads(self):
        self.thread_list = []
        for i in range(self.num_threads):
            t = Render_thread(self.conf, self.sel_files, self.placeholders, self.enqueue_op, self.sess)
            t.setDaemon(True)
            t.start()
            self.thread_list.append(t)

    def set_mean_displacement(self, mean_disp):
        print 'setting mean displacment to ', mean_disp
        for t in self.thread_list:
            t.set_mean_displacement(mean_disp)

def no_textures(bam_path, modelname):
    f = open(os.path.join(bam_path, modelname))
    s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return s.find('texture') == -1 and s.find('Texture') == -1

def test_online_renderer():
    sess = tf.InteractiveSession()
    conf = {}

    conf['batch_size'] = 10
    conf['data_dir'] = '/home/frederik/Documents/catkin_ws/src/dynamic_multiview_3d/trainingdata/cardataset_bam'

    r = OnlineRenderer('train', conf, sess)

    for i_run in range(10):
        print 'run number ', i_run

        image0, image0_mask0, image0_mask1, image1, image1_only0, \
        image1_only1, image1_mask0, image1_mask1, depth0, depth1, \
        depth1_only0, depth1_only1, displacement = sess.run([r.image0,
                                            r.image0_mask0,
                                            r.image0_mask1,
                                            r.image1,
                                            r.image1_only0,
                                            r.image1_only1,
                                            r.image1_mask0,
                                            r.image1_mask1,
                                            r.depth0,
                                            r.depth1,
                                            r.depth1_only0,
                                            r.depth1_only1,
                                            r.displacement,
                                            ])
        for b in range(1):
            print 'batchind', b
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

            plt.show()

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

        r.set_mean_displacement(tuple(np.array([5,5])*(i_run+2)))

if __name__ == "__main__":
    test_online_renderer()
    # sys.exit(render_thread()