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
import math
import time
import ray

import collections

from multiprocessing import Pool

import gc

import matplotlib.pyplot as plt
import threading
from collections import OrderedDict


NUM_PROC = 11
IM_PER_PROC = 20
# bam_path = "../obj_cars3"

# bam_path = "/mnt/sda1/shapenet/shapenetcore_v2/ShapeNetCore.v2/02958343"

output_path = "../shapenet_output"
save_images_as_pngs = False  # possibly only for debugging

MASK_APPROACH = 'closer'  # one of ('red_green', 'closer')

def _bytes_feature(value):
    if not isinstance(value, (np.ndarray, list, tuple)):
      value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    if not isinstance(value, (np.ndarray, list, tuple)):
      value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def l2_dist(arr1, arr2):
    return np.sum((np.array(arr1) - np.array(arr2)) ** 2)

class Output():
    def __init__(self):
        self.image0       = []
        self.image0_mask0 = []
        self.image0_mask1 = []
        self.image1       = []
        self.image1_only0 = []
        self.image1_only1 = []
        self.image1_mask0 = []
        self.image1_mask1 = []
        self.depth0       = []
        self.depth1       = []
        self.depth1_only0 = []
        self.depth1_only1 = []
        self.displacement = []

# @ray.remote
def render_worker(arglist):
    bam_path, model_names_subset, mean_abs_displacement = arglist

    print 'started worker with pid', os.getpid()

    print 'loading models...'
    num_load_models = IM_PER_PROC/2
    model_inds = np.random.randint(0, len(model_names_subset), num_load_models)
    model_names = [model_names_subset[ind] for ind in list(model_inds)]

    rend = Renderer(True, False)
    rend.loadModels(model_names, bam_path)
    out = Output()

    for _i in range(IM_PER_PROC):
        print '_i', _i

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
            camera_pos0 = np.array([
                rad * math.cos(el0_rad) * math.cos(az0_rad),
                rad * math.cos(el0_rad) * math.sin(az0_rad),
                rad * math.sin(el0_rad)
            ])
            closer_idx = int(l2_dist(camera_pos0, pos1 + (0,)) < l2_dist(camera_pos0, pos0 + (0,)))
            rend.hideModel(ind0 if closer_idx == 1 else ind1)
            _im00, _ = rend.renderView([rad, el0, az0], lights, blur0, blending0, default_bg_setting=False, reuse_camera_target=True)
            rend.hideModel(ind1 if closer_idx == 1 else ind0)
            if closer_idx == 1:
                _ind, _pos, _yaw = ind0, pos0, yaw0
            else:
                _ind, _pos, _yaw = ind1, pos1, yaw1
            rend.showModel(_ind, pos=_pos, yaw=_yaw)
            _im01, _ = rend.renderView([rad, el0, az0], lights, blur0, blending0, default_bg_setting=False,
                                      reuse_camera_target=True)
            if closer_idx == 1:
                _ind, _pos, _yaw = ind1, pos1, yaw1
            else:
                _ind, _pos, _yaw = ind0, pos0, yaw0
            rend.showModel(_ind, pos=_pos, yaw=_yaw)

            image0_closer_mask = copy.deepcopy(_im00)
            r_flat = image0_closer_mask[:, :, 0].flatten()
            g_flat = image0_closer_mask[:, :, 1].flatten()
            b_flat = image0_closer_mask[:, :, 2].flatten()
            mf_rgb = np.array((
              r_flat[np.histogram(r_flat)[0].argmax()],
              g_flat[np.histogram(g_flat)[0].argmax()],
              b_flat[np.histogram(b_flat)[0].argmax()],
            ))
            zero_indices = np.where(np.all(image0_closer_mask == mf_rgb, axis=-1))

            image0_closer_mask = np.ones_like(image0_closer_mask[:, :, 0])
            image0_closer_mask[zero_indices] = 0
            image0_farther_mask = copy.deepcopy(_im01)
            r_flat = image0_farther_mask[:, :, 0].flatten()
            g_flat = image0_farther_mask[:, :, 1].flatten()
            b_flat = image0_farther_mask[:, :, 2].flatten()
            mf_rgb = np.array((
              r_flat[np.histogram(r_flat)[0].argmax()],
              g_flat[np.histogram(g_flat)[0].argmax()],
              b_flat[np.histogram(b_flat)[0].argmax()],
            ))
            zero_indices = np.where(np.all(image0_farther_mask == mf_rgb, axis=-1))

            image0_farther_mask = np.ones_like(image0_farther_mask[:, :, 0])
            image0_farther_mask[zero_indices] = 0

            image0_farther_mask[image0_closer_mask > 0] = 0
            im0_closer_max = image0_closer_mask.max()
            if im0_closer_max == 0:
                print('error: current image all one color?')
                im0_closer_max = 1
            im0_farther_max = image0_farther_mask.max()
            if im0_farther_max == 0:
                print('error: current image all one color?')
                im0_farther_max = 1
            image0_closer_mask = (255.0 / im0_closer_max * (image0_closer_mask - image0_closer_mask.min())).astype(np.uint8)
            image0_farther_mask = (255.0 / im0_farther_max * (image0_farther_mask - image0_farther_mask.min())).astype(np.uint8)

            if closer_idx == 1:
                image0_mask0, image0_mask1 = image0_farther_mask, image0_closer_mask
            else:
                image0_mask0, image0_mask1 = image0_closer_mask, image0_farther_mask

        if save_images_as_pngs:
            mask0_v0_path = os.path.join(output_path, 'image0_mask0.png')
            mask1_v0_path = os.path.join(output_path, 'image0_mask1.png')
            Image.fromarray(image0_mask0).save(mask0_v0_path)
            Image.fromarray(image0_mask1).save(mask1_v0_path)

        # A randomly rotated view of the same two models
        # --------------------------------------------
        multiplier = (-1, 1)

        el1 = el0 + random.choice(multiplier) * np.random.normal(mean_abs_displacement[0], mean_abs_displacement[0])
        az1 = az0 + random.choice(multiplier) * np.random.normal(mean_abs_displacement[1], mean_abs_displacement[1])

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
        elif MASK_APPROACH == 'closer':
            rend.showModel(ind0, pos=pos0, yaw=yaw0)
            az1_rad = math.radians(az1)
            el1_rad = math.radians(el1)
            camera_pos1 = np.array([
                rad * math.cos(el1_rad) * math.cos(az1_rad),
                rad * math.cos(el1_rad) * math.sin(az1_rad),
                rad * math.sin(el1_rad)
            ])
            closer_idx = int(
                l2_dist(camera_pos1, pos1 + (0,)) < l2_dist(camera_pos1, pos0 + (0,)))
            rend.hideModel(ind0 if closer_idx == 1 else ind1)
            _im10, _ = rend.renderView([rad, el1, az1], lights, blur1, blending1, default_bg_setting=False,
                                       reuse_camera_target=True)
            rend.hideModel(ind1 if closer_idx == 1 else ind0)
            if closer_idx == 1:
                _ind, _pos, _yaw = ind0, pos0, yaw0
            else:
                _ind, _pos, _yaw = ind1, pos1, yaw1
            rend.showModel(_ind, pos=_pos, yaw=_yaw)
            _im11, _ = rend.renderView([rad, el1, az1], lights, blur1, blending1, default_bg_setting=False,
                                       reuse_camera_target=True)
            if closer_idx == 1:
                _ind, _pos, _yaw = ind1, pos1, yaw1
            else:
                _ind, _pos, _yaw = ind0, pos0, yaw0
            rend.showModel(_ind, pos=_pos, yaw=_yaw)

            image1_closer_mask = copy.deepcopy(_im10)
            r_flat = image1_closer_mask[:, :, 0].flatten()
            g_flat = image1_closer_mask[:, :, 1].flatten()
            b_flat = image1_closer_mask[:, :, 2].flatten()
            mf_rgb = np.array((
              r_flat[np.histogram(r_flat)[0].argmax()],
              g_flat[np.histogram(g_flat)[0].argmax()],
              b_flat[np.histogram(b_flat)[0].argmax()],
            ))
            zero_indices = np.where(np.all(image1_closer_mask == mf_rgb, axis=-1))

            image1_closer_mask = np.ones_like(image1_closer_mask[:, :, 0])
            image1_closer_mask[zero_indices] = 0

            image1_farther_mask = copy.deepcopy(_im11)
            r_flat = image1_farther_mask[:, :, 0].flatten()
            g_flat = image1_farther_mask[:, :, 1].flatten()
            b_flat = image1_farther_mask[:, :, 2].flatten()
            mf_rgb = np.array((
              r_flat[np.histogram(r_flat)[0].argmax()],
              g_flat[np.histogram(g_flat)[0].argmax()],
              b_flat[np.histogram(b_flat)[0].argmax()],
            ))
            zero_indices = np.where(np.all(image1_farther_mask == mf_rgb, axis=-1))

            image1_farther_mask = np.ones_like(image1_farther_mask[:, :, 0])
            image1_farther_mask[zero_indices] = 0
            image1_farther_mask[image1_closer_mask > 0] = 0
            im1_closer_max = image1_closer_mask.max()
            if im1_closer_max == 0:
                print('error: current image all one color?')
                im1_closer_max = 1
            im1_farther_max = image1_farther_mask.max()
            if im1_farther_max == 0:
                print('error: current image all one color?')
                im1_farther_max = 1
            image1_closer_mask = (
                255.0 / im1_closer_max * (image1_closer_mask - image1_closer_mask.min())).astype(np.uint8)
            image1_farther_mask = (
                255.0 / im1_farther_max * (image1_farther_mask - image1_farther_mask.min())).astype(np.uint8)

            if closer_idx == 1:
                image1_mask0, image1_mask1 = image1_farther_mask, image1_closer_mask
            else:
                image1_mask0, image1_mask1 = image1_closer_mask, image1_farther_mask

        if save_images_as_pngs:
            mask0_v1_path = os.path.join(output_path, 'image1_mask0.png')
            mask1_v1_path = os.path.join(output_path, 'image1_mask1.png')
            Image.fromarray(image1_mask0).save(mask0_v1_path)
            Image.fromarray(image1_mask1).save(mask1_v1_path)

            # for debugging
            output_name = os.path.join(output_path, 'mask_original.png')
            scipy.misc.toimage(_im1, cmin=0, cmax=255).save(output_name)

        rend.hideModel(ind0)
        rend.hideModel(ind1)

        _displacement_np = np.array([el1, az1]) - np.array([el0, az0])

        max_16bit_val = 65535
        out.image0.append(im0.astype(np.float32)/255.),

        out.image0_mask0.append(image0_mask0.astype(np.float32) / 255.),
        out.image0_mask1.append(image0_mask1.astype(np.float32) / 255.),

        out.image1.append(im1.astype(np.float32)/255.),

        out.image1_only0.append(im2.astype(np.float32)/255.),
        out.image1_only1.append(im3.astype(np.float32)/255.),

        out.image1_mask0.append(image1_mask0.astype(np.float32)/255.),
        out.image1_mask1.append(image1_mask1.astype(np.float32)/255.),

        out.depth0.append(dm0.astype(np.float32)/max_16bit_val),
        out.depth1.append(dm1.astype(np.float32)/max_16bit_val),

        out.depth1_only0.append(dm2.astype(np.float32)/max_16bit_val),
        out.depth1_only1.append(dm3.astype(np.float32)/max_16bit_val),

        out.displacement.append(_displacement_np.astype(np.float32)) # (elevation, azimuth)

    rend.delete()
    return out

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
        self.mean_abs_displacement = (10,10)  #####

        self.lock = threading.RLock()
        super(Render_thread, self).__init__()

    def set_mean_displacement(self, mean):
        with self.lock:
            self.mean_abs_displacement = mean

    def run(self):
        # ray.init()
        print 'starting pool'
        p = Pool()
        arg_list = [[self.bam_path, self.model_names_subset, self.mean_abs_displacement] for _ in range(NUM_PROC)]
        while True:
            # multiproc version
            results = p.map_async(render_worker, arg_list).get()
            # render_worker(arg_list[0])

            print 'received results'
            # ray version
            # object_id_list = [render_worker.remote(self.bam_path,
            #                                 self.model_names_subset,
            #                                 self.mean_abs_displacement) for _ in range(NUM_PROC)]
            # feedict = []
            # results = ray.get(object_id_list)

            feedict = {}
            for key in results[0].__dict__.keys():
                concat = []
                for r in results:
                    attrib = getattr(r, key)
                    attrib = np.stack(attrib, axis=0)
                    concat.append(attrib)
                concat = np.concatenate(concat, axis=0)
                feedict[self.placeholders[key]] = concat

            # max_16bit_val = 65535
            # placeholders = self.placeholders
            # feed_dict = {
            # placeholders['image0']: im0.astype(np.float32)/255.,
            # placeholders['image0_mask0']: image0_mask0.astype(np.float32) / 255.,
            # placeholders['image0_mask1']: image0_mask1.astype(np.float32) / 255.,
            # placeholders['image1']: im1.astype(np.float32)/255.,
            # placeholders['image1_only0']: im2.astype(np.float32)/255.,
            # placeholders['image1_only1']: im3.astype(np.float32)/255.,
            # placeholders['image1_mask0']: image1_mask0.astype(np.float32)/255.,
            # placeholders['image1_mask1']: image1_mask1.astype(np.float32)/255.,
            # placeholders['depth0']:dm0.astype(np.float32)/max_16bit_val,
            # placeholders['depth1']:dm1.astype(np.float32)/max_16bit_val,
            # placeholders['depth1_only0']:dm2.astype(np.float32)/max_16bit_val,
            # placeholders['depth1_only1']:dm3.astype(np.float32)/max_16bit_val,
            # placeholders['displacement']:_displacement_np.astype(np.float32)}   # (elevation, azimuth)

            self.sess.run(self.enqueue_op, feed_dict=feedict)

            print 'finished enqueue'


def write_tf_records(savedir, num_examples, bam_path, model_names_subset, mean_abs_displacement):
    p = Pool()
    arg_list = [[bam_path, model_names_subset, mean_abs_displacement] for _ in range(NUM_PROC)]

    ex_per_file = NUM_PROC/IM_PER_PROC
    num_runs = num_examples/ex_per_file

    ex_index = 0
    for i_ex in range(num_runs):

        results = p.map_async(render_worker, arg_list).get()
        print 'received results'

        feedict = {}
        out = Output()
        for key in results[0].__dict__.keys():
            concat = []
            for r in results:
                attrib = getattr(r, key)
                attrib = np.stack(attrib, axis=0)
                concat.append(attrib)
            concat = np.concatenate(concat, axis=0)
            setattr(out, key, concat)

        filename = savedir + '/{}_to_{}.tfrecords'.format(ex_index, ex_index + ex_per_file -1)
        ex_index += ex_per_file
        writer = tf.python_io.TFRecordWriter(filename)
        for b in range(IM_PER_PROC*NUM_PROC):
            # Write everything to TFRecords
            example = tf.train.Example(features=tf.train.Features(feature={
                'image0': _bytes_feature(tf.compat.as_bytes(out.image0[b].tostring())),
                'image0_mask0': _bytes_feature(tf.compat.as_bytes(out.image0_mask0[b].tostring())),
                'image0_mask1': _bytes_feature(tf.compat.as_bytes(out.image0_mask1[b].tostring())),
                'image1': _bytes_feature(tf.compat.as_bytes(out.image1[b].tostring())),
                'image1_only0': _bytes_feature(tf.compat.as_bytes(out.image1_only0[b].tostring())),
                'image1_only1': _bytes_feature(tf.compat.as_bytes(out.image1_only1[b].tostring())),
                'image1_mask0': _bytes_feature(tf.compat.as_bytes(out.image1_mask0[b].tostring())),
                'image1_mask1': _bytes_feature(tf.compat.as_bytes(out.image1_mask1[b].tostring())),
                'depth0': _bytes_feature(tf.compat.as_bytes(out.depth0[b].tostring())),
                'depth1': _bytes_feature(tf.compat.as_bytes(out.depth1[b].tostring())),
                'depth1_only0': _bytes_feature(tf.compat.as_bytes(out.depth1_only0[b].tostring())),
                'depth1_only1': _bytes_feature(tf.compat.as_bytes(out.depth1_only1[b].tostring())),
                'displacement': _float_feature(out.displacement[b]),  # (elevation, azimuth)
            }))
            writer.write(example.SerializeToString())
        writer.close()

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

        feed_batch_size = NUM_PROC * IM_PER_PROC
        rgb_shape = (feed_batch_size, 128,128,3)
        single_channel = (feed_batch_size, 128,128)
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
        placeholders['displacement'] = tf.placeholder(tf.float32, name='displacement', shape=[feed_batch_size, 2])

        shapes = [placeholders[key].get_shape().as_list()[1:] for key in placeholders.keys()]
        dtypes = [placeholders[key].dtype for key in placeholders.keys()]
        self.q = tf.FIFOQueue(self.batch_size*20, dtypes, shapes=shapes)

        placeholder_list = [placeholders[key] for key in placeholders.keys()]
        self.enqueue_op = self.q.enqueue_many(placeholder_list)

        self.placeholders = placeholders
        self.render_thread = Render_thread(self.conf, self.sel_files, self.placeholders,
                                           self.enqueue_op, self.sess)
        self.render_thread.setDaemon(True)
        self.render_thread.start()

        self.image0, self.image0_mask0, self.image0_mask1, self.image1, self.image1_only0, \
        self.image1_only1, self.image1_mask0, self.image1_mask1, self.depth0, self.depth1, \
        self.depth1_only0, self.depth1_only1, self.displacement = self.q.dequeue_many(self.batch_size)

    def set_mean_displacement(self, mean_disp):
        print 'setting mean displacment to ', mean_disp
        self.render_thread.set_mean_displacement(mean_disp)

def no_textures(bam_path, modelname):
    f = open(os.path.join(bam_path, modelname))
    try:
        s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    except ValueError:
        os.remove(os.path.join(bam_path, modelname))
        return False
    return s.find('texture') == -1 and s.find('Texture') == -1

def test_online_renderer():
    sess = tf.InteractiveSession()
    conf = {}

    conf['batch_size'] = 64
    conf['data_dir'] = '/home/frederik/Documents/catkin_ws/src/dynamic_multiview_3d/trainingdata/cardataset_bam'

    r = OnlineRenderer('train', conf, sess)
    nruns = 10
    # tstart = time.time()

    for i_run in range(nruns):
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

    # avgt  = (time.time() - tstart)/(nruns)
    # print 'aver time per batch {} seconds'.format(avgt)

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

        # r.set_mean_displacement(tuple(np.array([5,5])*(i_run+2)))

if __name__ == "__main__":
    test_online_renderer()
    # sys.exit(render_thread()