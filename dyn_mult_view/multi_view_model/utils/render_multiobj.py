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

bam_path = "../obj_cars2"
output_path = "../shapenet_output"
num_examples = 10
save_images_as_pngs = True  # possibly only for debugging


def _bytes_feature(value):
    if not isinstance(value, (np.ndarray, list, tuple)):
      value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    if not isinstance(value, (np.ndarray, list, tuple)):
      value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def main():

    model_names = os.listdir(bam_path)
    model_count = len(model_names)

    rend = Renderer(True, False)
    rend.loadModels(model_names[0:model_count], bam_path)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    curr_tfr_path = os.path.join(output_path, 'data_0.tfrecords')
    writer = tf.python_io.TFRecordWriter(curr_tfr_path)

    for _i in range(num_examples):
        ind0, ind1 = random.randint(0, model_count - 1), random.randint(0, model_count - 1)
        while ind0 == ind1:
            ind1 = random.randint(0, model_count - 1)
        pos0 = rend.showModel(ind0, debug=False)
        pos1 = rend.showModel(ind1, debug=False)

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
        rend.hideModel(ind1)
        rend.showModel(ind0, pos=pos0, color='red')
        rend.showModel(ind1, pos=pos1, color='green')
        _im0, _ = rend.renderView([rad, el0, az0], lights, blur0, blending0, default_bg_setting=False, reuse_camera_target=True)

        mask0_v0 = copy.deepcopy(_im0)[:, :, 0]
        mask0_v0[mask0_v0 >= 128] = 255
        mask0_v0[mask0_v0 <  128] = 0
        mask0_v0 = (255.0 / mask0_v0.max() * (mask0_v0 - mask0_v0.min())).astype(np.uint8)
        mask1_v0 = copy.deepcopy(_im0)[:, :, 1]
        mask1_v0[mask1_v0 >= 128] = 255
        mask1_v0[mask1_v0 <  128] = 0
        mask1_v0 = (255.0 / mask1_v0.max() * (mask1_v0 - mask1_v0.min())).astype(np.uint8)

        if save_images_as_pngs:
            mask0_v0_path = os.path.join(output_path, 'mask0_v0_view0.png')
            mask1_v0_path = os.path.join(output_path, 'mask1_v0_view0.png')
            Image.fromarray(mask0_v0).save(mask0_v0_path)
            Image.fromarray(mask1_v0).save(mask1_v0_path)

        rend.hideModel(ind0)
        rend.hideModel(ind1)
        rend.showModel(ind0, pos=pos0)
        rend.showModel(ind1, pos=pos1)

        # A randomly rotated view of the same two models
        # --------------------------------------------
        el1 = int(random.random() * 50 - 10)
        az1 = int(random.random() * 20) + 90  # reduced in order to better guarantee occlusion
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
        rend.showModel(ind1, pos=pos1)
        im3, dm3 = rend.renderView([rad, el1, az1], lights, blur1, blending1, default_bg_setting=False, reuse_camera_target=True)

        if save_images_as_pngs:
            output_name = os.path.join(output_path, 'rgb_%d_3.png' % _i)
            scipy.misc.toimage(im3, cmin=0, cmax=255).save(output_name)
            output_name = os.path.join(output_path, 'depth_%d_3.png' % _i)
            scipy.misc.toimage(dm3, cmin=0, cmax=65535, low=0, high=65535, mode='I').save(output_name)

        # Per-object masks of the models (part 2)
        # ---------------------------------------
        rend.hideModel(ind1)
        rend.showModel(ind0, pos=pos0, color='red')
        rend.showModel(ind1, pos=pos1, color='green')
        _im1, _ = rend.renderView([rad, el1, az1], lights, blur1, blending1, default_bg_setting=False, reuse_camera_target=True)

        mask0_v1 = copy.deepcopy(_im1)[:, :, 0]
        mask0_v1[mask0_v1 >= 128] = 255
        mask0_v1[mask0_v1 <  128] = 0
        mask0_v1 = (255.0 / mask0_v1.max() * (mask0_v1 - mask0_v1.min())).astype(np.uint8)
        mask1_v1 = copy.deepcopy(_im1)[:, :, 1]
        mask1_v1[mask1_v1 >= 128] = 255
        mask1_v1[mask1_v1 <  128] = 0
        mask1_v1 = (255.0 / mask1_v1.max() * (mask1_v1 - mask1_v1.min())).astype(np.uint8)

        if save_images_as_pngs:
            mask0_v1_path = os.path.join(output_path, 'mask0_v1_view1.png')
            mask1_v1_path = os.path.join(output_path, 'mask1_v1_view1.png')
            Image.fromarray(mask0_v1).save(mask0_v1_path)
            Image.fromarray(mask1_v1).save(mask1_v1_path)

            # for debugging
            output_name = os.path.join(output_path, 'mask_original.png')
            scipy.misc.toimage(_im1, cmin=0, cmax=255).save(output_name)

        rend.hideModel(ind0)
        rend.hideModel(ind1)

        # Write everything to TFRecords
        _displacement_np = np.array([el1, az1]) - np.array([el0, az0])
        example = tf.train.Example(features=tf.train.Features(feature={
            'image0': _bytes_feature(tf.compat.as_bytes(im0.tostring())),
            'image0_mask0': _bytes_feature(tf.compat.as_bytes(mask0_v0.tostring())),
            'image0_mask1': _bytes_feature(tf.compat.as_bytes(mask1_v0.tostring())),
            'image1': _bytes_feature(tf.compat.as_bytes(im1.tostring())),
            'image1_only0': _bytes_feature(tf.compat.as_bytes(im2.tostring())),
            'image1_only1': _bytes_feature(tf.compat.as_bytes(im3.tostring())),
            'image1_mask0': _bytes_feature(tf.compat.as_bytes(mask0_v1.tostring())),
            'image1_mask1': _bytes_feature(tf.compat.as_bytes(mask1_v1.tostring())),
            'depth0': _bytes_feature(tf.compat.as_bytes(dm0.tostring())),
            'depth1': _bytes_feature(tf.compat.as_bytes(dm1.tostring())),
            'depth1_only0': _bytes_feature(tf.compat.as_bytes(dm2.tostring())),
            'depth1_only1': _bytes_feature(tf.compat.as_bytes(dm3.tostring())),
            'displacement': _float_feature(_displacement_np),  # (elevation, azimuth)
        }))
        writer.write(example.SerializeToString())

    writer.close()

if __name__ == "__main__":
    sys.exit(main())
