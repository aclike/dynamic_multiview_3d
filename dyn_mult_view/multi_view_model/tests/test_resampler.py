from dyn_mult_view.mv3d.utils.tf_utils import *
from dyn_mult_view.multi_view_model.utils.read_tf_records import build_tfrecord_input

import pdb
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.contrib.resampler import resampler

# Test case to show the tf.contrib.resampler functionality. Takes a rectangle and applies a rotation, then interpolates to get the new points.
# The final result should be a rectangle rotated by about 10 degrees.

filename_queue = tf.train.string_input_producer(["rectangle.png"])
image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(filename_queue)
image = tf.image.decode_png(image_file)
data_inp = tf.cast(tf.expand_dims(image, 0), tf.float32)

angle = 10
rads = np.radians(angle)
c, s = np.cos(rads), np.sin(rads)
rotation = tf.convert_to_tensor(np.array([[c, -s], [s, c]]), dtype=tf.float32)

y = tf.cast(np.arange(1500), tf.float32)
x = tf.cast(np.arange(2100), tf.float32)
Y,X = tf.meshgrid(y,x,indexing='ij')
combined = tf.stack([X,Y], axis=2)

Y1 = tf.reshape(Y, [1500*2100])
X1 = tf.reshape(X, [1500*2100])
pts = tf.stack([X1,Y1], axis=1)
warp_pts = tf.matmul(pts, rotation)
warp_Y = warp_pts[:,1]
warp_X = warp_pts[:,0]
warp_Y1 = tf.reshape(warp_Y, [1500,2100])
warp_X1 = tf.reshape(warp_X, [1500,2100])
warp_Y2 = tf.clip_by_value(warp_Y1, 0, 1500)
warp_X2 = tf.clip_by_value(warp_X1, 0, 2100)
warp_combined = tf.stack([warp_X2,warp_Y2], axis=2)

warp = tf.cast(warp_combined, tf.float32)
warp_inp = tf.expand_dims(warp, 0)
resampled = tf.cast(resampler(data_inp, warp_inp), tf.uint8)

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.global_variables_initializer().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    output = sess.run(resampled)
    print output[0].shape

    plt.figure()
    plt.imshow(output[0])
    plt.show()

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)