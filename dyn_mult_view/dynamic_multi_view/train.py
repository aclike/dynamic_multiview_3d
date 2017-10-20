import os
import numpy as np
import tensorflow as tf
import sys
import imp
from tensorflow.python.platform import flags
from datetime import datetime
from main_model import Base_Prediction_Model

from dyn_mult_view.mv3d.utils.tf_utils import load_snapshot

if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
  flags.DEFINE_string('visualize', '', 'model within hyperparameter folder from which to create gifs')
  flags.DEFINE_integer('device', 0, 'the value for CUDA_VISIBLE_DEVICES variable')
  flags.DEFINE_string('pretrained', None, 'path to model file from which to resume training')

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 400

# How often to run a batch through the validation model.
VAL_INTERVAL = 500

# How often to save a model checkpoint
SAVE_INTERVAL = 4000


def main(unused_argv, conf_script=None):
  os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
  print 'using CUDA_VISIBLE_DEVICES=', FLAGS.device
  from tensorflow.python.client import device_lib
  print device_lib.list_local_devices()

  if conf_script == None:
    conf_file = FLAGS.hyper
  else:
    conf_file = conf_script

  if not os.path.exists(FLAGS.hyper):
    sys.exit("Experiment configuration not found")
  hyperparams = imp.load_source('hyperparams', conf_file)

  conf = hyperparams.configuration

  if FLAGS.visualize:
    print 'creating visualizations ...'
    conf['data_dir'] = '/'.join(str.split(conf['data_dir'], '/')[:-1] + ['test'])
    conf['event_log_dir'] = '/tmp'
    conf['batch_size'] = 5
    conf['sequence_length'] = 14
    if FLAGS.diffmotions:
      conf['sequence_length'] = 30

  if 'model' in conf:
    Model = conf['model']
  else:
    Model = Base_Prediction_Model

  model = Model(conf, load_tfrec=True)

  print 'Constructing saver.'
  # Make saver.

  vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  saver = tf.train.Saver(vars, max_to_keep=0)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
  # Make training session.
  sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
  summary_writer = tf.summary.FileWriter(conf['output_dir'], graph=sess.graph, flush_secs=10)

  tf.train.start_queue_runners(sess)
  sess.run(tf.global_variables_initializer())

  if conf['visualize']:
    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
      print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    saver.restore(sess, conf['visualize'])
    print 'restore done.'

    model.visualize(sess, 0)


  itr_0 = 0
  if FLAGS.pretrained != None:
    if FLAGS.pretrained == 'True':
      load_snapshot(saver, sess, conf['output_dir'])
    else:
      conf['pretrained_model'] = FLAGS.pretrained
      saver.restore(sess, conf['pretrained_model'])



  print '-------------------------------------------------------------------'
  print 'verify current settings!! '
  for key in conf.keys():
    print key, ': ', conf[key]
  print '-------------------------------------------------------------------'

  tf.logging.info('iteration number, cost')

  starttime = datetime.now()
  t_iter = []
  # Run training.

  for itr in range(itr_0, conf['num_iterations'], 1):
    t_startiter = datetime.now()
    # Generate new batch of data_files.
    feed_dict = {model.iter_num: np.float32(itr),
                 model.train_cond: 1}

    cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
                                    feed_dict)

    if (itr) % 10 == 0:
      tf.logging.info(str(itr) + ' ' + str(cost))

    if (itr) % VAL_INTERVAL == 2:
      # Run through validation set.
      feed_dict = {model.iter_num: np.float32(itr),
                   model.train_cond: 0}
      [val_summary_str] = sess.run([model.summ_op], feed_dict)
      summary_writer.add_summary(val_summary_str, itr)

    if (itr) % SAVE_INTERVAL == 2:
      tf.logging.info('Saving model to' + conf['output_dir'])
      saver.save(sess, conf['output_dir'] + '/model' + str(itr))

    t_iter.append((datetime.now() - t_startiter).seconds * 1e6 + (datetime.now() - t_startiter).microseconds)

    if itr % 100 == 1:
      hours = (datetime.now() - starttime).seconds / 3600
      tf.logging.info('running for {0}d, {1}h, {2}min'.format(
        (datetime.now() - starttime).days,
        hours, +
               (datetime.now() - starttime).seconds / 60 - hours * 60))
      avg_t_iter = np.sum(np.asarray(t_iter)) / len(t_iter)
      tf.logging.info('time per iteration: {0}'.format(avg_t_iter / 1e6))
      tf.logging.info('expected for complete training: {0}h '.format(avg_t_iter / 1e6 / 3600 * conf['num_iterations']))

    if (itr) % SUMMARY_INTERVAL:
      summary_writer.add_summary(summary_str, itr)

  tf.logging.info('Saving model.')
  saver.save(sess, conf['output_dir'] + '/model')
  tf.logging.info('Training complete')
  tf.logging.flush()
