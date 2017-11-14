import rospy
import roslib
roslib.load_manifest('collect_data')
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import shutil

import os
import traceback

def make_dir(path):
  real_path = os.path.expanduser(path)
  if not os.path.exists(real_path):
    os.makedirs(real_path)
  return real_path

def service_handler(handler, parse_req=lambda x: x):
  def handler_func(req):
    success = False
    try:
      handler(**parse_req(req))
      success = True
    except Exception as e:
      rospy.logerr('Service handling failed! Exception: %s' %
                   traceback.format_exception_only(e.__class__, e)[0])
      traceback.print_exc()
    return success
  return handler_func

def persistent_service_proxy(topic, service, pool):
  def create_proxy():
    rospy.wait_for_service(topic, timeout=5)
    pool[topic] = rospy.ServiceProxy(topic, service, persistent=True)
  def service_caller(*args, **kwargs):
    try:
      res = pool[topic](*args, **kwargs)
    except rospy.ServiceException as e:
      rospy.logwarn('Service call %s failed. Error: %s' % (topic, e.message))
      pool[topic].close()
      create_proxy()
      res = service_caller(*args, **kwargs)
    return res
  assert topic not in pool, 'already exists'
  create_proxy()
  return service_caller

def from_sensor_msgs_img(img, depth=False):
  """Converts sensor_msgs/Image representation into a NumPy array."""
  bridge = CvBridge()
  if depth:
    _img = np.copy(bridge.imgmsg_to_cv2(img))
    _img[np.isnan(_img)] = 0
    return (255.0 / _img.max() * (_img - _img.min())).astype(np.uint8)
  else:
    img.step = img.width * 3
    try:
      cv_img = bridge.imgmsg_to_cv2(img, 'rgb8')
    except CvBridgeError as e:
      print(e)
      return
    return np.asarray(cv_img).astype(np.uint8)

def rm_rf(dir, require_confirmation=True):
  print('WARNING: about to delete the full contents of `%s`!' % dir)
  if require_confirmation:
    confirmation = raw_input('Are you sure you want to proceed? (True/False) ')
  else:
    confirmation = True
  if (isinstance(confirmation, bool) and confirmation) \
      or (isinstance(confirmation, str) and confirmation.lower() == 'true'):
    for filename in os.listdir(dir):
      filepath = os.path.join(dir, filename)
      try:
        if os.path.isfile(filepath):
          os.unlink(filepath)
        elif os.path.isdir(filepath):
          shutil.rmtree(filepath)
      except Exception as e:
        print(e)
    print('Successfully removed everything inside of `%s`.' % dir)
    return True
  else:
    print('Operation `rm -rf %s` aborted.' % dir)
    return False
