import rospy
import roslib
roslib.load_manifest('collect_data')
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

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
    return np.asarray(bridge.imgmsg_to_cv2(img))
  else:
    img.step = img.width * 3
    try:
      cv_img = bridge.imgmsg_to_cv2(img, 'rgb8')
    except CvBridgeError as e:
      print(e); return
    return np.asarray(cv_img).astype(np.float32)  ## TODO:shouldn't this be uint8 ?
