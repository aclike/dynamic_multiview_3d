import rospy
import roslib
roslib.load_manifest('collect_data')
import std_srvs.srv as std_srv

import os
import traceback

def make_dir(path):
  real_path = os.path.expanduser(path)
  if not os.path.exists(real_path):
    os.makedirs(real_path)
  return real_path

def service_handler(handler, service_class=std_srv.TriggerResponse, *args, **kwargs):
  def handler_func(req):
    success = False
    try:
      handler(*args, **kwargs)
      success = True
    except Exception as e:
      rospy.logerr('Service handling failed! Exception: %s' %
                   traceback.format_exception_only(e.__class__, e)[0])
      traceback.print_exc()
    return service_class(success=success)
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
