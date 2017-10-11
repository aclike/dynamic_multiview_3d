#!/usr/bin/env python

import rospy
import roslib
roslib.load_manifest('gazebo_msgs')
roslib.load_manifest('geometry_msgs')
import gazebo_msgs.srv as gazebo_srv
import geometry_msgs.msg as geometry_msg

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

class ObjectManager(object):
  def __init__(self):
    rospy.init_node('object_manager', anonymous=True)
    self._service_proxies = {}
    self._call_spawn_model = persistent_service_proxy(
      'gazebo/spawn_sdf_model', gazebo_srv.SpawnModel, self._service_proxies)
    self._call_delete_model = persistent_service_proxy(
      'gazebo/delete_model', gazebo_srv.DeleteModel, self._service_proxies)
    self.active_models = []

    models = {
      'distorted_camera': {
        'model_sdf_file': '/home/owen/.gazebo/models/distorted_camera/model.sdf',
        'position': [0, 0, 1.0],
        'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
      },
      'airplane': {
        'model_sdf_file': '/home/owen/.gazebo/models/airplane/model.sdf',
        'position': [18.0, 2.0, 0],
        'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
      },
    }

    for model_name, model_info in models.items():
      with open(model_info['model_sdf_file']) as f:
        model_sdf = f.read()
      self._call_spawn_model(
        model_name=model_name, model_xml=model_sdf,
        initial_pose=geometry_msg.Pose(
          position=geometry_msg.Point(*model_info['position']),
          orientation=geometry_msg.Quaternion(**model_info['orientation'])
        )
      )
      self.active_models.append(model_name)

  def listen(self):
    try:
      while not rospy.is_shutdown():
        rospy.sleep(0.1)
    finally:
      for model_name in self.active_models:
        self._call_delete_model(model_name=model_name)

if __name__ == '__main__':
  object_manager = ObjectManager()
  try:
    object_manager.listen()
  except rospy.ROSInterruptException:
    pass
