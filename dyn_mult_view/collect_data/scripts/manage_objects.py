#!/usr/bin/env python

import roslib
import rospy

roslib.load_manifest('collect_data')
roslib.load_manifest('gazebo_msgs')
roslib.load_manifest('geometry_msgs')
roslib.load_manifest('tf')
import gazebo_msgs.srv as gazebo_srv
import gazebo_msgs.msg as gazebo_msg
import geometry_msgs.msg as geometry_msg
import collect_data.srv as collect_srv

import tf
import numpy as np
import utils

class ObjectManager(object):
  def __init__(self):
    rospy.init_node('object_manager', anonymous=True)
    self._service_proxies = {}
    self._call_spawn_model = utils.persistent_service_proxy(
      'gazebo/spawn_sdf_model', gazebo_srv.SpawnModel, self._service_proxies)
    self._call_delete_model = utils.persistent_service_proxy(
      'gazebo/delete_model', gazebo_srv.DeleteModel, self._service_proxies)
    self._call_set_model_state = utils.persistent_service_proxy(
      'gazebo/set_model_state', gazebo_srv.SetModelState, self._service_proxies)
    self._call_get_model_state = utils.persistent_service_proxy(
      'gazebo/get_model_state', gazebo_srv.GetModelState, self._service_proxies)
    self.active_models = []

    # Spawn the camera and any other initial models
    init_models = {
      'distorted_camera': {
        'model_sdf_file': '/home/owen/.gazebo/models/kinect/model.sdf',
        'position': [-1.7, 0, 5.0],
        'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
      },
    }
    for model_name, model_info in init_models.items():
      self.spawn_object(model_name, model_info['model_sdf_file'],
                        model_info['position'], model_info['orientation'])

    # Define services
    def parse_spawn_object_req(req):
      return {
        'model_name': req.model_name,
        'model_sdf_file': req.model_sdf_file,
        'pos_x': req.pos_x,
        'pos_y': req.pos_y,
        'pos_z': req.pos_z,
        'rot_w': req.rot_w,
        'rot_x': req.rot_x,
        'rot_y': req.rot_y,
        'rot_z': req.rot_z,
      }
    rospy.Service('/manage_objects/spawn_object', collect_srv.SpawnObject,
                  utils.service_handler(self.spawn_object_wrapper, parse_spawn_object_req))

    def parse_rotate_object_req(req):
      return {
        'model_name': req.model_name,
        'r': req.r,
        'p': req.p,
        'y': req.y,
      }
    rospy.Service('/manage_objects/rotate_object', collect_srv.RotateObject,
                  utils.service_handler(self.rotate_object_wrapper, parse_rotate_object_req))

    def parse_set_orientation_req(req):
      return {
        'model_name': req.model_name,
        'w': req.w,
        'x': req.x,
        'y': req.y,
        'z': req.z,
      }
    rospy.Service('/manage_objects/set_orientation', collect_srv.SetOrientation,
                  utils.service_handler(self.set_orientation, parse_set_orientation_req))

  def spawn_object_wrapper(self, model_name, model_sdf_file,
                           pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z):
    self.spawn_object(model_name, model_sdf_file, [pos_x, pos_y, pos_z],
                      {'w': rot_w, 'x': rot_x, 'y': rot_y, 'z': rot_z})
    return True

  def spawn_object(self, model_name, model_sdf_file, position, orientation):
    """
    MODEL_NAME - a string representing the name of the object to rotate
    MODEL_SDF_FILE - a string representing the path to the model's SDF file
    POSITION - a 3-tuple containing the XYZ coordinates of the spawn point
    ORIENTATION - a dict containing the desired orientation as a quaternion
    """
    with open(model_sdf_file) as f:
      model_sdf = f.read()
    self._call_spawn_model(
      model_name=model_name, model_xml=model_sdf,
      initial_pose=geometry_msg.Pose(
        position=geometry_msg.Point(*position),
        orientation=geometry_msg.Quaternion(**orientation)
      )
    )
    self.active_models.append(model_name)

  def rotate_object_wrapper(self, model_name, r, p, y):
    self.rotate_object(model_name, (r, p, y,))
    return True

  def rotate_object(self, model_name, rotation):
    """
    MODEL_NAME - a string representing the name of the object to rotate
    ROTATION - a 3-tuple containing RPY angles
    """
    model_state_raw = self._call_get_model_state(model_name=model_name)
    model_state = gazebo_msg.ModelState()
    model_state.model_name = model_name
    model_state.pose = model_state_raw.pose
    model_state.twist = model_state_raw.twist

    Rq = tf.transformations.quaternion_from_euler(*rotation)
    rotated = tf.transformations.quaternion_multiply(Rq, np.array([
      model_state.pose.orientation.x,
      model_state.pose.orientation.y,
      model_state.pose.orientation.z,
      model_state.pose.orientation.w,
    ]))

    model_state.pose.orientation = geometry_msg.Quaternion(*rotated)
    self._call_set_model_state(model_state=model_state)

  def set_orientation(self, model_name, w, x, y, z):
    model_state_raw = self._call_get_model_state(model_name=model_name)
    model_state = gazebo_msg.ModelState(
      model_name=model_name, pose=model_state_raw.pose, twist=model_state_raw.twist)
    model_state.pose.orientation = geometry_msg.Quaternion(x, y, z, w)
    self._call_set_model_state(model_state=model_state)
    return True

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
