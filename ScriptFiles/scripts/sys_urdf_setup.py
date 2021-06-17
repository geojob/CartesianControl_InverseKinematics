#!/usr/bin/env python3
# coding: UTF-8
import sys
l1ll111_opy_ = sys.version_info [0] == 2
l1111l11_opy_ = 2048
l1l1l1_opy_ = 7
def l1l1ll11_opy_ (l111111l_opy_):
    global l11111l_opy_
    l1llll11l_opy_ = ord (l111111l_opy_ [-1])
    l1llll11_opy_ = l111111l_opy_ [:-1]
    l11ll1l1_opy_ = l1llll11l_opy_ % len (l1llll11_opy_)
    l1111lll_opy_ = l1llll11_opy_ [:l11ll1l1_opy_] + l1llll11_opy_ [l11ll1l1_opy_:]
    if l1ll111_opy_:
        l1lll1_opy_ = unicode () .join ([unichr (ord (char) - l1111l11_opy_ - (l1lll1ll_opy_ + l1llll11l_opy_) % l1l1l1_opy_) for l1lll1ll_opy_, char in enumerate (l1111lll_opy_)])
    else:
        l1lll1_opy_ = str () .join ([chr (ord (char) - l1111l11_opy_ - (l1lll1ll_opy_ + l1llll11l_opy_) % l1l1l1_opy_) for l1lll1ll_opy_, char in enumerate (l1111lll_opy_)])
    return eval (l1lll1_opy_)
import rospy
import numpy
import tf
import time
from sensor_msgs.msg import JointState
from cartesian_control.msg import CartesianCommand
import geometry_msgs.msg
from geometry_msgs.msg import Transform
from visualization_msgs.msg import InteractiveMarkerControl
from visualization_msgs.msg import Marker
# from interactive_markers.interactive_marker_server import *
from urdf_parser_py.urdf import URDF
def l1ll1l1l1_opy_(T):
    t = geometry_msgs.msg.Pose()
    position = tf.transformations.translation_from_matrix(T)
    orientation = tf.transformations.quaternion_from_matrix(T)
    t.position.x = position[0]
    t.position.y = position[1]
    t.position.z = position[2]
    t.orientation.x = orientation[0]
    t.orientation.y = orientation[1]
    t.orientation.z = orientation[2]
    t.orientation.w = orientation[3]
    return t
def is_same_trans(l1lll11l1_opy_, l1lll1l11_opy_):
    return (abs(numpy.subtract(l1lll11l1_opy_, l1lll1l11_opy_)) < 10e-3).all()
def is_same(matrix0, matrix1):
    matrix0 = numpy.array(matrix0, dtype=numpy.float64, copy=True)
    matrix0 /= matrix0[3, 3]
    matrix1 = numpy.array(matrix1, dtype=numpy.float64, copy=True)
    matrix1 /= matrix1[3, 3]
    return numpy.allclose(matrix0, matrix1, 0, 1e-2)
def l1lll1111_opy_(T):
    t = Transform()
    position = tf.transformations.translation_from_matrix(T)
    orientation = tf.transformations.quaternion_from_matrix(T)
    t.translation.x = position[0]
    t.translation.y = position[1]
    t.translation.z = position[2]
    t.rotation.x = orientation[0]
    t.rotation.y = orientation[1]
    t.rotation.z = orientation[2]
    t.rotation.w = orientation[3]
    return t
class CartesianGrader(object):
    #l1ll1llll_opy_
    def __init__(self):
        self.l1ll1lll1_opy_ = rospy.Publisher(l1l1ll11_opy_ (u"ࠧ࠵ࡪࡰ࡫ࡱࡸࡤࡩ࡯࡮࡯ࡤࡲࡩࠨࡿ"), JointState, queue_size=1)
        #Publisher to l1lll11ll_opy_ l1ll1ll1l_opy_
        self.l1lll1lll_opy_ = rospy.Publisher(l1l1ll11_opy_ (u"ࠨ࠯ࡤࡣࡵࡸࡪࡹࡩࡢࡰࡢࡧࡴࡳ࡭ࡢࡰࡧࠦࢀ"), CartesianCommand, queue_size=1)
        self.l1ll1ll11_opy_ = rospy.Publisher(l1l1ll11_opy_ (u"ࠢ࠰࡫࡮ࡣࡨࡵ࡭࡮ࡣࡱࡨࠧࢁ"), Transform, queue_size=1)
        rospy.Subscriber(l1l1ll11_opy_ (u"ࠨ࠱࡭ࡳ࡮ࡴࡴࡠࡵࡷࡥࡹ࡫ࡳࠨࢂ"), JointState, self.get_joint_state)
        self.listener = tf.TransformListener()
        self.robot = URDF.from_parameter_server()
        self.num_joints = 0
        self.joint_names = []
        self.joint_axes = []
        link = self.robot.get_root()
        while True:
            if link not in self.robot.child_map: break
            (joint_name, next_link) = self.robot.child_map[link][0]
            current_joint = self.robot.joint_map[joint_name]
            if current_joint.type != l1l1ll11_opy_ (u"ࠩࡩ࡭ࡽ࡫ࡤࠨࢃ"):
                self.num_joints = self.num_joints + 1
                self.joint_names.append(current_joint.name)
                self.joint_axes.append(current_joint.axis)
            link = next_link
        self.l1lll1l1l_opy_ = link
    def get_joint_state(self, msg):
        self.l1ll1l1ll_opy_ = msg.position[0]
    def reset_robot(self):
        cmd = JointState()
        if self.robot.name == l1l1ll11_opy_ (u"ࠪࡹࡷ࠻ࠧࢄ"):
            cmd.position.append(1.9)
            cmd.position.append(0.65)
            cmd.position.append(1.5)
            cmd.position.append(1.15)
            cmd.position.append(0)
            cmd.position.append(0.3)
        else:
            cmd.position.append(0.0)
            cmd.position.append(1.0)
            cmd.position.append(0.0)
            cmd.position.append(1.0)
            cmd.position.append(0.0)
            cmd.position.append(0.0)
            cmd.position.append(0.0)
        self.l1ll1lll1_opy_.publish(cmd)
        rospy.sleep(1.0)
    def go_to_ik_pose(self, name, T, timeout):
        msg = l1lll1111_opy_(T)
        start_time = time.time()
        done = False
        self.l1ll1ll11_opy_.publish(msg)
        while not done and not rospy.is_shutdown():
            try:
                (trans,rot) = self.listener.lookupTransform(self.robot.get_root(),self.l1lll1l1l_opy_,
                                                            rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print(l1l1ll11_opy_ (u"࡙ࠦࡌࠠࡆࡺࡦࡩࡵࡺࡩࡰࡰࠤࠦࢅ"))
                continue
            l1lll111l_opy_ = numpy.dot(tf.transformations.translation_matrix(trans),
                           tf.transformations.quaternion_matrix(rot))
            if (is_same_trans(tf.transformations.translation_from_matrix(T), tf.transformations.translation_from_matrix(l1lll111l_opy_))):
               print(name + l1l1ll11_opy_ (u"ࠧࡀࠠࡑࡃࡖࡗࡊࡊࠢࢆ"))
               done = True
               return 1
            if (time.time() - start_time > timeout) :
                print(name + l1l1ll11_opy_ (u"ࠨ࠺ࠡࡔࡲࡦࡴࡺࠠࡵࡱࡲ࡯ࠥࡺ࡯ࡰࠢ࡯ࡳࡳ࡭ࠠࡵࡱࠣࡶࡪࡧࡣࡩࠢࡧࡩࡸ࡯ࡲࡦࡦࠣࡴࡴࡹࡥࠣࢇ"))
                print(l1l1ll11_opy_ (u"ࠢࡓࡱࡥࡳࡹࠦࡴࡰࡱ࡮ࠤࡹࡵ࡯ࠡ࡮ࡲࡲ࡬ࠦࡴࡰࠢࡵࡩࡦࡩࡨࠡࡦࡨࡷ࡮ࡸࡥࡥࠢࡳࡳࡸ࡫࠮ࠡࡉࡵࡥࡩ࡫ࡲࠡࡶ࡬ࡱࡪࡪࠠࡰࡷࡷࠦ࢈"))
                done = True
            else:
                rospy.sleep(0.1)
        return 0
    def go_to_pose(self, name, T, secondary_objective, q0_target, timeout, points):
        msg = l1lll1111_opy_(T)
        l1lll1ll1_opy_ = CartesianCommand()
        l1lll1ll1_opy_.x_target = msg
        l1lll1ll1_opy_.secondary_objective = secondary_objective
        l1lll1ll1_opy_.q0_target = q0_target
        start_time = time.time()
        done = False
        index = 0
        while not done and not rospy.is_shutdown():
            index += 1
            self.l1lll1lll_opy_.publish(l1lll1ll1_opy_)
            try:
                (trans,rot) = self.listener.lookupTransform(self.robot.get_root(),self.l1lll1l1l_opy_,
                                                            rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print(l1l1ll11_opy_ (u"ࠣࡖࡉࠤࡊࡾࡣࡦࡲࡷ࡭ࡴࡴࠡࠣࢉ"))
                continue
            l1lll111l_opy_ = numpy.dot(tf.transformations.translation_matrix(trans),
                           tf.transformations.quaternion_matrix(rot))
            if (is_same(T, l1lll111l_opy_)):
                if secondary_objective:
                    if self.robot.name == l1l1ll11_opy_ (u"ࠩࡸࡶ࠺࠭ࢊ") and index>30:
                        return points
                    else:
                        if abs(self.l1ll1l1ll_opy_-q0_target)< 10e-3:
                            #print(name + l1l1ll11_opy_ (u"ࠥ࠾ࠥࡖࡁࡔࡕࡈࡈࠧࢋ"))
                            return points
                else:
                    #print(name + l1l1ll11_opy_ (u"ࠦ࠿ࠦࡐࡂࡕࡖࡉࡉࠨࢌ"))
                    return points
            if (time.time() - start_time > timeout) :
                #print(name + l1l1ll11_opy_ (u"ࠧࡀࠠࡓࡱࡥࡳࡹࠦࡴࡰࡱ࡮ࠤࡹࡵ࡯ࠡ࡮ࡲࡲ࡬ࠦࡴࡰࠢࡵࡩࡦࡩࡨࠡࡦࡨࡷ࡮ࡸࡥࡥࠢࡳࡳࡸ࡫ࠢࢍ"))
                #print(l1l1ll11_opy_ (u"ࠨࡒࡰࡤࡲࡸࠥࡺ࡯ࡰ࡭ࠣࡸࡴࡵࠠ࡭ࡱࡱ࡫ࠥࡺ࡯ࠡࡴࡨࡥࡨ࡮ࠠࡥࡧࡶ࡭ࡷ࡫ࡤࠡࡲࡲࡷࡪ࠴ࠠࡈࡴࡤࡨࡪࡸࠠࡵ࡫ࡰࡩࡩࠦ࡯ࡶࡶࠥࢎ"))
                return 0
            else:
                rospy.sleep(0.1)
        return 0