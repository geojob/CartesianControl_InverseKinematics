#!/usr/bin/env python3

import math
import numpy
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Transform
from cartesian_control.msg import CartesianCommand
from urdf_parser_py.urdf import URDF
import random
import tf
from threading import Thread, Lock

'''This is a class which will perform both cartesian control and inverse
   kinematics'''
class CCIK(object):
    def __init__(self):
    #Load robot from parameter server
        self.robot = URDF.from_parameter_server()

    #Subscribe to current joint state of the robot
        rospy.Subscriber('/joint_states', JointState, self.get_joint_state)

    #This will load information about the joints of the robot
        self.num_joints = 0
        self.joint_names = []
        self.q_current = []
        self.joint_axes = []
        self.get_joint_info()

    #This is a mutex
        self.mutex = Lock()

    #Subscribers and publishers for for cartesian control
        rospy.Subscriber('/cartesian_command', CartesianCommand, self.get_cartesian_command)
        self.velocity_pub = rospy.Publisher('/joint_velocities', JointState, queue_size=10)
        self.joint_velocity_msg = JointState()

        #Subscribers and publishers for numerical IK
        rospy.Subscriber('/ik_command', Transform, self.get_ik_command)
        self.joint_command_pub = rospy.Publisher('/joint_command', JointState, queue_size=10)
        self.joint_command_msg = JointState()

    '''This is a function which will collect information about the robot which
       has been loaded from the parameter server. It will populate the variables
       self.num_joints (the number of joints), self.joint_names and
       self.joint_axes (the axes around which the joints rotate)'''
    def get_joint_info(self):
        link = self.robot.get_root()
        while True:
            if link not in self.robot.child_map: break
            (joint_name, next_link) = self.robot.child_map[link][0]
            current_joint = self.robot.joint_map[joint_name]
            if current_joint.type != 'fixed':
                self.num_joints = self.num_joints + 1
                self.joint_names.append(current_joint.name)
                self.joint_axes.append(current_joint.axis)
            link = next_link

    '''This is the callback which will be executed when the cartesian control
       recieves a new command. The command will contain information about the
       secondary objective and the target q0. At the end of this callback, 
       you should publish to the /joint_velocities topic.'''
    def get_cartesian_command(self, command):
        self.mutex.acquire()
        #--------------------------------------------------------------------------
        primarygain = 1
        secondarygain = 3
        
        curr_q_values = self.q_current
        
        des = command.x_target
        
        t = tf.transformations.translation_matrix(numpy.array([des.translation.x, des.translation.y, des.translation.z]))
        q = tf.transformations.quaternion_matrix(numpy.array([des.rotation.x, des.rotation.y, des.rotation.z, des.rotation.w]))
        
        
        b_T_desee = numpy.dot(t,q)
        
        
        
             
        joint_transforms, b_T_ee = self.forward_kinematics(curr_q_values)        
        b_T_ee_inv = tf.transformations.inverse_matrix(b_T_ee)  
        
        '''
        angle2,axis2 = self.rotation_from_matrix(b_T_ee[:3,:3])
        Rot_vec2 = numpy.dot(angle2,axis2)
        angle3,axis3 = self.rotation_from_matrix(b_T_desee[:3,:3])
        Rot_vec3 = numpy.dot(angle3,axis3)
        
        x1 = numpy.zeros((6,1))
        
        x1[:3,0] = b_T_ee[:3,3]
        x1[3:6,0] = Rot_vec2[0:3]
       
        
        x2 = numpy.zeros((6,1))
        
        x2[:3,0] = b_T_desee[:3,3]
        x2[3:6,0] = Rot_vec3[0:3]
     
        diff = x2-x1
        deltax2 = numpy.zeros((6,1))
        deltax2[:3,0] = diff[:3,0]
        deltax2[3:6,0] = diff[3:6,0]
        
        Rot_i = tf.transformations.inverse_matrix(b_T_ee[:3,:3])
        
        row1 = numpy.append(Rot_i,[[0,0,0],[0,0,0],[0,0,0]], axis = 1)
        row2 = numpy.append([[0,0,0],[0,0,0],[0,0,0]], Rot_i, axis = 1)
        
        conv_mat = numpy.append(row1, row2, axis=0)
        
        xdot2 = deltax2*primarygain
        xdot2 = numpy.dot(conv_mat,xdot2)
        print("This is difference computed")
        print(xdot2)
        '''
        
        ee_T_desee = numpy.dot(b_T_ee_inv,b_T_desee)
        
        angle,axis = self.rotation_from_matrix(ee_T_desee[:3,:3])
        
        Rot_vec = numpy.dot(angle,axis)
        
        
        
        deltax = numpy.zeros((6,1))
        
        deltax[:3,0] = ee_T_desee[:3,3]
        deltax[3:6,0] = Rot_vec[0:3]
        
        xdot = deltax*primarygain
        #print("This is transform computed")
        #print(xdot)
        max_vel = max(abs(xdot[:3,0]))
        max_rad = max(abs(xdot[3:6,0]))
        #print(xdot[:3,0])
        #print(xdot[3:6,0])
        #print(max_vel)
        #print(max_rad)
        if(numpy.linalg.norm(xdot[:3,0])>0.17321):
          xdot[0,0] = (xdot[0,0])*(0.1/max_vel)
          xdot[1,0] = (xdot[1,0])*(0.1/max_vel)
          xdot[2,0] = (xdot[2,0])*(0.1/max_vel)
        #print(xdot[:3,0])
        if(numpy.linalg.norm(xdot[3:6,0])>1):
          xdot[3,0] = (xdot[3,0])/max_rad
          xdot[4,0] = (xdot[4,0])/max_rad
          xdot[5,0] = (xdot[5,0])/max_rad
        
  	
        J = self.get_jacobian(b_T_ee, joint_transforms)
        J_plus_s = numpy.linalg.pinv(J,1.0e-2)
        #print(J_plus_s)
        
        qdot = numpy.dot(J_plus_s,xdot)
        
        if (command.secondary_objective == True):
        
          q0_tg = command.q0_target
          J_plus = numpy.linalg.pinv(J)
          #print(J_plus)
          qdot_sec = numpy.zeros((self.num_joints,1))
          qdot_sec[0,0] = secondarygain*(q0_tg - self.q_current[0])
          I_mat = numpy.identity(self.num_joints)
          
          proj_mat = I_mat - numpy.dot(J_plus,J)
          qdot_null = numpy.dot(proj_mat,qdot_sec)
          qdot = qdot + qdot_null
          
          
        
        max_rad2 = max(abs(qdot[0:6,0]))
        if (max_rad2 > 1):
          qdot = qdot/max_rad2
       
        
          
          
        
        #print(self.joint_axes)
        self.joint_velocity_msg.name = self.joint_names
        self.joint_velocity_msg.velocity = qdot
        
        self.velocity_pub.publish(self.joint_velocity_msg)
        
        
        
        
        #print("Transform")
        #print(ee_T_desee)
        #print("Axis")
        #print(axis)
        #print("Angle")
        #print(angle)
        #print("Rot_vec")
        #print(Rot_vec)
        #print("Deltax")
        #print(deltax)


        #--------------------------------------------------------------------------
        self.mutex.release()
        
    '''This is a function which will assemble the jacobian of the robot using the
       current joint transforms and the transform from the base to the end
       effector (b_T_ee). Both the cartesian control callback and the
       inverse kinematics callback will make use of this function.
       Usage: J = self.get_jacobian(b_T_ee, joint_transforms)'''
    def get_jacobian(self, b_T_ee, joint_transforms):
        J = numpy.zeros((6,self.num_joints))
        #--------------------------------------------------------------------------
        for i in range(len(J[0])):
            axis = self.joint_axes[i]
            b_T_currJ = joint_transforms[i]
            b_T_currJ_inv = tf.transformations.inverse_matrix(b_T_currJ)
            
            j_T_ee = numpy.dot(b_T_currJ_inv,b_T_ee)
            
            j_T_ee_inv = tf.transformations.inverse_matrix(j_T_ee)
            
            t_vec = j_T_ee[:3,3]
            
            r_mat = j_T_ee_inv[:3,:3]
            
            skew_mat = numpy.array([[0,-1*t_vec[2],t_vec[1]],[t_vec[2],0,-1*t_vec[0]],[-1*t_vec[1], t_vec[0],0]])
            
            moment_trans = numpy.dot(r_mat, skew_mat)
            
            row1 = numpy.append(r_mat, -1*moment_trans, axis = 1)
            row2 = numpy.append([[0,0,0],[0,0,0],[0,0,0]], r_mat, axis = 1)
            
            Vj = numpy.append(row1, row2, axis = 0)
            
            
            if(axis[0] == 1):
              J[:,i] = Vj[:,3]
            
            if(axis[1] == 1):
              J[:,i] = Vj[:,4]
            
            if(axis[2] == 1):
              J[:,i] = Vj[:,5]
              
            if(axis[0] == -1):
              J[:,i] = -Vj[:,3]
            
            if(axis[1] == -1):
              J[:,i] = -Vj[:,4]
            
            if(axis[2] == -1):
              J[:,i] = -Vj[:,5]
            
            
            
            

        
        #print(len(J[0]))
          #--------------------------------------------------------------------------
        return J

    '''This is the callback which will be executed when the inverse kinematics
       recieve a new command. The command will contain information about desired
       end effector pose relative to the root of your robot. At the end of this
       callback, you should publish to the /joint_command topic. This should not
       search for a solution indefinitely - there should be a time limit. When
       searching for two matrices which are the same, we expect numerical
       precision of 10e-3.'''
    def get_ik_command(self, command):
        self.mutex.acquire()
        #--------------------------------------------------------------------------
     
        trans = tf.transformations.translation_matrix(numpy.array([command.translation.x, command.translation.y, command.translation.z]))
        rot = tf.transformations.quaternion_matrix(numpy.array([command.rotation.x, command.rotation.y, command.rotation.z, command.rotation.w]))
        
        
        des_pos_T = numpy.dot(trans,rot)
        num_tries = 3
        check = False
        
        for i in range(num_tries):
          
          
          q_now = numpy.random.rand(self.num_joints,1)
          q_now = 2*numpy.pi*q_now
          
          time_start = rospy.get_rostime()
          loop_duration = rospy.Duration.from_sec(10)
          
          while(rospy.get_rostime() < time_start+loop_duration):
            
            joint_transforms, b_T_cee = self.forward_kinematics(q_now)        
            b_T_cee_inv = tf.transformations.inverse_matrix(b_T_cee)  
            
            cee_T_despos = numpy.dot(b_T_cee_inv, des_pos_T)
            
            angle,axis = self.rotation_from_matrix(cee_T_despos[:3,:3])
            
            Rotvec = numpy.dot(angle,axis)
            
            deltax = numpy.zeros((6,1))
            
            deltax[:3,0] = cee_T_despos[:3,3]
            deltax[3:6,0] = Rotvec [0:3]
            
            print("This is deltax")
            print(deltax)
            J = self.get_jacobian(b_T_cee, joint_transforms)
            
            J_plus = numpy.linalg.pinv(J)
            
            delq = numpy.dot(J_plus, deltax)
            
            q_now = q_now + delq
            
            max_val = max(abs(deltax[:6,0]))
            if max_val<0.001:
                 check = True
                 break

          if check == True:
               
               self.joint_command_msg.name = self.joint_names
               self.joint_command_msg.position = q_now
               self.joint_command_pub.publish(self.joint_command_msg)
               break  
         
        '''   
        max_rad2 = max(abs(q_rand[0:6,0]))
        print("Final q_rand")
        print(q_rand)
        if (max_rad2 > 1):
          q_rand = q_rand/max_rad2
          
        print(q_rand)  
        
        '''
        
        
        #--------------------------------------------------------------------------
        self.mutex.release()

    '''This function will return the angle-axis representation of the rotation
       contained in the input matrix. Use like this: 
       angle, axis = rotation_from_matrix(R)'''
    def rotation_from_matrix(self, matrix):
        R = numpy.array(matrix, dtype=numpy.float64, copy=False)
        R33 = R[:3, :3]
        # axis: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, W = numpy.linalg.eig(R33.T)
        i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        axis = numpy.real(W[:, i[-1]]).squeeze()
        # point: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, Q = numpy.linalg.eig(R)
        i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        # rotation angle depending on axis
        cosa = (numpy.trace(R33) - 1.0) / 2.0
        if abs(axis[2]) > 1e-8:
            sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
        elif abs(axis[1]) > 1e-8:
            sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
        else:
            sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
        angle = math.atan2(sina, cosa)
        return angle, axis

    '''This is the function which will perform forward kinematics for your 
       cartesian control and inverse kinematics functions. It takes as input
       joint values for the robot and will return an array of 4x4 transforms
       from the base to each joint of the robot, as well as the transform from
       the base to the end effector.
       Usage: joint_transforms, b_T_ee = self.forward_kinematics(joint_values)'''
    def forward_kinematics(self, joint_values):
        joint_transforms = []

        link = self.robot.get_root()
        T = tf.transformations.identity_matrix()

        while True:
            if link not in self.robot.child_map:
                break

            (joint_name, next_link) = self.robot.child_map[link][0]
            joint = self.robot.joint_map[joint_name]

            T_l = numpy.dot(tf.transformations.translation_matrix(joint.origin.xyz), tf.transformations.euler_matrix(joint.origin.rpy[0], joint.origin.rpy[1], joint.origin.rpy[2]))
            T = numpy.dot(T, T_l)

            if joint.type != "fixed":
                joint_transforms.append(T)
                q_index = self.joint_names.index(joint_name)
                T_j = tf.transformations.rotation_matrix(joint_values[q_index], numpy.asarray(joint.axis))
                T = numpy.dot(T, T_j)

            link = next_link
        return joint_transforms, T #where T = b_T_ee

    '''This is the callback which will recieve and store the current robot
       joint states.'''
    def get_joint_state(self, msg):
        self.mutex.acquire()
        self.q_current = []
        for name in self.joint_names:
            self.q_current.append(msg.position[msg.name.index(name)])
        self.mutex.release()


if __name__ == '__main__':
    rospy.init_node('cartesian_control_and_IK', anonymous=True)
    CCIK()
    rospy.spin()
