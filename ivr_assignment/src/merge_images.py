#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class image_merger:
    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_merger', anonymous=True)

        # initialize a publisher to send joints' position to a topic called joints_pos1
        self.joint2_pub = rospy.Publisher("joint2", Float64, queue_size=10)
        self.joint3_pub = rospy.Publisher("joint3", Float64, queue_size=10)
        self.joint4_pub = rospy.Publisher("joint4", Float64, queue_size=10)

        self.targetx_pub = rospy.Publisher("target_xpos", Float64, queue_size=10)
        self.targety_pub = rospy.Publisher("target_ypos", Float64, queue_size=10)
        self.targetz_pub = rospy.Publisher("target_zpos", Float64, queue_size=10)

        self.visionx_pub = rospy.Publisher("visionee_xpos", Float64, queue_size=10)
        self.visiony_pub = rospy.Publisher("visionee_ypos", Float64, queue_size=10)
        self.visionz_pub = rospy.Publisher("visionee_zpos", Float64, queue_size=10)

        self.FKx_pub = rospy.Publisher("FKee_xpos", Float64, queue_size=10)
        self.FKy_pub = rospy.Publisher("FKee_ypos", Float64, queue_size=10)
        self.FKz_pub = rospy.Publisher("FKee_zpos", Float64, queue_size=10)

        self.targetx_pub = rospy.Publisher("target_xpos", Float64, queue_size=10)
        self.targety_pub = rospy.Publisher("target_ypos", Float64, queue_size=10)
        self.targetz_pub = rospy.Publisher("target_zpos", Float64, queue_size=10)

        # initialize a publisher to send joints' angular position to the robot
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

        self.targetx_actual_sub = message_filters.Subscriber("/target/x_position_controller/command", Float64)
        self.targety_actual_sub = message_filters.Subscriber("/target/y_position_controller/command", Float64)
        self.targetz_actual_sub = message_filters.Subscriber("/target/z_position_controller/command", Float64)

        """"
        self.joint1_actual_sub = message_filters.Subscriber("/robot/joint1_position_controller/command", Float64)
        self.joint2_actual_sub = message_filters.Subscriber("/robot/joint2_position_controller/command", Float64)
        self.joint3_actual_sub = message_filters.Subscriber("/robot/joint3_position_controller/command", Float64)
        self.joint4_actual_sub = message_filters.Subscriber("/robot/joint4_position_controller/command", Float64)
        """
        # initialize a subscriber to receive messages from a topic named target_pos1
        self.target_pos1_sub = message_filters.Subscriber("target_pos1", Float64MultiArray)
        # initialize a subscriber to receive messages from a topic named target_pos2
        self.target_pos2_sub = message_filters.Subscriber("target_pos2", Float64MultiArray)

        # initialize a subscriber to receive messages from a topic named joints_pos1_sub
        self.joints_pos1_sub = message_filters.Subscriber("joints_pos1", Float64MultiArray)
        # initialize a subscriber to receive messages from a topic named joints_pos2_sub
        self.joints_pos2_sub = message_filters.Subscriber("joints_pos2", Float64MultiArray)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.joints_pos1_sub, self.joints_pos2_sub, self.target_pos1_sub, self.target_pos2_sub,
             #self.joint1_actual_sub, self.joint2_actual_sub, self.joint3_actual_sub, self.joint4_actual_sub,
             self.targetx_actual_sub, self.targety_actual_sub, self.targetz_actual_sub],
            queue_size=10, slop=0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        self.lastJoint1Angle = 0
        self.lastJoint2Angle = 0
        self.lastJoint3Angle = 0
        self.lastJoint4Angle = 0

        # record the begining time
        self.time_trajectory = rospy.get_time()
        # initialize errors
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
        self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')
        # initialize error and derivative of error for trajectory tracking
        self.error = np.array([0.0, 0.0, 0.0], dtype='float64')
        self.error_d = np.array([0.0, 0.0, 0.0], dtype='float64')

    def calc3dCoords(self, source1, source2):
        coords3d = np.zeros((source1.shape[0], 3))
        for j in range(source1.shape[0]):
            x = source2[j, 0]
            y = source1[j, 0]
            # if camera1 can see the joint but camera2 cannot
            if source1[j, 2] == 0 and source2[j, 2] != 0:
                z = source1[j, 1]
            # else if camera1 cannot see the joint but camera2 can
            elif source1[j, 2] != 0 and source2[j, 2] == 0:
                z = source2[j, 1]
            # else if both cameras have the same amount of information
            else:
                # average the z from both cameras
                z = ((source1[j, 1] + source2[j, 1]) / 2)
            coords3d[j] = [x, y, z]
        return coords3d

    def makeRelative(self, coordinates):
        relativePositions = np.zeros(coordinates.shape)
        relativePositions[:, 0] = coordinates[:, 0] - self.joints_pos_orig[0, 0]
        relativePositions[:, 1] = coordinates[:, 1] - self.joints_pos_orig[0, 1]
        relativePositions[:, 2] = self.joints_pos_orig[0, 2] - coordinates[:, 2]
        return relativePositions

    def pixel2meter(self, joints_pos):
        # use last two joints as the distance between them will always be the same
        # metreInPixels = (np.linalg.norm(joints_pos[2] - joints_pos[3])) / 3
        return joints_pos / 25.934213568650648

    def calcJoint2Angle(self):
        if self.joints_pos[1, 1] < -0.00336898:
            orientation = np.array([0, -0.00336898, 2.16924723])
        else:
            orientation = np.array([0, -0.00336898, 2.21869815])

        joint2Angle = np.arctan2(self.joints_pos[1, 2] - orientation[2],
                                 self.joints_pos[1, 1] - orientation[1])

        if joint2Angle < -np.pi / 2:
            joint2Angle = np.pi / 2
        elif joint2Angle < 0:
            joint2Angle = -np.pi / 2
        else:
            joint2Angle -= np.pi / 2

        difference = self.lastJoint2Angle - joint2Angle
        joint2Angle += difference / 2

        if joint2Angle > (np.pi / 2):
            joint2Angle = np.pi / 2
        elif joint2Angle < -(np.pi / 2):
            joint2Angle = -np.pi / 2

        self.lastJoint2Angle = joint2Angle
        return joint2Angle

    def calcJoint3Angle(self, joint2Angle):
        if self.joints_pos[1, 1] < -0.00336898:
            orientation = np.array([0, -0.00336898, 2.16924723])
        else:
            orientation = np.array([0, -0.00336898, 2.21869815])

        x = self.joints_pos[2, 0] - orientation[0]
        y = self.joints_pos[2, 1] - orientation[1]
        z = self.joints_pos[2, 2] - orientation[2]
        theta = -joint2Angle

        xrot = np.array([x,
                         y * np.cos(theta) - z * np.sin(theta),
                         y * np.sin(theta) + z * np.cos(theta)])

        joint3Angle = np.arctan2(xrot[2], xrot[0])

        if joint3Angle < -np.pi / 2:
            joint3Angle = np.pi / 2
        elif joint3Angle < 0:
            joint3Angle = -np.pi / 2
        else:
            joint3Angle -= np.pi / 2

        if joint3Angle > (np.pi / 2):
            joint3Angle = np.pi / 2
        elif joint3Angle < -(np.pi / 2):
            joint3Angle = -np.pi / 2

        return -joint3Angle

    def calcJoint4Angle(self, joint2Angle, joint3Angle):
        if self.joints_pos[1, 1] < -0.00336898:
            orientation = np.array([0, -0.00336898, 2.16924723])
        else:
            orientation = np.array([0, -0.00336898, 2.21869815])

        x1 = self.joints_pos[2, 0] - orientation[0]
        y1 = self.joints_pos[2, 1] - orientation[1]
        z1 = self.joints_pos[2, 2] - orientation[2]

        x2 = self.joints_pos[3, 0] - orientation[0]
        y2 = self.joints_pos[3, 1] - orientation[1]
        z2 = self.joints_pos[3, 2] - orientation[2]

        theta = -joint3Angle

        # print(self.joints_pos[2:4])
        # print("-")
        yrot1 = np.array([x1 * np.cos(theta) + z1 * np.sin(theta),
                          y1,
                          -x1 * np.sin(theta) + z1 * np.cos(theta)])

        # print(yrot1 + orientation)

        yrot2 = np.array([x2 * np.cos(theta) + z2 * np.sin(theta),
                          y2,
                          -x2 * np.sin(theta) + z2 * np.cos(theta)])

        # print(yrot2 + orientation)
        # print("-")

        theta = -joint2Angle

        yxrot1 = np.array([yrot1[0],
                           yrot1[1] * np.cos(theta) - yrot1[2] * np.sin(theta),
                           yrot1[1] * np.sin(theta) + yrot1[2] * np.cos(theta)])

        yxrot1 += orientation

        yxrot2 = np.array([yrot2[0],
                           yrot2[1] * np.cos(theta) - yrot2[2] * np.sin(theta),
                           yrot2[1] * np.sin(theta) + yrot2[2] * np.cos(theta)])

        yxrot2 += orientation
        # print(yrot2)
        # print("#####")

        joint4Angle = np.arctan2(yrot2[2] - yrot1[2], yrot2[1] - yrot1[1]) - joint2Angle
        # joint4Angle = np.arctan2(yxrot2[2] - yxrot1[2], yxrot2[1] - yxrot1[1])

        # joint4Angle = np.arctan2(self.joints_pos[3, 2] - self.joints_pos[2, 2],
        #                         self.joints_pos[3, 1] - self.joints_pos[2, 1])

        if joint4Angle < -np.pi / 2:
            joint4Angle = np.pi / 2
        elif joint4Angle < 0:
            joint4Angle = -np.pi / 2
        else:
            joint4Angle -= np.pi / 2

        if joint4Angle > (np.pi / 2):
            joint4Angle = np.pi / 2
        elif joint4Angle < -(np.pi / 2):
            joint4Angle = -np.pi / 2

        return joint4Angle

    def forwardKinematics(self, joint1Angle, joint2Angle, joint3Angle, joint4Angle):
        x = 3 * (np.sin(joint3Angle) * np.sin(joint1Angle + np.pi / 2) + np.cos(joint3Angle) * np.cos(
            joint1Angle + np.pi / 2) * np.cos(joint2Angle + np.pi / 2)) * np.cos(joint4Angle) + \
            7 * np.sin(joint3Angle) * np.sin(joint1Angle + np.pi / 2) / 2 - \
            3 * np.sin(joint4Angle) * np.sin(joint2Angle + np.pi / 2) * np.cos(joint1Angle + np.pi / 2) + \
            7 * np.cos(joint3Angle) * np.cos(joint1Angle + np.pi / 2) * np.cos(joint2Angle + np.pi / 2) / 2

        """
        xx = 3 * (sympy.sin(t3) * sympy.sin(sympy.Add(t1 + np.pi / 2)) + sympy.cos(t3) * sympy.cos(
            sympy.Add(t1 + np.pi / 2)) * sympy.cos(sympy.Add(t2 + np.pi / 2))) * sympy.cos(t4) + \
            7 * sympy.sin(t3) * sympy.sin(sympy.Add(t1 + np.pi / 2)) / 2 - \
            3 * sympy.sin(t4) * sympy.sin(sympy.Add(t2 + np.pi / 2)) * sympy.cos(sympy.Add(t1 + np.pi / 2)) + \
            7 * sympy.cos(t3) * sympy.cos(sympy.Add(t1 + np.pi / 2)) * sympy.cos(sympy.Add(t2 + np.pi / 2)) / 2
        """

        y = 3 * (-np.sin(joint3Angle) * np.cos(joint1Angle + np.pi / 2) + np.sin(joint1Angle + np.pi / 2) * np.cos(
            joint3Angle) * np.cos(joint2Angle + np.pi / 2)) * np.cos(joint4Angle) - \
            7 * np.sin(joint3Angle) * np.cos(joint1Angle + np.pi / 2) / 2 - \
            3 * np.sin(joint4Angle) * np.sin(joint1Angle + np.pi / 2) * np.sin(joint2Angle + np.pi / 2) + \
            7 * np.sin(joint1Angle + np.pi / 2) * np.cos(joint3Angle) * np.cos(joint2Angle + np.pi / 2) / 2

        """
        yy = 3 * (-sympy.sin(joint3Angle) * sympy.cos(sympy.Add(joint1Angle + np.pi / 2)) + sympy.sin(
            sympy.Add(joint1Angle + np.pi / 2)) * sympy.cos(
            joint3Angle) * sympy.cos(sympy.Add(joint2Angle + np.pi / 2))) * sympy.cos(joint4Angle) - \
             7 * sympy.sin(joint3Angle) * sympy.cos(sympy.Add(joint1Angle + np.pi / 2)) / 2 - \
             3 * sympy.sin(joint4Angle) * sympy.sin(sympy.Add(joint1Angle + np.pi / 2)) * sympy.sin(
            sympy.Add(joint2Angle + np.pi / 2)) + \
             7 * sympy.sin(sympy.Add(joint1Angle + np.pi / 2)) * sympy.cos(joint3Angle) * sympy.cos(
            sympy.Add(joint2Angle + np.pi / 2)) / 2
        """

        z = 3 * np.sin(joint4Angle) * np.cos(joint2Angle + np.pi / 2) + \
            3 * np.sin(joint2Angle + np.pi / 2) * np.cos(joint3Angle) * np.cos(joint4Angle) + \
            7 * np.sin(joint2Angle + np.pi / 2) * np.cos(joint3Angle) / 2 + \
            5 / 2

        """
        zz = 3 * sympy.sin(joint4Angle) * sympy.cos(sympy.Add(joint2Angle + np.pi / 2)) + \
             3 * sympy.sin(sympy.Add(joint2Angle + np.pi / 2)) * sympy.cos(joint3Angle) * sympy.cos(joint4Angle) + \
             7 * sympy.sin(sympy.Add(joint2Angle + np.pi / 2)) * sympy.cos(joint3Angle) / 2 + \
             5 / 2
        """

        return np.array([x, y, z])

    def calcJacobian(self, joint1Angle, joint2Angle, joint3Angle, joint4Angle):

        x1 = (3 * np.sin(joint3Angle) * np.cos(joint1Angle + np.pi / 2) - 3 * np.sin(joint1Angle + np.pi / 2) * np.cos(
            joint3Angle) * np.cos(joint2Angle + np.pi / 2)) * np.cos(joint4Angle) + \
             7 * np.sin(joint3Angle) * np.cos(joint1Angle + np.pi / 2) / 2 + 3 * np.sin(joint4Angle) * np.sin(
            joint1Angle + np.pi / 2) * np.sin(joint2Angle + np.pi / 2) - \
             7 * np.sin(joint1Angle + np.pi / 2) * np.cos(joint3Angle) * np.cos(joint2Angle + np.pi / 2) / 2

        x2 = -3 * np.sin(joint4Angle) * np.cos(joint1Angle + np.pi / 2) * np.cos(joint2Angle + np.pi / 2) - \
             3 * np.sin(joint2Angle + np.pi / 2) * np.cos(joint3Angle) * np.cos(joint4Angle) * np.cos(
            joint1Angle + np.pi / 2) - \
             7 * np.sin(joint2Angle + np.pi / 2) * np.cos(joint3Angle) * np.cos(joint1Angle + np.pi / 2) / 2

        x3 = (-3 * np.sin(joint3Angle) * np.cos(joint1Angle + np.pi / 2) * np.cos(joint2Angle + np.pi / 2) + 3 * np.sin(
            joint1Angle + np.pi / 2) * np.cos(joint3Angle)) * np.cos(joint4Angle) - \
             7 * np.sin(joint3Angle) * np.cos(joint1Angle + np.pi / 2) * np.cos(joint2Angle + np.pi / 2) / 2 + \
             7 * np.sin(joint1Angle + np.pi / 2) * np.cos(joint3Angle) / 2

        x4 = -(3 * np.sin(joint3Angle) * np.sin(joint1Angle + np.pi / 2) + 3 * np.cos(joint3Angle) * np.cos(
            joint1Angle + np.pi / 2) * np.cos(joint2Angle + np.pi / 2)) * np.sin(joint4Angle) - \
             3 * np.sin(joint2Angle + np.pi / 2) * np.cos(joint4Angle) * np.cos(joint1Angle + np.pi / 2)

        y1 = (3 * np.sin(joint3Angle) * np.sin(joint1Angle + np.pi / 2) + 3 * np.cos(joint3Angle) * np.cos(
            joint1Angle + np.pi / 2) * np.cos(joint2Angle + np.pi / 2)) * np.cos(joint4Angle) + \
             7 * np.sin(joint3Angle) * np.sin(joint1Angle + np.pi / 2) / 2 - \
             3 * np.sin(joint4Angle) * np.sin(joint2Angle + np.pi / 2) * np.cos(joint1Angle + np.pi / 2) + \
             7 * np.cos(joint3Angle) * np.cos(joint1Angle + np.pi / 2) * np.cos(joint2Angle + np.pi / 2) / 2

        y2 = -3 * np.sin(joint4Angle) * np.sin(joint1Angle + np.pi / 2) * np.cos(joint2Angle + np.pi / 2) - \
             3 * np.sin(joint1Angle + np.pi / 2) * np.sin(joint2Angle + np.pi / 2) * np.cos(joint3Angle) * np.cos(
            joint4Angle) - \
             7 * np.sin(joint1Angle + np.pi / 2) * np.sin(joint2Angle + np.pi / 2) * np.cos(joint3Angle) / 2

        y3 = (-3 * np.sin(joint3Angle) * np.sin(joint1Angle + np.pi / 2) * np.cos(joint2Angle + np.pi / 2) - 3 * np.cos(
            joint3Angle) * np.cos(joint1Angle + np.pi / 2)) * np.cos(joint4Angle) - \
             7 * np.sin(joint3Angle) * np.sin(joint1Angle + np.pi / 2) * np.cos(joint2Angle + np.pi / 2) / 2 - \
             7 * np.cos(joint3Angle) * np.cos(joint1Angle + np.pi / 2) / 2

        y4 = -(-3 * np.sin(joint3Angle) * np.cos(joint1Angle + np.pi / 2) + 3 * np.sin(
            joint1Angle + np.pi / 2) * np.cos(joint3Angle) * np.cos(joint2Angle + np.pi / 2)) * np.sin(joint4Angle) - \
             3 * np.sin(joint1Angle + np.pi / 2) * np.sin(joint2Angle + np.pi / 2) * np.cos(joint4Angle)

        z1 = 0

        z2 = -3*np.sin(joint4Angle)*np.sin(joint2Angle + np.pi / 2) + \
             3*np.cos(joint3Angle)*np.cos(joint4Angle)*np.cos(joint2Angle + np.pi / 2) + \
             7*np.cos(joint3Angle)*np.cos(joint2Angle + np.pi / 2)/2

        z3 = -3*np.sin(joint3Angle)*np.sin(joint2Angle + np.pi / 2)*np.cos(joint4Angle) - \
             7*np.sin(joint3Angle)*np.sin(joint2Angle + np.pi / 2)/2

        z4 = -3*np.sin(joint4Angle)*np.sin(joint2Angle + np.pi / 2)*np.cos(joint3Angle) + \
             3*np.cos(joint4Angle)*np.cos(joint2Angle + np.pi / 2)

        jacobian = np.array([[x1, x2, x3, x4],
                             [y1, y2, y3, y4],
                             [z1, z2, z3, z4]])

        return jacobian

    def control_closed(self, ee, ee_d, joint1Angle, joint2Angle, joint3Angle, joint4Angle):
        # P gain
        K_p = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        # D gain
        K_d = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        # estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        self.time_previous_step = cur_time
        # estimate derivative of error
        self.error_d = ((ee_d - ee) - self.error) / dt
        # estimate error
        self.error = ee_d - ee
        J_inv = np.linalg.pinv(self.calcJacobian(joint1Angle, joint2Angle, joint3Angle, joint4Angle))  # calculating the psudeo inverse of Jacobian
        dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p,
                                                                             self.error.transpose())))  # control input (angular velocity of joints)
        q_d = np.array([joint1Angle, joint2Angle, joint3Angle, joint4Angle]) + (dt * dq_d)  # control input (angular position of joints)
        return q_d

    def callback(self, camera1data, camera2data, target_data1, target_data2,
                 #joint1_actual, joint2_actual, joint3_actual, joint4_actual,
                 targetx_actual, targety_actual, targetz_actual):
        # recieve the position data from each image
        try:
            joints_pos1 = np.asarray(camera1data.data, dtype='float64').reshape(4, 3)
            joints_pos2 = np.asarray(camera2data.data, dtype='float64').reshape(4, 3)
            target_pos1 = np.asarray(target_data1.data, dtype='float64').reshape(1, 3)
            target_pos2 = np.asarray(target_data2.data, dtype='float64').reshape(1, 3)
            # joint1Actual = joint1_actual.data
            # joint2Actual = joint2_actual.data
            # joint3Actual = joint3_actual.data
            # joint4Actual = joint4_actual.data
            tx = targetx_actual.data
            ty = targety_actual.data
            tz = targetz_actual.data
        except CvBridgeError as e:
            print(e)

        np.set_printoptions(suppress=True)
        # merge the data into 3d position coordinates
        self.joints_pos_orig = self.calc3dCoords(joints_pos1, joints_pos2)
        # make the coordinates relative to the unmoving yellow joint
        self.joints_pos = self.makeRelative(self.joints_pos_orig)
        # make the coordinates in terms of meters
        self.joints_pos = self.pixel2meter(self.joints_pos)

        ######################## JOINT DETECTION ########################
        joint2Angle = self.calcJoint2Angle()
        self.joint2 = Float64()
        self.joint2.data = joint2Angle

        joint3Angle = self.calcJoint3Angle(joint2Angle)
        self.joint3 = Float64()
        self.joint3.data = joint3Angle

        joint4Angle = self.calcJoint4Angle(joint2Angle, joint3Angle)
        self.joint4 = Float64()
        self.joint4.data = joint4Angle

        ######################## TARGET DETECTION ########################
        target_pos = self.calc3dCoords(target_pos1, target_pos2)
        # target_pos = self.makeRelative(target_pos)
        target_pos = self.pixel2meter(target_pos)
        self.targetx = Float64()
        self.targetx.data = target_pos[0, 0] - 15.35
        self.targety = Float64()
        self.targety.data = target_pos[0, 1] - 15.35
        self.targetz = Float64()
        self.targetz.data = target_pos[0, 2]

        ######################## CONTROL ########################
        endEffector = self.forwardKinematics(self.lastJoint1Angle, self.lastJoint2Angle, self.lastJoint3Angle, self.lastJoint4Angle)
        # print(endEffector)

        self.fkEndEffectorx = endEffector[0]
        self.fkEndEffectory = endEffector[1]
        self.fkEndEffectorz = endEffector[2]

        # print(self.joints_pos)

        q_d = self.control_closed(endEffector, np.array([tx, ty, tz]), self.lastJoint1Angle, self.lastJoint2Angle, self.lastJoint3Angle, self.lastJoint4Angle)

        print(q_d)

        self.joint1 = Float64()
        self.joint1.data = q_d[0]
        self.joint2 = Float64()
        self.joint2.data = q_d[1]
        self.joint3 = Float64()
        self.joint3.data = q_d[2]
        self.joint4 = Float64()
        self.joint4.data = q_d[3]

        self.lastJoint1Angle = q_d[0]
        self.lastJoint2Angle = q_d[1]
        self.lastJoint3Angle = q_d[2]
        self.lastJoint4Angle = q_d[3]

        try:
            self.targetx_pub.publish(self.targetx)
            self.targety_pub.publish(self.targety)
            self.targetz_pub.publish(self.targetz)

            self.joint2_pub.publish(self.joint2)
            self.joint3_pub.publish(self.joint3)
            self.joint4_pub.publish(self.joint4)

            self.visionx_pub.publish(self.joints_pos[3, 0])
            self.visiony_pub.publish(self.joints_pos[3, 1])
            self.visionz_pub.publish(self.joints_pos[3, 2])

            self.FKx_pub.publish(self.fkEndEffectorx)
            self.FKy_pub.publish(self.fkEndEffectory)
            self.FKz_pub.publish(self.fkEndEffectorz)

            self.robot_joint1_pub.publish(self.joint1)
            self.robot_joint2_pub.publish(self.joint2)
            self.robot_joint3_pub.publish(self.joint3)
            self.robot_joint4_pub.publish(self.joint4)

        except CvBridgeError as e:
            print(e)
        return


# call the class
def main(args):
    im = image_merger()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
