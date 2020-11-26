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
        rospy.init_node('control', anonymous=True)

        # initialize a publisher to send joints' angular position to the robot
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

        # initialize a subscriber to receive messages from a topic named target_pos1
        self.target_pos_sub = message_filters.Subscriber("target_pos", Float64MultiArray)
        # initialize a subscriber to receive messages from a topic named box_pos1
        self.box_pos_sub = message_filters.Subscriber("box_pos", Float64MultiArray)
        # initialize a subscriber to receive messages from a topic named joints_pos1_sub
        self.joints_pos_sub = message_filters.Subscriber("visionee_pos", Float64MultiArray)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.joints_pos_sub, self.target_pos_sub, self.box_pos_sub],
            queue_size=10, slop=0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        self.currentJoint1Angle = 0
        self.currentJoint2Angle = 0
        self.currentJoint3Angle = 0
        self.currentJoint4Angle = 0

        self.previous_J_inv = np.zeros((4, 3))

        # record the beginning time
        self.time_trajectory = rospy.get_time()
        # initialize errors
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
        self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')
        # initialize error and derivative of error for trajectory tracking
        self.error = np.array([0.0, 0.0, 0.0], dtype='float64')
        self.error_d = np.array([0.0, 0.0, 0.0], dtype='float64')
        self.last_w1 = 0
        self.q_delta = np.array([0.00001, 0.00001, 0.00001, 0.00001])

    def calcJacobian(self, joint1Angle, joint2Angle, joint3Angle, joint4Angle):

        x1 = (3 * np.sin(joint3Angle) * np.cos(joint1Angle + np.pi / 2) - 3 * np.sin(joint1Angle + np.pi / 2) * np.cos(
            joint3Angle) * np.cos(joint2Angle + np.pi / 2)) * np.cos(joint4Angle) + \
             7 * np.sin(joint3Angle) * np.cos(joint1Angle + np.pi / 2) / 2 + \
             3 * np.sin(joint4Angle) * np.sin(joint1Angle + np.pi / 2) * np.sin(joint2Angle + np.pi / 2) - \
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

        z2 = -3 * np.sin(joint4Angle) * np.sin(joint2Angle + np.pi / 2) + \
             3 * np.cos(joint3Angle) * np.cos(joint4Angle) * np.cos(joint2Angle + np.pi / 2) + \
             7 * np.cos(joint3Angle) * np.cos(joint2Angle + np.pi / 2) / 2

        z3 = -3 * np.sin(joint3Angle) * np.sin(joint2Angle + np.pi / 2) * np.cos(joint4Angle) - \
             7 * np.sin(joint3Angle) * np.sin(joint2Angle + np.pi / 2) / 2

        z4 = -3 * np.sin(joint4Angle) * np.sin(joint2Angle + np.pi / 2) * np.cos(joint3Angle) + \
             3 * np.cos(joint4Angle) * np.cos(joint2Angle + np.pi / 2)

        jacobian = np.array([[x1, x2, x3, x4],
                             [y1, y2, y3, y4],
                             [z1, z2, z3, z4]])

        return jacobian

    def controlClosed(self, ee, ee_d, joint1Angle, joint2Angle, joint3Angle, joint4Angle):
        # P gain
        K_p = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        # D gain
        K_d = np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]])
        # estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        if dt == 0:
            dt = [0.001]
        self.time_previous_step = cur_time
        # estimate derivative of error
        self.error_d = ((ee_d - ee) - self.error) / dt
        # estimate error
        self.error = ee_d - ee
        J = self.calcJacobian(joint1Angle, joint2Angle, joint3Angle, joint4Angle)
        # calculating the psuedo inverse of the Jacobian
        k = 10
        # use a damped Jacobian instead of pseudo-inverse to help prevent singularity
        J_damp = np.dot(J.T, np.linalg.inv(np.dot(J, J.T) + (k ** 2) * np.eye(3)))
        # control input (angular velocity of joints)
        dq_d = np.dot(J_damp, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p, self.error.transpose())))
        # control input (angular position of joints)
        q_d = np.array([joint1Angle, joint2Angle, joint3Angle, joint4Angle]) + (dt * dq_d)
        return q_d

    def nullControlClosed(self, ee, ee_d, joint1Angle, joint2Angle, joint3Angle, joint4Angle):
        # P gain
        K_p = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
        # D gain
        K_d = np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]])
        # estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        if dt == 0:
            dt = [0.001]
        self.time_previous_step = cur_time
        # estimate derivative of error
        self.error_d = ((ee_d - ee) - self.error) / dt
        # estimate error
        self.error = ee_d - ee
        J = self.calcJacobian(joint1Angle, joint2Angle, joint3Angle, joint4Angle)
        k = 10
        # calculating the psuedo inverse of the Jacobian
        # use a damped Jacobian instead of pseudo-inverse to help prevent singularity
        J_damp = np.dot(J.T, np.linalg.inv(np.dot(J, J.T) + np.dot(k**2, np.eye(3))))
        # control input (angular velocity of joints)
        dq_d = np.dot(J_damp, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p, self.error.transpose())))

        # TASK 4.2 #
        #self.q_delta[self.q_delta == 0] += self.q_delta[self.q_delta == 0] + 0.00001
        # find dq0 (box)
        w1 = np.linalg.norm(ee - self.box_pos)
        dq0 = ((w1 - self.last_w1)/self.q_delta)
        k = 0.01
        dq0 = k * dq0.T

        # Alternate solution for dq0:
        # dq0 = [(w-self.box_distance)/self.q_delta[0], (w-self.box_distance)/self.q_delta[0], (w-self.box_distance)/self.q_delta[0], (w-self.box_distance)/self.q_delta[0]]

        # add the secondary tasks to the calculated velocity
        dq_d = dq_d + np.dot((np.eye(4) - np.dot(J_damp, J)), dq0)
        self.last_w1 = w1

        self.q_delta = dt * dq_d
        # control input (angular position of joints)
        q_d = np.array([joint1Angle, joint2Angle, joint3Angle, joint4Angle]) + (dt * dq_d)

        return q_d

    def callback(self, joints_pos, target_pos, box_pos):
        # recieve the position data from each image
        try:
            self.joints_pos = np.asarray(joints_pos.data, dtype='float64').reshape(3)
            self.target_pos = np.asarray(target_pos.data, dtype='float64').reshape(3)
            self.box_pos = np.asarray(box_pos.data, dtype='float64').reshape(3)
        except CvBridgeError as e:
            print(e)

        ######################## CONTROL ########################
        # find the set of joint angles to make the end effector move towards the target
        q_d = self.controlClosed(self.joints_pos,
                                 self.target_pos,
                                 self.currentJoint1Angle, self.currentJoint2Angle,
                                 self.currentJoint3Angle, self.currentJoint4Angle)

        # set up the data to be published
        self.joint1 = Float64()
        self.joint1.data = q_d[0]
        self.joint2 = Float64()
        self.joint2.data = q_d[1]
        self.joint3 = Float64()
        self.joint3.data = q_d[2]
        self.joint4 = Float64()
        self.joint4.data = q_d[3]

        # update the saved joint angles to the ones found in this frame
        self.currentJoint1Angle = q_d[0]
        self.currentJoint2Angle = q_d[1]
        self.currentJoint3Angle = q_d[2]
        self.currentJoint4Angle = q_d[3]

        try:
            # PUBLISH DATA #
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
