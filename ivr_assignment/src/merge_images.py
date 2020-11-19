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

        # initialize a subscriber to receive messages from a topic named target_pos1
        self.target_pos1_sub = message_filters.Subscriber("target_pos1", Float64MultiArray)
        # initialize a subscriber to receive messages from a topic named target_pos2
        self.target_pos2_sub = message_filters.Subscriber("target_pos2", Float64MultiArray)

        # initialize a subscriber to receive messages from a topic named joints_pos1_sub
        self.joints_pos1_sub = message_filters.Subscriber("joints_pos1", Float64MultiArray)
        # initialize a subscriber to receive messages from a topic named joints_pos2_sub
        self.joints_pos2_sub = message_filters.Subscriber("joints_pos2", Float64MultiArray)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.joints_pos1_sub, self.joints_pos2_sub, self.target_pos1_sub, self.target_pos2_sub],
            queue_size=10, slop=0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        self.lastJoint2Angle = 0

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
        # metreInPixels = (np.linalg.norm(joints_pos[0] - joints_pos[1])) / 2.5
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
        orientation = np.array([0, -0.00336898, 2.16924723])

        x1 = self.joints_pos[2, 0] - orientation[0]
        y1 = self.joints_pos[2, 1] - orientation[1]
        z1 = self.joints_pos[2, 2] - orientation[2]

        x2 = self.joints_pos[3, 0] - orientation[0]
        y2 = self.joints_pos[3, 1] - orientation[1]
        z2 = self.joints_pos[3, 2] - orientation[2]

        theta = -joint2Angle

        # print(self.joints_pos[2:4])
        # print("-")

        xrot1 = np.array([x1,
                          y1 * np.cos(theta) - z1 * np.sin(theta),
                          y1 * np.sin(theta) + z1 * np.cos(theta)])

        # print(xrot1 + orientation)

        xrot2 = np.array([x2,
                          y2 * np.cos(theta) - z2 * np.sin(theta),
                          y2 * np.sin(theta) + z2 * np.cos(theta)])

        # print(xrot2 + orientation)
        # print("-")

        theta = -joint3Angle

        xyrot1 = np.array([xrot1[0] * np.cos(theta) + xrot1[2] * np.sin(theta),
                           xrot1[1],
                           -xrot1[0] * np.sin(theta) + xrot1[2] * np.cos(theta)])

        # print(xyrot1 + orientation)

        xyrot2 = np.array([xrot2[0] * np.cos(theta) + xrot2[2] * np.sin(theta),
                           xrot2[1],
                           -xrot2[0] * np.sin(theta) + xrot2[2] * np.cos(theta)])

        # print(xyrot2 + orientation)
        # print("#####")

        joint4Angle = np.arctan2(xyrot2[2] - xyrot1[2], xyrot2[0] - xyrot1[0])

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

        return -joint4Angle

    def forwardKinematics(self, joint1Angle, joint2Angle, joint3Angle, joint4Angle):
        d = 2.5
        theta = 90
        a = 0
        alpha = 90
        joint1FK = np.array(
            [[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
             [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
             [0, np.sin(alpha), np.cos(alpha), d],
             [0, 0, 0, 1]])

        d = 0
        theta = 90
        a = 0
        alpha = 90
        joint2FK = np.array(
            [[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
             [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
             [0, np.sin(alpha), np.cos(alpha), d],
             [0, 0, 0, 1]])

        d = 3.5
        theta = 0
        a = 0
        alpha = 270
        joint3FK = np.array(
            [[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
             [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
             [0, np.sin(alpha), np.cos(alpha), d],
             [0, 0, 0, 1]])

        d = 3
        theta = 0
        a = 0
        alpha = 0
        joint4FK = np.array(
            [[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
             [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
             [0, np.sin(alpha), np.cos(alpha), d],
             [0, 0, 0, 1]])

        return

    def callback(self, camera1data, camera2data, target_data1, target_data2):
        # recieve the position data from each image
        try:
            joints_pos1 = np.asarray(camera1data.data, dtype='float64').reshape(4, 3)
            joints_pos2 = np.asarray(camera2data.data, dtype='float64').reshape(4, 3)
            target_pos1 = np.asarray(target_data1.data, dtype='float64').reshape(1, 3)
            target_pos2 = np.asarray(target_data2.data, dtype='float64').reshape(1, 3)
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
        target_pos = self.makeRelative(target_pos)
        target_pos = self.pixel2meter(target_pos)
        self.targetx = Float64()
        self.targetx.data = target_pos[0, 0]
        self.targety = Float64()
        self.targety.data = target_pos[0, 1]
        self.targetz = Float64()
        self.targetz.data = target_pos[0, 2]

        try:
            self.targetx_pub.publish(self.targetx)
            self.targety_pub.publish(self.targety)
            self.targetz_pub.publish(self.targetz)

            self.joint2_pub.publish(self.joint2)
            self.joint3_pub.publish(self.joint3)
            self.joint4_pub.publish(self.joint4)
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
