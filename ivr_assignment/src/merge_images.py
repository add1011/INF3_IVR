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

        # self.image2_sub = rospy.Subscriber("joints_pos2", Float64MultiArray, self.listen_joints_pos2)
        self.joint2_pub = rospy.Publisher("joint2", Float64, queue_size=10)
        self.joint4_pub = rospy.Publisher("joint4", Float64, queue_size=10)

        # initialize a subscriber to receive messages from a topic named joints_pos1_sub and use listen_joints_pos1 function to receive data
        self.joints_pos1_sub = message_filters.Subscriber("joints_pos1", Float64MultiArray)
        # initialize a subscriber to receive messages from a topic named joints_pos2_sub and use listen_joints_pos2 function to receive data
        self.joints_pos2_sub = message_filters.Subscriber("joints_pos2", Float64MultiArray)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.joints_pos1_sub, self.joints_pos2_sub],
                                                              queue_size=10, slop=0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

    def calc3dCoords(self, joints_pos1, joints_pos2):
        joints_pos = np.zeros((4, 3))
        for j in range(4):
            x = joints_pos2[j, 0]
            y = joints_pos1[j, 0]
            # if camera1 can see the joint but camera2 cannot
            if joints_pos1[j, 2] == 0 and joints_pos2[j, 2] != 0:
                z = joints_pos1[j, 1]
            # else if camera1 cannot see the joint but camera2 can
            elif joints_pos1[j, 2] != 0 and joints_pos2[j, 2] == 0:
                z = joints_pos2[j, 1]
            # else if both cameras have the same amount of information
            else:
                # average the z from both cameras
                z = ((joints_pos1[j, 1] + joints_pos2[j, 1]) / 2)
            joints_pos[j] = [x, y, z]
        return joints_pos

    def makeRelative(self, joints_pos):
        relativePositions = np.zeros((4, 3))
        relativePositions[:, 0] = joints_pos[:, 0] - joints_pos[0, 0]
        relativePositions[:, 1] = joints_pos[:, 1] - joints_pos[0, 1]
        relativePositions[:, 2] = joints_pos[0, 2] - joints_pos[:, 2]
        return relativePositions

    def pixel2meter(self, joints_pos):
        # use last two joints as the distance between them will always be the same
        metreInPixels = (np.linalg.norm(joints_pos[2] - joints_pos[3])) / 3
        return joints_pos / metreInPixels

    def calcAngle(self, v1, v2):
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        s = np.linalg.norm(np.cross(v1, v2))
        c = np.dot(v1, v2)
        angle = np.arctan2(s, c)
        return angle

    def callback(self, camera1data, camera2data):
        # recieve the position data from each image
        try:
            joints_pos1 = np.asarray(camera1data.data, dtype='float64').reshape(4, 3)
            joints_pos2 = np.asarray(camera2data.data, dtype='float64').reshape(4, 3)
        except CvBridgeError as e:
            print(e)

        # print(joints_pos1, ",")
        # print(joints_pos2)
        # print()
        # merge the data into 3d position coordinates
        joints_pos = self.calc3dCoords(joints_pos1, joints_pos2)
        # print(joints_pos)
        # make the coordinates relative to the unmoving yellow joint
        joints_pos = self.makeRelative(joints_pos)
        # print(joints_pos)
        # make the coordinates in terms of meters
        joints_pos = self.pixel2meter(joints_pos)

        a = np.arctan2(joints_pos[1][2], joints_pos[1][1]) - np.pi / 2

        joint2a = a * 9.05330324620832639392

        if joint2a > 1:
            joint2a += (abs(joint2a) - 1)/2
        elif joint2a < -1:
            joint2a -= (abs(joint2a) - 1)/2


        print(joint2a)

        if joint2a > (np.pi / 2):
            joint2a = np.pi / 2
        elif joint2a < -(np.pi / 2):
            joint2a = -np.pi / 2

        self.joint2 = Float64()
        self.joint2.data = joint2a

        #self.joint4 = Float64()
        #self.joint4.data = self.calcAngle(joints_pos[2] - joints_pos[1], joints_pos[3] - joints_pos[2])

        try:
            self.joint2_pub.publish(self.joint2)
            #self.joint4_pub.publish(self.joint4)
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
