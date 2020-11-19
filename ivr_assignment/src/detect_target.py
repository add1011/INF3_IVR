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


class detect_target:
    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('detect_target', anonymous=True)

        self.target_pub = rospy.Publisher("target_pos", Float64MultiArray, queue_size=10)

        # initialize a subscriber to receive messages from a topic named target_pos1
        self.target_pos1_sub = message_filters.Subscriber("target_pos1", Float64MultiArray)
        # initialize a subscriber to receive messages from a topic named target_pos2
        self.target_pos2_sub = message_filters.Subscriber("target_pos2", Float64MultiArray)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.target_pos1_sub, self.target_pos2_sub],
            queue_size=10, slop=0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

    def pixel2meter(self, target_pos):
        # use last two joints as the distance between them will always be the same
        # metreInPixels = (np.linalg.norm(joints_pos[0] - joints_pos[1])) / 2.5
        return target_pos / 25.934213568650648

    def target3Dcord(self, target_pos1, target_pos2):
        target_pos = np.zeros((1, 3))
        x = target_pos2[0, 0]
        y = target_pos1[0, 0]
        if target_pos1[0, 2] == 0 and target_pos2[0, 2] != 0:
            z = target_pos1[0, 1]
        elif target_pos1[0, 2] != 0 and target_pos2[0, 2] == 0:
            z = target_pos2[0, 1]
        else:
            z = ((target_pos1[0, 1]+target_pos2[0, 1])/2)
        target_pos[0] = [x, y, z]
        return target_pos

    def callback(self, target_data1, target_data2):
        try:
            target_pos1 = np.asarray(target_data1.data, dtype='float64').reshape(1, 3)
            target_pos2 = np.asarray(target_data2.data, dtype='float64').reshape(1, 3)
        except CvBridgeError as e:
            print(e)

        target_pos = self.target3Dcord(target_pos1, target_pos2)
        target_pos = self.pixel2meter(target_pos)

        self.target = Float64MultiArray()
        self.target.data = target_pos[0]

        try:
            self.target_pub.publish(self.target)
        except CvBridgeError as e:
            print(e)
        return

# call the class
def main(args):
    im = detect_target()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
