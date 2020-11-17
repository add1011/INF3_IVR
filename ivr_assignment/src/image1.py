#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send images from camera1 to a topic named image_topic1
        self.image_pub1 = rospy.Publisher("image_topic1", Image, queue_size=1)
        # initialize a publisher to send joints' position to a topic called joints_pos1
        self.joints_pos1_pub = rospy.Publisher("joints_pos1", Float64MultiArray, queue_size=10)
        # initialize a publisher to send joints' angular position to the robot
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
        # initialize a subscriber to receive messages from a topic named /robot/camera1/image_raw and use callback function to receive data
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        # third column is to store if this node is currently guessing the position of the joint
        self.joint_centres = np.zeros((4, 3), dtype='float64')

        # save the momentum of each joint to help estimate position when they cannot be seen
        self.joint_momentums = np.zeros((4, 2), dtype='float64')

    def detectColour(self, hueFloor, hueCeiling, jointIndex):
        colourMask = cv2.inRange(self.img1HSV, (hueFloor, 80, 80), (hueCeiling, 255, 255))
        colourImg = cv2.bitwise_and(self.cv_image1, self.cv_image1, mask=colourMask)
        img_grey = cv2.cvtColor(colourImg, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_grey, 1, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(self.cv_image1, contours, -1, (0, 255, 0), 3)
        cv2.waitKey(1)
        #M = cv2.moments(thresh)
        #if M["m00"] != 0:
        if contours:
            (cY, cZ), radius = cv2.minEnclosingCircle(contours[0])
            #cY = (M["m10"] / M["m00"])
            #cZ = (M["m01"] / M["m00"])
            self.joint_momentums[jointIndex] = np.subtract([cY, cZ], self.joint_centres[jointIndex, :2])
            self.joint_centres[jointIndex] = [cY, cZ, 0]
        else:
            # cY, cZ = np.add(self.joint_centres[jointIndex, :2], self.joint_momentums[jointIndex])
            cY, cZ = self.joint_centres[jointIndex, :2]
            self.joint_centres[jointIndex] = [cY, cZ, 1]

        cv2.circle(self.cv_image1, (int(cY), int(cZ)), 2, (255, 255, 255), -1)
        return

    # Recieve data from camera 1, process it, and publish
    def callback1(self, data):
        # Recieve the image
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


        self.img1HSV = cv2.cvtColor(self.cv_image1, cv2.COLOR_BGR2HSV)

        self.detectColour(20, 40, 0)
        self.detectColour(110, 130, 1)
        self.detectColour(50, 70, 2)
        self.detectColour(0, 10, 3)

        self.js = Float64MultiArray()
        self.js.data = self.joint_centres.flatten()

        # Uncomment if you want to save the image
        # cv2.imwrite('image_copy.png', cv_image)
        im1 = cv2.imshow('window1', self.cv_image1)
        cv2.waitKey(1)

        t = rospy.get_time()

        self.joint2 = Float64()
        self.joint2.data = np.pi / 2 * np.sin((np.pi / 15) * t)
        self.joint3 = Float64()
        self.joint3.data = np.pi / 2 * np.sin((np.pi / 18) * t)
        # Use pi/3 rather than pi/2 to prevent the arm knocking itself about
        self.joint4 = Float64()
        self.joint4.data = np.pi / 3 * np.sin((np.pi / 20) * t)

        #self.joint2.data = 0
        #self.joint3.data = 0
        #self.joint4.data = 0

        # Publish the results
        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
            self.joints_pos1_pub.publish(self.js)
            self.robot_joint2_pub.publish(self.joint2)
            self.robot_joint3_pub.publish(self.joint3)
            self.robot_joint4_pub.publish(self.joint4)
        except CvBridgeError as e:
            print(e)


# call the class
def main(args):
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
