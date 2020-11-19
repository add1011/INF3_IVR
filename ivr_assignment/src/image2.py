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
        # initialize a publisher to send images from camera2 to a topic named image_topic2
        self.image_pub2 = rospy.Publisher("image_topic2", Image, queue_size=1)
        # initialize a publisher to send joints' angular position to a topic called joints_pos2
        self.joints_pos2_pub = rospy.Publisher("joints_pos2", Float64MultiArray, queue_size=10)
        # initialize a publisher to send position of target to a topic called target_pos2
        self.target_pos2_pub = rospy.Publisher("target_pos2", Float64MultiArray, queue_size=10)
        # initialize a subscriber to receive messages from a topic named /robot/camera2/image_raw and use callback function to receive data
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        # third column is to store if this node is currently guessing the position of the joint
        self.joint_centres = np.zeros((4, 3), dtype='float64')

        # save the momentum of each joint to help estimate position when they cannot be seen
        self.joint_momentums = np.zeros((4, 2), dtype='float64')

        self.target_centre = np.zeros((1, 3), dtype='float64')

    def detectColour(self, hueFloor, hueCeiling, jointIndex):
        colourMask = cv2.inRange(self.img2HSV, (hueFloor, 80, 80), (hueCeiling, 255, 255))
        colourImg = cv2.bitwise_and(self.cv_image2, self.cv_image2, mask=colourMask)
        img_grey = cv2.cvtColor(colourImg, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_grey, 1, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if contours:
            (cX, cZ), radius = cv2.minEnclosingCircle(contours[0])
            cv2.circle(self.cv_image2, (int(cX), int(cZ)), int(radius), (255, 255, 255), 1)
            self.joint_momentums[jointIndex] = np.subtract([cX, cZ], self.joint_centres[jointIndex, :2])
            self.joint_centres[jointIndex] = [cX, cZ, 0]
        else:
            # cX, cZ = np.add(self.joint_centres[jointIndex, :2], self.joint_momentums[jointIndex])
            cX, cZ = self.joint_centres[jointIndex, :2]
            self.joint_centres[jointIndex] = [cX, cZ, 1]

        cv2.circle(self.cv_image2, (int(cX), int(cZ)), 2, (255, 255, 255), -1)
        return

    def detect_shape(self, c):
        shape = "unidentified"
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
        if len(approx) == 4:
            shape = "square"
        else:
            shape = "circle"
        return shape

    def detect_target(self, image):
        orangeMask = cv2.inRange(image, (5, 50, 50), (15, 255, 255))
        orangeImg = cv2.bitwise_and(image, image, mask=orangeMask)
        orangeImg_grey = cv2.cvtColor(orangeImg, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(orangeImg_grey, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = imutils.grab_contours(contours)
        cX, cZ = 0, 0
        for i in contours:
            shape = self.detect_shape(i)
            if shape == "circle":
                (cX, cZ), radius = cv2.minEnclosingCircle(contours[0])
                cv2.circle(self.cv_image2, (int(cX), int(cZ)), int(radius), (255, 255, 255), 1)
                self.target_centre[0] = [cX, cZ, 0]
                """
                M = cv2.moments(i)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cZ = int(M["m01"] / M["m00"])
                    self.target_centre[0] = [cX, cZ, 0]
                else:
                    cX, cZ = self.target_centre[0, :2]
                    self.target_centre[0] = [cX, cZ, 1]
                """
        cv2.circle(self.cv_image2, (int(cX), int(cZ)), 1, (255, 255, 255), -1)
        return np.array([cX, cZ])

    # Recieve data, process it, and publish
    def callback2(self, data):
        # Recieve the image
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.img2HSV = cv2.cvtColor(self.cv_image2, cv2.COLOR_BGR2HSV)

        self.detectColour(20, 40, 0)
        self.detectColour(110, 130, 1)
        self.detectColour(50, 70, 2)
        self.detectColour(0, 10, 3)

        self.js = Float64MultiArray()
        self.js.data = self.joint_centres.flatten()

        self.detect_target(self.img2HSV)
        self.target = Float64MultiArray()
        self.target.data = self.target_centre.flatten()

        # Uncomment if you want to save the image
        # cv2.imwrite('image_copy.png', cv_image)
        im2 = cv2.imshow('window2', self.cv_image2)
        cv2.waitKey(1)

        # Publish the results
        try:
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
            self.joints_pos2_pub.publish(self.js)
            self.target_pos2_pub.publish(self.target)
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
