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
        # initialize a subscriber to receive messages from a topic named /robot/camera2/image_raw and use callback function to receive data
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        # save the position of the red and green joints to use them if they cannot be found
        self.red_pos = np.array([0, 0])
        self.red_momentum = np.array([0, 0])
        self.green_pos = np.array([0, 0])
        self.green_momentum = np.array([0, 0])
        self.blue_pos = np.array([0, 0])
        self.blue_momentum = np.array([0, 0])
        self.yellow_pos = np.array([0, 0])
        self.yellow_momentum = np.array([0, 0])

        # third column is to store if this node is currently guessing the position of the joint
        self.joint_centres = np.zeros((4, 3), dtype='float64')

    def detect_yellow(self, image):
        yellowMask = cv2.inRange(self.img2HSV, (20, 100, 100), (40, 255, 255))
        yellowImg = cv2.bitwise_and(self.cv_image2, self.cv_image2, mask=yellowMask)
        yellowImg_grey = cv2.cvtColor(yellowImg, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(yellowImg_grey, 1, 255, 0)
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cZ = int(M["m01"] / M["m00"])
            self.yellow_momentum = np.subtract([cX, cZ], self.yellow_pos)
            self.yellow_pos = [cX, cZ]
            self.joint_centres[0] = [cX, cZ, 0]
        else:
            cX, cZ = np.add(self.yellow_pos, self.yellow_momentum)
            self.yellow_pos = [cX, cZ]
            self.joint_centres[0] = [cX, cZ, 1]

        cv2.circle(self.cv_image2, (cX, cZ), 2, (255, 255, 255), -1)
        """
        cv2.putText(self.cv_image2, "Yellow Center", (cX - 25, cZ - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        """
        return np.array([cX, cZ])

    def detect_blue(self, image):
        blueMask = cv2.inRange(self.img2HSV, (110, 100, 100), (130, 255, 255))
        blueImg = cv2.bitwise_and(self.cv_image2, self.cv_image2, mask=blueMask)
        blueImg_grey = cv2.cvtColor(blueImg, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(blueImg_grey, 1, 255, 0)
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cZ = int(M["m01"] / M["m00"])
            self.blue_momentum = np.subtract([cX, cZ], self.blue_pos)
            self.blue_pos = [cX, cZ]
            self.joint_centres[1] = [cX, cZ, 0]
        else:
            cX, cZ = np.add(self.blue_pos, self.blue_momentum)
            self.blue_pos = [cX, cZ]
            self.joint_centres[1] = [cX, cZ, 1]

        cv2.circle(self.cv_image2, (cX, cZ), 2, (255, 255, 255), -1)
        """
         cv2.putText(self.cv_image2, "Blue Center", (cX - 25, cZ - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
         """

        return np.array([cX, cZ])

    def detect_green(self, image):
        greenMask = cv2.inRange(self.img2HSV, (50, 100, 100), (70, 255, 255))
        greenImg = cv2.bitwise_and(self.cv_image2, self.cv_image2, mask=greenMask)
        greenImg_grey = cv2.cvtColor(greenImg, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(greenImg_grey, 1, 255, 0)
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cZ = int(M["m01"] / M["m00"])
            self.green_momentum = np.subtract([cX, cZ], self.green_pos)
            self.green_pos = [cX, cZ]
            self.joint_centres[2] = [cX, cZ, 0]
        else:
            cX, cZ = np.add(self.green_pos, self.green_momentum)
            self.green_pos = [cX, cZ]
            self.joint_centres[2] = [cX, cZ, 1]

        cv2.circle(self.cv_image2, (cX, cZ), 2, (255, 255, 255), -1)
        """
         cv2.putText(self.cv_image2, "Green Center", (cX - 25, cZ - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        """

        return np.array([cX, cZ])

    def detect_red(self, image):
        redMask = cv2.inRange(self.img2HSV, (0, 100, 100), (10, 255, 255))
        redImg = cv2.bitwise_and(self.cv_image2, self.cv_image2, mask=redMask)
        redImg_grey = cv2.cvtColor(redImg, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(redImg_grey, 1, 255, 0)
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cZ = int(M["m01"] / M["m00"])
            self.red_momentum = np.subtract([cX, cZ], self.red_pos)
            self.red_pos = [cX, cZ]
            self.joint_centres[3] = [cX, cZ, 0]
        else:
            cX, cZ = np.add(self.red_pos, self.red_momentum)
            self.red_pos = [cX, cZ]
            self.joint_centres[3] = [cX, cZ, 1]

        cv2.circle(self.cv_image2, (cX, cZ), 2, (255, 255, 255), -1)
        """
        cv2.putText(self.cv_image2, "Red Center", (cX - 25, cZ - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        """
        return np.array([cX, cZ])

    # Recieve data, process it, and publish
    def callback2(self, data):
        # Recieve the image
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.img2HSV = cv2.cvtColor(self.cv_image2, cv2.COLOR_BGR2HSV)

        self.detect_red(self.cv_image2)
        self.detect_green(self.cv_image2)
        self.detect_blue(self.cv_image2)
        self.detect_yellow(self.cv_image2)

        self.js = Float64MultiArray()
        self.js.data = self.joint_centres.flatten()

        # Uncomment if you want to save the image
        # cv2.imwrite('image_copy.png', cv_image)
        im2 = cv2.imshow('window2', self.cv_image2)
        cv2.waitKey(1)

        # Publish the results
        try:
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
            self.joints_pos2_pub.publish(self.js)
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
