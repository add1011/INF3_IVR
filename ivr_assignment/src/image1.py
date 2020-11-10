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
        yellowMask = cv2.inRange(self.img1HSV, (20, 100, 100), (40, 255, 255))
        yellowImg = cv2.bitwise_and(self.cv_image1, self.cv_image1, mask=yellowMask)
        yellowImg_grey = cv2.cvtColor(yellowImg, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(yellowImg_grey, 1, 255, 0)
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cY = int(M["m10"] / M["m00"])
            cZ = int(M["m01"] / M["m00"])
            self.yellow_momentum = np.subtract([cY, cZ], self.yellow_pos)
            self.yellow_pos = [cY, cZ]
            self.joint_centres[0] = [cY, cZ, 0]
        else:
            cY, cZ = np.add(self.yellow_pos, self.yellow_momentum)
            self.yellow_pos = [cY, cZ]
            self.joint_centres[0] = [cY, cZ, 1]

        cv2.circle(self.cv_image1, (cY, cZ), 2, (255, 255, 255), -1)
        """
      cv2.putText(self.cv_image1, "Yellow Center", (cY - 25, cZ - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      """

        return np.array([cY, cZ])

    def detect_blue(self, image):
        blueMask = cv2.inRange(self.img1HSV, (110, 100, 100), (130, 255, 255))
        blueImg = cv2.bitwise_and(self.cv_image1, self.cv_image1, mask=blueMask)
        blueImg_grey = cv2.cvtColor(blueImg, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(blueImg_grey, 1, 255, 0)
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cY = int(M["m10"] / M["m00"])
            cZ = int(M["m01"] / M["m00"])
            self.blue_momentum = np.subtract([cY, cZ], self.blue_pos)
            self.blue_pos = [cY, cZ]
            self.joint_centres[1] = [cY, cZ, 0]
        else:
            cY, cZ = np.add(self.blue_pos, self.blue_momentum)
            self.blue_pos = [cY, cZ]
            self.joint_centres[1] = [cY, cZ, 1]

        cv2.circle(self.cv_image1, (cY, cZ), 2, (255, 255, 255), -1)
        """
      cv2.putText(self.cv_image1, "Blue Center", (cY - 25, cZ - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      """

        return np.array([cY, cZ])

    def detect_green(self, image):
        greenMask = cv2.inRange(self.img1HSV, (50, 100, 100), (70, 255, 255))
        greenImg = cv2.bitwise_and(self.cv_image1, self.cv_image1, mask=greenMask)
        greenImg_grey = cv2.cvtColor(greenImg, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(greenImg_grey, 1, 255, 0)
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cY = int(M["m10"] / M["m00"])
            cZ = int(M["m01"] / M["m00"])
            self.green_momentum = np.subtract([cY, cZ], self.green_pos)
            self.green_pos = [cY, cZ]
            self.joint_centres[2] = [cY, cZ, 0]
        else:
            cY, cZ = np.add(self.green_pos, self.green_momentum)
            self.green_pos = [cY, cZ]
            self.joint_centres[2] = [cY, cZ, 1]

        cv2.circle(self.cv_image1, (cY, cZ), 2, (255, 255, 255), -1)
        """
      cv2.putText(self.cv_image1, "Green Center", (cY - 25, cZ - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      """
        return np.array([cY, cZ])

    def detect_red(self, image):
        redMask = cv2.inRange(self.img1HSV, (0, 100, 100), (10, 255, 255))
        redImg = cv2.bitwise_and(self.cv_image1, self.cv_image1, mask=redMask)
        redImg_grey = cv2.cvtColor(redImg, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(redImg_grey, 1, 255, 0)
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cY = int(M["m10"] / M["m00"])
            cZ = int(M["m01"] / M["m00"])
            self.red_momentum = np.subtract([cY, cZ], self.red_pos)
            self.red_pos = [cY, cZ]
            self.joint_centres[3] = [cY, cZ, 0]
        else:
            cY, cZ = np.add(self.red_pos, self.red_momentum)
            self.red_pos = [cY, cZ]
            self.joint_centres[3] = [cY, cZ, 1]

        cv2.circle(self.cv_image1, (cY, cZ), 2, (255, 255, 255), -1)
        """
      cv2.putText(self.cv_image1, "Red Center", (cY - 25, cZ - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      """
        return np.array([cY, cZ])

    def pixel2meter(self, image):
        # Obtain the centre of each coloured blob
        circle2Pos = self.detect_blue(image)
        circle3Pos = self.detect_green(image)
        # find the distance between two circles
        dist = np.sum((circle2Pos - circle3Pos) ** 2)
        return 3.5 / np.sqrt(dist)

    # Recieve data from camera 1, process it, and publish
    def callback1(self, data):
        # Recieve the image
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.img1HSV = cv2.cvtColor(self.cv_image1, cv2.COLOR_BGR2HSV)

        self.detect_red(self.cv_image1)
        self.detect_green(self.cv_image1)
        self.detect_blue(self.cv_image1)
        self.detect_yellow(self.cv_image1)

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
        self.joint4 = Float64()
        self.joint4.data = np.pi / 2 * np.sin((np.pi / 20) * t)

        # Publish the results
        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
            self.joints_pos1_pub.publish(self.js)
            """
            self.robot_joint2_pub.publish(self.joint2)
            self.robot_joint3_pub.publish(self.joint3)
            self.robot_joint4_pub.publish(self.joint4)
            """
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
