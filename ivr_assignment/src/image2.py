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
    self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
    # initialize a subscriber to recieve messages from a topic named /robot/camera2/image_raw and use callback function to recieve data
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    
    # save the position of the red and green joints to use them if they cannot be found
    self.red_pos = np.array([0, 0])
    self.green_pos = np.array([0, 0])

  def detect_red(self,image):
      redMask = cv2.inRange(self.img2HSV, (0, 100, 100), (10, 255, 255))
      redImg = cv2.bitwise_and(self.cv_image2, self.cv_image2, mask=redMask)
      redImg_grey = cv2.cvtColor(redImg, cv2.COLOR_BGR2GRAY)
      ret, thresh = cv2.threshold(redImg_grey, 1, 255, 0)
      M = cv2.moments(thresh)
      if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        self.red_pos = [cX, cY]
      else:
        cX, cY = self.red_pos
      
      cv2.circle(self.cv_image2, (cX, cY), 2, (255, 255, 255), -1)
      """
      cv2.putText(self.cv_image2, "Red Center", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      """
      return np.array([cX, cY])

  def detect_green(self,image):
      greenMask = cv2.inRange(self.img2HSV, (50, 100, 100), (70, 255, 255))
      greenImg = cv2.bitwise_and(self.cv_image2, self.cv_image2, mask=greenMask)
      greenImg_grey = cv2.cvtColor(greenImg, cv2.COLOR_BGR2GRAY)
      ret, thresh = cv2.threshold(greenImg_grey, 1, 255, 0)
      M = cv2.moments(thresh)
      if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        self.green_pos = [cX, cY]
      else:
        cX, cY = self.green_pos
      
      cv2.circle(self.cv_image2, (cX, cY), 2, (255, 255, 255), -1)
      """
      cv2.putText(self.cv_image2, "Green Center", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      """
      
      return np.array([cX, cY])

  def detect_blue(self,image):
      blueMask = cv2.inRange(self.img2HSV, (110, 100, 100), (130, 255, 255))
      blueImg = cv2.bitwise_and(self.cv_image2, self.cv_image2, mask=blueMask)
      blueImg_grey = cv2.cvtColor(blueImg, cv2.COLOR_BGR2GRAY)
      ret, thresh = cv2.threshold(blueImg_grey, 1, 255, 0)
      M = cv2.moments(thresh)
      if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
      else:
        cX, cY = 0, 0
      
      cv2.circle(self.cv_image2, (cX, cY), 2, (255, 255, 255), -1)
      """
      cv2.putText(self.cv_image2, "Blue Center", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      """
      
      return np.array([cX, cY])

  def detect_yellow(self,image):
      yellowMask = cv2.inRange(self.img2HSV, (20, 100, 100), (40, 255, 255))
      yellowImg = cv2.bitwise_and(self.cv_image2, self.cv_image2, mask=yellowMask)
      yellowImg_grey = cv2.cvtColor(yellowImg, cv2.COLOR_BGR2GRAY)
      ret, thresh = cv2.threshold(yellowImg_grey, 1, 255, 0)
      M = cv2.moments(thresh)
      if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
      else:
        cX, cY = 0, 0
      
      cv2.circle(self.cv_image2, (cX, cY), 2, (255, 255, 255), -1)
      """
      cv2.putText(self.cv_image2, "Yellow Center", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      """
      
      return np.array([cX, cY])

  # Recieve data, process it, and publish
  def callback2(self,data):
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
    
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)
    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    # Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
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


