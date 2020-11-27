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

        # initialize publishers to publish the estimated joint angles
        self.joint2_pub = rospy.Publisher("joint2", Float64, queue_size=10)
        self.joint3_pub = rospy.Publisher("joint3", Float64, queue_size=10)
        self.joint4_pub = rospy.Publisher("joint4", Float64, queue_size=10)

        # initialize publishers to publish the 3D positions of the end effector, target, box and FK-predicted end effector
        self.endeffector_pub = rospy.Publisher("visionee_pos", Float64MultiArray, queue_size=10)
        self.target_pub = rospy.Publisher("target_pos", Float64MultiArray, queue_size=10)
        self.box_pub = rospy.Publisher("box_pos", Float64MultiArray, queue_size=10)
        self.FK_pub = rospy.Publisher("fkee_pos", Float64MultiArray, queue_size=10)

        # REDUNDANT PUBLISHERS ONLY BEING USED SINCE PLOTTING WITH ARRAYS WON'T WORK #
        self.targetx_pub = rospy.Publisher("target_xpos", Float64, queue_size=10)
        self.targety_pub = rospy.Publisher("target_ypos", Float64, queue_size=10)
        self.targetz_pub = rospy.Publisher("target_zpos", Float64, queue_size=10)

        self.boxx_pub = rospy.Publisher("box_xpos", Float64, queue_size=10)
        self.boxy_pub = rospy.Publisher("box_ypos", Float64, queue_size=10)
        self.boxz_pub = rospy.Publisher("box_zpos", Float64, queue_size=10)

        self.endeffectorx_pub = rospy.Publisher("visionee_xpos", Float64, queue_size=10)
        self.endeffectory_pub = rospy.Publisher("visionee_ypos", Float64, queue_size=10)
        self.endeffectorz_pub = rospy.Publisher("visionee_zpos", Float64, queue_size=10)

        self.FKx_pub = rospy.Publisher("fkee_xpos", Float64, queue_size=10)
        self.FKy_pub = rospy.Publisher("fkee_ypos", Float64, queue_size=10)
        self.FKz_pub = rospy.Publisher("fkee_zpos", Float64, queue_size=10)

        # initialize a subscriber to receive the position of the target from camera1
        self.target_pos1_sub = message_filters.Subscriber("target_pos1", Float64MultiArray)
        # initialize a subscriber to receive the position of the target from camera2
        self.target_pos2_sub = message_filters.Subscriber("target_pos2", Float64MultiArray)

        # initialize a subscriber to receive the position of the box from camera1
        self.box_pos1_sub = message_filters.Subscriber("box_pos1", Float64MultiArray)
        # initialize a subscriber to receive the position of the box from camera2
        self.box_pos2_sub = message_filters.Subscriber("box_pos2", Float64MultiArray)

        # initialize a subscriber to receive the position of the joints from camera1
        self.joints_pos1_sub = message_filters.Subscriber("joints_pos1", Float64MultiArray)
        # initialize a subscriber to receive the position of the joints from camera2
        self.joints_pos2_sub = message_filters.Subscriber("joints_pos2", Float64MultiArray)

        # synchronize the topics and use the callback function to process them
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.joints_pos1_sub, self.joints_pos2_sub, self.target_pos1_sub, self.target_pos2_sub,
             self.box_pos1_sub, self.box_pos2_sub],
            queue_size=10, slop=0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        # save the last angle of joint2 in order to smooth the next one
        self.lastJoint2Angle = 0

    # takes a nx3 array and merges the data from two sources into 3D coordinates. Returns a nx3 array.
    def calc3dCoords(self, source1, source2):
        coords3d = np.zeros((source1.shape[0], 3))
        for i in range(source1.shape[0]):
            # take the x and y value from each camera
            x = source2[i, 0]
            y = source1[i, 0]
            # if camera1 can see the joint but camera2 cannot
            if source1[i, 2] == 0 and source2[i, 2] != 0:
                # use z from camera1
                z = source1[i, 1]
            # else if camera1 cannot see the joint but camera2 can
            elif source1[i, 2] != 0 and source2[i, 2] == 0:
                # use z from camera2
                z = source2[i, 1]
            # else if both cameras have the same amount of information
            else:
                # change weight of each z value depending on how close the point is to each camera
                # to counteract inaccuracies from distance
                x_inac = 1 - (abs(x - 400)/300)
                y_inac = 1 - (abs(y - 400)/300)

                # calculate the weight of each input according to their inaccuracies relative to each other
                x_w = x_inac/(x_inac + y_inac)
                y_w = 1 - x_w

                # find z by adding the z from each source according to their weights
                z = (x_w * source1[i, 1]) + (y_w * source2[i, 1])
                # average the z from both cameras. Not used now since it's less accurate
                #z = ((source1[i, 1] + source2[i, 1]) / 2)
            coords3d[i] = [x, y, z]
        return coords3d

    # makes a given array of points relative to the yellow joint (joint1)
    def makeRelative(self, coordinates):
        relativePositions = np.zeros(coordinates.shape)
        relativePositions[:, 0] = coordinates[:, 0] - self.joints_pos_orig[0, 0]
        relativePositions[:, 1] = coordinates[:, 1] - self.joints_pos_orig[0, 1]
        relativePositions[:, 2] = self.joints_pos_orig[0, 2] - coordinates[:, 2]
        return relativePositions

    # converts a given array of points to metres from pixels
    def pixel2meter(self, coordinates):
        # use a constant as there is no reliable way to get a ratio due to joints moving in place and being obscured
        return coordinates / 25.88333468967014

    def calcJoint2Angle(self):
        # vary point of rotation since actual rotation is not straight
        if self.joints_pos[1, 1] < -0.00336898:
            orientation = np.array([0, -0.00336898, 2.16924723])
        else:
            orientation = np.array([0, -0.00336898, 2.21869815])

        # use atan2 to find rotation around x-axis based on how the joint physically rotates
        joint2Angle = np.arctan2(self.joints_pos[1, 2] - orientation[2],
                                 self.joints_pos[1, 1] - orientation[1])

        # offset angle to come from north (atan2 calculates relative to east)
        if joint2Angle < -np.pi / 2:
            joint2Angle = np.pi / 2
        elif joint2Angle < 0:
            joint2Angle = -np.pi / 2
        else:
            joint2Angle -= np.pi / 2

        # weigh the angle with the last one to smooth it out
        difference = self.lastJoint2Angle - joint2Angle
        joint2Angle += difference / 2

        # cap the angle at -pi/2 to pi/2
        if joint2Angle > (np.pi / 2):
            joint2Angle = np.pi / 2
        elif joint2Angle < -(np.pi / 2):
            joint2Angle = -np.pi / 2

        self.lastJoint2Angle = joint2Angle
        return joint2Angle

    def calcJoint3Angle(self, joint2Angle):
        # vary point of rotation since actual rotation is not straight
        if self.joints_pos[1, 1] < -0.00336898:
            orientation = np.array([0, -0.00336898, 2.16924723])
        else:
            orientation = np.array([0, -0.00336898, 2.21869815])

        # make green joint relative to point of rotation
        x = self.joints_pos[2, 0] - orientation[0]
        y = self.joints_pos[2, 1] - orientation[1]
        z = self.joints_pos[2, 2] - orientation[2]
        theta = -joint2Angle

        # rotate the green point around the x-axis by the opposite of joint2
        xrot = np.array([x,
                         y * np.cos(theta) - z * np.sin(theta),
                         y * np.sin(theta) + z * np.cos(theta)])

        # calculate the rotation around y-axis using atan2 now that it is on the XZ plane
        joint3Angle = np.arctan2(xrot[2], xrot[0])

        # offset angle to come from north (atan2 calculates relative to east)
        if joint3Angle < -np.pi / 2:
            joint3Angle = np.pi / 2
        elif joint3Angle < 0:
            joint3Angle = -np.pi / 2
        else:
            joint3Angle -= np.pi / 2

        # cap the angle at -pi/2 to pi/2
        if joint3Angle > (np.pi / 2):
            joint3Angle = np.pi / 2
        elif joint3Angle < -(np.pi / 2):
            joint3Angle = -np.pi / 2

        return -joint3Angle

    def calcJoint4Angle(self, joint2Angle, joint3Angle):
        # vary point of rotation since actual rotation is not straight
        if self.joints_pos[1, 1] < -0.00336898:
            orientation = np.array([0, -0.00336898, 2.16924723])
        else:
            orientation = np.array([0, -0.00336898, 2.21869815])

        # make green joint relative to point of rotation
        x1 = self.joints_pos[2, 0] - orientation[0]
        y1 = self.joints_pos[2, 1] - orientation[1]
        z1 = self.joints_pos[2, 2] - orientation[2]
        # make red joint relative to point of rotation
        x2 = self.joints_pos[3, 0] - orientation[0]
        y2 = self.joints_pos[3, 1] - orientation[1]
        z2 = self.joints_pos[3, 2] - orientation[2]
        theta = -joint3Angle

        # print(self.joints_pos[2:4])
        # print("-")
        # rotate the green point around the y-axis by the opposite of joint3
        yrot1 = np.array([x1 * np.cos(theta) + z1 * np.sin(theta),
                          y1,
                          -x1 * np.sin(theta) + z1 * np.cos(theta)])

        # print(yrot1 + orientation)
        # rotate the red point around the y-axis by the opposite of joint3
        yrot2 = np.array([x2 * np.cos(theta) + z2 * np.sin(theta),
                          y2,
                          -x2 * np.sin(theta) + z2 * np.cos(theta)])

        # print(yrot2 + orientation)
        # print("-")
        theta = -joint2Angle

        # rotate the green point around the x-axis by the opposite of joint2
        yxrot1 = np.array([yrot1[0],
                           yrot1[1] * np.cos(theta) - yrot1[2] * np.sin(theta),
                           yrot1[1] * np.sin(theta) + yrot1[2] * np.cos(theta)])

        # rotate the red point around the x-axis by the opposite of joint2
        yxrot2 = np.array([yrot2[0],
                           yrot2[1] * np.cos(theta) - yrot2[2] * np.sin(theta),
                           yrot2[1] * np.sin(theta) + yrot2[2] * np.cos(theta)])

        # put the joints back to original frame
        yxrot1 += orientation
        yxrot2 += orientation
        # print(yrot2)
        # print("#####")

        # put red point relative to green point and find angle with atan2
        # joint4Angle = np.arctan2(yrot2[2] - yrot1[2], yrot2[1] - yrot1[1]) - joint2Angle
        joint4Angle = np.arctan2(yxrot2[2] - yxrot1[2], yxrot2[1] - yxrot1[1])

        # joint4Angle = np.arctan2(self.joints_pos[3, 2] - self.joints_pos[2, 2],
        #                         self.joints_pos[3, 1] - self.joints_pos[2, 1])

        # offset angle to come from north (atan2 calculates relative to east)
        if joint4Angle < -np.pi / 2:
            joint4Angle = np.pi / 2
        elif joint4Angle < 0:
            joint4Angle = -np.pi / 2
        else:
            joint4Angle -= np.pi / 2
        # cap the angle at -pi/2 to pi/2
        if joint4Angle > (np.pi / 2):
            joint4Angle = np.pi / 2
        elif joint4Angle < -(np.pi / 2):
            joint4Angle = -np.pi / 2

        return joint4Angle

    # return the estimated end effector position for a given set of angles using Forward Kinematics
    def forwardKinematics(self, joint1Angle, joint2Angle, joint3Angle, joint4Angle):
        x = 3 * (np.sin(joint3Angle) * np.sin(joint1Angle + np.pi / 2) + np.cos(joint3Angle) * np.cos(
            joint1Angle + np.pi / 2) * np.cos(joint2Angle + np.pi / 2)) * np.cos(joint4Angle) + \
            7 * np.sin(joint3Angle) * np.sin(joint1Angle + np.pi / 2) / 2 - \
            3 * np.sin(joint4Angle) * np.sin(joint2Angle + np.pi / 2) * np.cos(joint1Angle + np.pi / 2) + \
            7 * np.cos(joint3Angle) * np.cos(joint1Angle + np.pi / 2) * np.cos(joint2Angle + np.pi / 2) / 2

        y = 3 * (-np.sin(joint3Angle) * np.cos(joint1Angle + np.pi / 2) + np.sin(joint1Angle + np.pi / 2) * np.cos(
            joint3Angle) * np.cos(joint2Angle + np.pi / 2)) * np.cos(joint4Angle) - \
            7 * np.sin(joint3Angle) * np.cos(joint1Angle + np.pi / 2) / 2 - \
            3 * np.sin(joint4Angle) * np.sin(joint1Angle + np.pi / 2) * np.sin(joint2Angle + np.pi / 2) + \
            7 * np.sin(joint1Angle + np.pi / 2) * np.cos(joint3Angle) * np.cos(joint2Angle + np.pi / 2) / 2

        z = 3 * np.sin(joint4Angle) * np.cos(joint2Angle + np.pi / 2) + \
            3 * np.sin(joint2Angle + np.pi / 2) * np.cos(joint3Angle) * np.cos(joint4Angle) + \
            7 * np.sin(joint2Angle + np.pi / 2) * np.cos(joint3Angle) / 2 + \
            5 / 2
        return np.array([x, y, z])

    def callback(self, camera1data, camera2data, target_data1, target_data2,
                 boxdata1, boxdata2):
        # recieve the position data from each image
        try:
            joints_pos1 = np.asarray(camera1data.data, dtype='float64').reshape(4, 3)
            joints_pos2 = np.asarray(camera2data.data, dtype='float64').reshape(4, 3)
            target_pos1 = np.asarray(target_data1.data, dtype='float64').reshape(1, 3)
            target_pos2 = np.asarray(target_data2.data, dtype='float64').reshape(1, 3)
            box_pos1 = np.asarray(boxdata1.data, dtype='float64').reshape(1, 3)
            box_pos2 = np.asarray(boxdata2.data, dtype='float64').reshape(1, 3)
        except CvBridgeError as e:
            print(e)

        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        # merge the data into 3d position coordinates
        self.joints_pos_orig = self.calc3dCoords(joints_pos1, joints_pos2)
        # make the coordinates relative to the unmoving yellow joint
        self.joints_pos = self.makeRelative(self.joints_pos_orig)
        # make the coordinates in terms of meters
        self.joints_pos = self.pixel2meter(self.joints_pos)

        self.endEffector = Float64MultiArray(data=self.joints_pos[3])
        self.endEffectorx = Float64(data=self.joints_pos[3, 0])
        self.endEffectory = Float64(data=self.joints_pos[3, 1])
        self.endEffectorz = Float64(data=self.joints_pos[3, 2])

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
        # merge the coordinates of the target and make it relative to the yellow joint as well as in metres
        self.target_pos = self.calc3dCoords(target_pos1, target_pos2)
        self.target_pos = self.makeRelative(self.target_pos)
        self.target_pos = self.pixel2meter(self.target_pos)
        self.target_pos = self.target_pos[0]

        self.target = Float64MultiArray(data=self.target_pos)
        self.targetx = Float64(data=self.target_pos[0])
        self.targety = Float64(data=self.target_pos[1])
        self.targetz = Float64(data=self.target_pos[2])

        # merge the coordinates of the box and make it relative to the yellow joint as well as in metres
        self.box_pos = self.calc3dCoords(box_pos1, box_pos2)
        self.box_pos = self.makeRelative(self.box_pos)
        self.box_pos = self.pixel2meter(self.box_pos)
        self.box_pos = self.box_pos[0]

        self.box = Float64MultiArray(data=self.box_pos)
        self.boxx = Float64(data=self.box_pos[0])
        self.boxy = Float64(data=self.box_pos[1])
        self.boxz = Float64(data=self.box_pos[2])

        # Task 3.1 #
        # set the values of each joint, the calculated end effector position will be published
        actualJoint1 = 0
        actualJoint2 = 0
        actualJoint3 = 0
        actualJoint4 = 0
        # calculate the position of the endEffector using joint angles
        fkee = self.forwardKinematics(actualJoint1, actualJoint2, actualJoint3, actualJoint4)
        # set up the data to be published
        self.fkEndEffector = Float64MultiArray(data=fkee)
        self.fkEndEffectorx = Float64(data=fkee[0])
        self.fkEndEffectory = Float64(data=fkee[1])
        self.fkEndEffectorz = Float64(data=fkee[2])

        try:
            # PUBLISH DATA #
            # publish prediction of joint angles
            self.joint2_pub.publish(self.joint2)
            self.joint3_pub.publish(self.joint3)
            self.joint4_pub.publish(self.joint4)

            # publish prediction of target position
            self.target_pub.publish(self.target)
            self.targetx_pub.publish(self.targetx)
            self.targety_pub.publish(self.targety)
            self.targetz_pub.publish(self.targetz)
            self.box_pub.publish(self.box)
            self.boxx_pub.publish(self.boxx)
            self.boxy_pub.publish(self.boxy)
            self.boxz_pub.publish(self.boxz)
            self.endeffector_pub.publish(self.endEffector)
            self.endeffectorx_pub.publish(self.endEffectorx)
            self.endeffectory_pub.publish(self.endEffectory)
            self.endeffectorz_pub.publish(self.endEffectorz)

            # publish Forward Kinematics prediction of end effector
            self.FK_pub.publish(self.fkEndEffector)
            self.FKx_pub.publish(self.fkEndEffectorx)
            self.FKy_pub.publish(self.fkEndEffectory)
            self.FKz_pub.publish(self.fkEndEffectorz)
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
