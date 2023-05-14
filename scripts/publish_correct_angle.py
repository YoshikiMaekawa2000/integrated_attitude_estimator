#!/usr/bin/env python3

import sys, codecs
from tkinter import W
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout)
sys.dont_write_bytecode = True

from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import random
import cv2

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger
import csv

import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError

#Need in running in ROS
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from integrated_attitude_estimator.msg import EularAngle

class PublishCorrectAngle:
    def __init__(self, FLAGS, CFG):
        self.FLAGS = FLAGS
        self.CFG = CFG

        # self.gimbal_angle_topic_name = self.CFG["ros_params"]["gimbal_angle_topic_name"]
        # self.imu_calibrated_angle_topic_name = self.CFG["ros_params"]["imu_calibrated_angle_topic_name"]
        # self.gt_angle_topic_name = self.CFG["ros_params"]["gt_angle_topic_name"]

        self.gimbal_angle_topic_name = rospy.get_param("~gimbal_angle_topic_name", "/gimbal_angle")
        self.imu_calibrated_angle_topic_name = rospy.get_param("~imu_calibrated_angle_topic_name", "/imu_correct_angle")
        self.gt_angle_topic_name = rospy.get_param("~gt_angle_topic_name", "/gt_correct_angle")

        self.gimbal_angle = EularAngle()
        self.imu_calibrated_angle = EularAngle()
        self.gt_angle = EularAngle()

        self.gimbal_angle_sub = rospy.Subscriber(self.gimbal_angle_topic_name, EularAngle, self.gimbal_angle_callback)
        self.imu_calibrated_angle_sub = rospy.Subscriber(self.imu_calibrated_angle_topic_name, EularAngle, self.imu_calibrated_angle_callback)
        self.gt_angle_pub = rospy.Publisher(self.gt_angle_topic_name, EularAngle, queue_size=1)

    def gimbal_angle_callback(self, msg):
        self.gimbal_angle = msg
    
    def imu_calibrated_angle_callback(self, msg):
        self.imu_calibrated_angle = msg
        self.publish_gt_angle()

    def publish_gt_angle(self):
        self.gt_angle.roll = (self.gimbal_angle.roll + self.imu_calibrated_angle.roll)/2.0
        self.gt_angle.pitch = (self.gimbal_angle.pitch + self.imu_calibrated_angle.pitch)/2.0
        self.gt_angle.yaw = (self.gimbal_angle.yaw + self.imu_calibrated_angle.yaw)/2.0
        self.gt_angle.header.stamp = rospy.Time.now()
        self.gt_angle_pub.publish(self.gt_angle)

if __name__ == '__main__':
    rospy.init_node('publsih_correct_angle', anonymous=True)
    parser = argparse.ArgumentParser("./publish_correct_angle.py")

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="../pyyaml/publish_correct_angle_config.yaml",
        required=False,
        help="Infer config file path"
    )

    FLAGS, unparsed = parser.parse_known_args()

    try:
        print("Opening Infer config file %s", FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Failed to open config file %s", FLAGS.config)
        quit()

    publish_correct_angle = PublishCorrectAngle(FLAGS, CFG)
    rospy.spin()