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
import PIL.Image as Image

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger
import csv

import torch
from torchvision import models
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as nn_functional
import torch.optim as optim
import torch.backends.cudnn as cudnn

from collections import OrderedDict

from tensorboardX import SummaryWriter

from einops import rearrange, reduce, repeat

import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError

#Need in running in ROS
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from models import vit
from models import senet
from common import dataset_mod_Gimbal
from common import dataset_mod_AirSim
from common import make_datalist_mod
from common import data_transform_mod

from integrated_attitude_estimator.msg import EularAngle

class IntegratedAttitudeEstimator:
    def __init__(self, FLAGS, CFG):
        self.count = 0

        self.FLAGS = FLAGS
        self.CFG = CFG

        self.weights_top_directory = self.CFG["weights_top_directory"]
        self.weights_file_name = self.CFG["weights_file_name"]
        self.weights_path = os.path.join(self.weights_top_directory, self.weights_file_name)

        self.save_in_csv = bool(self.CFG["save_in_csv"])
        self.infer_log_top_directory = self.CFG["infer_log_top_directory"]
        self.infer_log_file_name = self.CFG["infer_log_file_name"]
        self.infer_log_file_path = os.path.join(self.infer_log_top_directory, self.infer_log_file_name)

        self.yaml_name = self.CFG["yaml_name"]
        self.yaml_path = os.path.join(self.infer_log_top_directory, self.yaml_name)
        # shutil.copy(FLAGS.config, self.yaml_path)

        self.index_dict_path = self.CFG["index_csv_path"]

        # self.image_topic_name = self.CFG["ros_params"]["image_topic_name"]
        # self.gt_angle_topic_name = self.CFG["ros_params"]["gt_angle_topic_name"]
        # self.inferenced_angle_topic_name = self.CFG["ros_params"]["inferenced_angle_topic_name"]
        # self.absolute_error_topic_name = self.CFG["ros_params"]["absolute_error_topic_name"]

        self.image_topic_name = rospy.get_param("~image_topic_name")
        self.gt_angle_topic_name = rospy.get_param('~ground_truth_angle_topic_name')
        self.inferenced_angle_topic_name = rospy.get_param('~inferenced_angle_topic_name')
        self.absolute_error_topic_name = rospy.get_param('~absolute_error_topic_name')

        self.network_type = str(CFG["hyperparameters"]["network_type"])
        self.img_size = int(self.CFG['hyperparameters']['img_size'])
        self.resize = int(CFG["hyperparameters"]["transform_params"]["resize"])
        self.num_classes = int(self.CFG['hyperparameters']['num_classes'])
        self.num_frames = int(self.CFG['hyperparameters']['num_frames'])
        self.deg_threshold = int(self.CFG['hyperparameters']['deg_threshold'])
        self.mean_element = float(self.CFG['hyperparameters']['mean_element'])
        self.std_element = float(self.CFG['hyperparameters']['std_element'])

        # TimeSformer params
        self.patch_size = int(CFG["hyperparameters"]["timesformer"]["patch_size"])
        self.attention_type = str(CFG["hyperparameters"]["timesformer"]["attention_type"])
        self.depth = int(CFG["hyperparameters"]["timesformer"]["depth"])
        self.num_heads = int(CFG["hyperparameters"]["timesformer"]["num_heads"])

        # SENet params
        self.resnet_model = str(CFG["hyperparameters"]["senet"]["resnet_model"])

        self.value_dict = []
        self.value_dict.append([-1*int(self.deg_threshold)-1, 0])
        with open(self.index_dict_path) as fd:
            reader = csv.reader(fd)
            for row in reader:
                #num = float(row[0])
                tmp_row = [int(row[0]), int(row[1])+1]
                self.value_dict.append(tmp_row)
        self.value_dict.append([int(self.deg_threshold)+1, int(self.num_classes)-1])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        # Network params
        self.load_net = False
        self.net = self.getNetwork()
        self.input_tensor = torch.zeros([self.num_frames, 3, self.img_size, self.img_size], dtype=torch.float32)

        # Transform Params
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((self.mean_element,), (self.std_element,))
        ])

        # ROS Image Subscriber
        self.sub_image = rospy.Subscriber(self.image_topic_name, ImageMsg, self.image_callback, queue_size=1)
        self.image_queue = []
        self.image_count = 0
        self.bridge = CvBridge()

        # ROS GT Angle Subscriber
        self.sub_gt_angle = rospy.Subscriber(self.gt_angle_topic_name, EularAngle, self.gt_angle_callback, queue_size=1)
        self.gt_angle = EularAngle()

        self.rate = rospy.Rate(50) # 50hz

        # Publishing
        self.inferenced_angle = EularAngle()
        self.pub_infer_angle = rospy.Publisher( self.inferenced_angle_topic_name, EularAngle, queue_size=1)

        self.diff_angle = EularAngle()
        self.pub_diff_angle = rospy.Publisher( self.absolute_error_topic_name, EularAngle, queue_size=1)

    def getNetwork(self):
        print("Load Network")
        if self.network_type == "TimeSformer":
            net = vit.TimeSformer(self.img_size, self.patch_size, self.num_classes, self.num_frames, self.depth, self.num_heads, self.attention_type, self.weights_path, 'eval')
        elif self.network_type == "SENet":
            net = senet.SENet(model=self.resnet_model, dim_fc_out=self.num_classes, norm_layer=nn.BatchNorm2d, pretrained_model=self.weights_path, time_step=self.num_frames, use_SELayer=True)
        else:
            print("Error: Network type is not defined")
            quit()

        print(net)
        net.to(self.device)
        net.eval()

        print("Load state_dict")
        if torch.cuda.is_available:
            state_dict = torch.load(self.weights_path, map_location=lambda storage, loc: storage)
            #print(state_dict.keys())
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v

            state_dict = new_state_dict
            print("Load .pth file")
            # print(state_dict.keys())
            # print(state_dict['model.mlp_head_roll.1.weight'].size())
        else:
            state_dict = torch.load(self.weights_path, map_location={"cuda:0": "cpu"})
            print("Load to CPU")

        # net.load_state_dict(state_dict, strict=False)
        net.load_state_dict(state_dict)
        self.load_net = True
        return net

    def gt_angle_callback(self, msg):
        self.gt_angle = msg

    def image_callback(self, msg):
        try:
            # start_clock = time.time()

            self.color_img_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.input_tensor = self.convert_to_tensor().detach().clone()

            # print("Period [s]: ", time.time() - start_clock)
            # if self.image_count < self.num_frames:
            #     self.image_queue.append(self.color_img_cv)
            #     self.image_count += 1
            # elif self.image_count == self.num_frames:
            #     self.image_queue.pop(0)
            #     self.image_queue.append(self.color_img_cv)
            #     if self.load_net == True:
            #         self.network_prediction()
            # else:
            #     print("Error: image_count is out of range")
            #     quit()
        except CvBridgeError as e:
            print(e)

    def network_prediction(self):
        start_clock = time.time()
        # print("Prediction in count: ", self.count)
        input = self.input_tensor.unsqueeze(0)
        input = rearrange(input, 'b t c h w -> b c t h w')
        input = input.to(self.device)

        roll_hist_array = [0.0 for _ in range(self.num_classes)]
        pitch_hist_array = [0.0 for _ in range(self.num_classes)]

        roll_inf, pitch_inf = self.net(input)
        roll_inf = np.array(roll_inf.to('cpu').detach().numpy().copy())
        pitch_inf = np.array(pitch_inf.to('cpu').detach().numpy().copy())

        roll = self.array_to_value_simple(roll_inf)
        pitch = self.array_to_value_simple(pitch_inf)
        
        correct_roll = self.gt_angle.roll/3.141592*180
        correct_pitch = self.gt_angle.pitch/3.141592*180

        diff_roll = np.abs(roll - correct_roll)
        diff_pitch = np.abs(pitch - correct_pitch)

        # diff_total_roll += diff_roll
        # diff_total_pitch += diff_pitch

        # print("------------------------------------")
        # print("Inference    :", self.count)
        # print("Infered Roll :" + str(roll) +  "[deg]")
        # print("GT Roll      :" + str(correct_roll) + "[deg]")
        # print("Infered Pitch:" + str(pitch) + "[deg]")
        # print("GT Pitch     :" + str(correct_pitch) + "[deg]")
        # print("Diff Roll    :" + str(diff_roll) + " [deg]")
        # print("Diff Pitch   :" + str(diff_pitch) + " [deg]")

        tmp_result_csv = [roll, pitch, correct_roll, correct_pitch, diff_roll, diff_pitch]
        self.inferenced_angle.roll = roll/180*3.141592
        self.inferenced_angle.pitch = pitch/180*3.141592
        self.inferenced_angle.yaw = 0.0/180*3.141592
        self.inferenced_angle.header.stamp = rospy.Time.now()

        self.diff_angle.roll = diff_roll/180*3.141592
        self.diff_angle.pitch = diff_pitch/180*3.141592
        self.diff_angle.yaw = 0.0/180*3.141592
        self.diff_angle.header.stamp = rospy.Time.now()

        self.pub_infer_angle.publish(self.inferenced_angle)
        self.pub_diff_angle.publish(self.diff_angle)
    
        # print("Period [s]: ", time.time() - start_clock)

    def convert_to_tensor(self):
        input_tensor = self.input_tensor[1:, :, :, :]
        # count = 0
        # for tmp_img in self.image_queue:
        #     tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        #     tmp_img = Image.fromarray(tmp_img)
        #     tmp_img_tensor = self.img_transform(tmp_img)
        #     tmp_img_tensor = tmp_img_tensor.unsqueeze(0)
        #     if count == 0:
        #         input_tensor = tmp_img_tensor.detach().clone()
        #     else:
        #         input_tensor = torch.cat((input_tensor, tmp_img_tensor), dim=0)
        #     # print(tmp_img_tensor.size())

        #     count += 1

        tmp_img = cv2.cvtColor(self.color_img_cv, cv2.COLOR_BGR2RGB)
        tmp_img = Image.fromarray(tmp_img)
        tmp_img_tensor = self.img_transform(tmp_img)
        tmp_img_tensor = tmp_img_tensor.unsqueeze(0)
        input_tensor = torch.cat((input_tensor, tmp_img_tensor), dim=0)


        # print(input_tensor.size())
        return input_tensor

    def array_to_value_simple(self, output_array):
        max_index = int(np.argmax(output_array))
        plus_index = max_index + 1
        minus_index = max_index - 1
        value = 0.0
        
        for tmp, label in zip(output_array[0], self.value_dict):
            value += tmp * float(label[0])

        if max_index == 0:
            value = -31.0
        elif max_index == 62: #361
            value = 31.0

        return value

    def spin_node(self):
        while not rospy.is_shutdown():
            self.rate.sleep()
            self.network_prediction()
            self.count += 1

        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('ros_infer', anonymous=True)
    parser = argparse.ArgumentParser("./ros_infer.py")

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="../pyyaml/ros_infer_config.yaml",
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

    integrated_attitude_estimator = IntegratedAttitudeEstimator(FLAGS, CFG)
    integrated_attitude_estimator.spin_node()

    rospy.spin()