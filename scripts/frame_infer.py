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
import torch.nn as nn
import torch.nn.functional as nn_functional
import torch.optim as optim
import torch.backends.cudnn as cudnn

from collections import OrderedDict

from tensorboardX import SummaryWriter

import rospy

from models import vit
from models import senet
from common import dataset_mod_Gimbal
from common import dataset_mod_AirSim
from common import make_datalist_mod
from common import data_transform_mod

class FrameInfer:
    def __init__(self, CFG, FLAGS):
        self.cfg = CFG
        self.FLAGS = FLAGS

        self.infer_sequence = self.cfg['infer_dataset']

        self.csv_name = self.cfg['csv_name']

        self.weights_top_directory = self.cfg['weights_top_directory']
        self.weights_file_name = self.cfg['weights_file_name']
        self.weights_path = os.path.join(self.weights_top_directory, self.weights_file_name)

        self.infer_log_top_directory = self.cfg['infer_log_top_directory']
        self.infer_log_file_name = self.cfg['infer_log_file_name']
        self.infer_log_path = os.path.join(self.infer_log_top_directory, self.infer_log_file_name)

        yaml_name = self.cfg['yaml_name']
        yaml_path = self.infer_log_top_directory + yaml_name
        shutil.copy(FLAGS.config, yaml_path)

        self.index_dict_name = self.cfg['index_dict_name']
        self.index_dict_path = "../index_dict/" + self.index_dict_name

        self.index_csv_path = self.cfg['index_csv_path']

        self.dataset_type = str(CFG["dataset_type"])
        print("Training type: ", self.dataset_type)

        self.network_type = str(CFG["hyperparameters"]["network_type"])

        self.img_size = int(self.cfg['hyperparameters']['img_size'])

        self.do_domain_randomization =str(self.cfg['hyperparameters']["transform_params"]['do_domain_randomization'])
        self.resize = int(CFG["hyperparameters"]["transform_params"]["resize"])
        self.brightness = float(CFG["hyperparameters"]["transform_params"]["brightness"])
        self.contrast = float(CFG["hyperparameters"]["transform_params"]["contrast"])
        self.saturation = float(CFG["hyperparameters"]["transform_params"]["saturation"])
        self.hue = float(CFG["hyperparameters"]["transform_params"]["hue"])
        self.kernel_size = int(CFG["hyperparameters"]["transform_params"]["kernel_size"])
        self.sigma_min = float(CFG["hyperparameters"]["transform_params"]["sigma_min"])
        self.sigma_max = float(CFG["hyperparameters"]["transform_params"]["sigma_max"])
        self.equalize_p = float(CFG["hyperparameters"]["transform_params"]["equalize_p"])
        self.elastic_alpha = float(CFG["hyperparameters"]["transform_params"]["elastic_alpha"])

        self.num_classes = int(self.cfg['hyperparameters']['num_classes'])
        self.num_frames = int(self.cfg['hyperparameters']['num_frames'])
        self.deg_threshold = int(self.cfg['hyperparameters']['deg_threshold'])
        self.mean_element = float(self.cfg['hyperparameters']['mean_element'])
        self.std_element = float(self.cfg['hyperparameters']['std_element'])
        self.do_white_makeup = bool(self.cfg['hyperparameters']['do_white_makeup'])
        self.do_white_makeup_from_back = bool(self.cfg['hyperparameters']['do_white_makeup_from_back'])
        self.whiteup_frame = int(self.cfg['hyperparameters']['whiteup_frame'])

        # TimeSformer params
        self.patch_size = int(CFG["hyperparameters"]["timesformer"]["patch_size"])
        self.attention_type = str(CFG["hyperparameters"]["timesformer"]["attention_type"])
        self.depth = int(CFG["hyperparameters"]["timesformer"]["depth"])
        self.num_heads = int(CFG["hyperparameters"]["timesformer"]["num_heads"])

        # SENet params
        self.resnet_model = str(CFG["hyperparameters"]["senet"]["resnet_model"])

        #self.transform = data_transform_mod.DataTransform(self.resize, self.mean_element, self.std_element)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        self.net = self.getNetwork()

        self.value_dict = []

        self.value_dict.append([-1*int(self.deg_threshold)-1, 0])

        with open(self.index_dict_path) as fd:
            reader = csv.reader(fd)
            for row in reader:
                #num = float(row[0])
                tmp_row = [int(row[0]), int(row[1])+1]
                self.value_dict.append(tmp_row)

        self.value_dict.append([int(self.deg_threshold)+1, int(self.num_classes)-1])

        # print("Value: ", self.value_dict)


        #self.data_list = self.getDatalist()

        print("Load Test Dataset")

        if self.dataset_type == "AirSim":
            self.test_dataset = dataset_mod_AirSim.AttitudeEstimatorDataset(
                data_list = make_datalist_mod.makeMultiDataList(self.infer_sequence, self.csv_name),
                transform = data_transform_mod.DataTransform(
                    self.resize,
                    self.mean_element,
                    self.std_element,
                    self.brightness,
                    self.contrast,
                    self.saturation,
                    self.hue,
                    self.kernel_size,
                    self.sigma_min,
                    self.sigma_max,
                    self.equalize_p,
                    self.elastic_alpha,
                    self.do_domain_randomization
                ),
                phase = "valid",
                index_dict_path = self.index_csv_path,
                dim_fc_out = self.num_classes,
                timesteps = self.num_frames,
                deg_threshold = self.deg_threshold,
                resize = self.resize,
                do_white_makeup = self.do_white_makeup,
                do_white_makeup_from_back = self.do_white_makeup_from_back,
                whiteup_frame = self.whiteup_frame
            )
        elif self.dataset_type == "Gimbal":
            self.test_dataset = dataset_mod_Gimbal.AttitudeEstimatorDataset(
                data_list = make_datalist_mod.makeMultiDataList(self.infer_sequence, self.csv_name),
                transform = data_transform_mod.DataTransform(
                    self.resize,
                    self.mean_element,
                    self.std_element,
                    self.brightness,
                    self.contrast,
                    self.saturation,
                    self.hue,
                    self.kernel_size,
                    self.sigma_min,
                    self.sigma_max,
                    self.equalize_p,
                    self.elastic_alpha,
                    self.do_domain_randomization
                ),
                phase = "valid",
                index_dict_path = self.index_csv_path,
                dim_fc_out = self.num_classes,
                timesteps = self.num_frames,
                deg_threshold = self.deg_threshold,
                resize = self.resize,
                do_white_makeup = self.do_white_makeup,
                do_white_makeup_from_back = self.do_white_makeup_from_back,
                whiteup_frame = self.whiteup_frame
            )
        else:
            print("Error: train_type is not defined")
            quit()

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
        return net

    def spin(self):
        print("Start inference")

        result_csv = []
        infer_count = 0

        diff_total_roll = 0.0
        diff_total_pitch = 0.0

        ros_init_time = 0.0

        for input_image, label_roll, label_pitch, label_time in self.test_dataset:
            start_clock = time.time()
            input_image = input_image.unsqueeze(dim=0)
            input_image = input_image.to(self.device)

            roll_hist_array = [0.0 for _ in range(self.num_classes)]
            pitch_hist_array = [0.0 for _ in range(self.num_classes)]

            roll_inf, pitch_inf = self.prediction(input_image)

            roll = self.array_to_value_simple(roll_inf)
            pitch = self.array_to_value_simple(pitch_inf)

            correct_roll = self.array_to_value_simple_label(np.array(label_roll.to('cpu').detach().numpy().copy()))
            correct_pitch = self.array_to_value_simple_label(np.array(label_pitch.to('cpu').detach().numpy().copy()))


            roll_hist_array += roll_inf[0]
            pitch_hist_array += pitch_inf[0]

            diff_roll = np.abs(roll - correct_roll)
            diff_pitch = np.abs(pitch - correct_pitch)

            diff_total_roll += diff_roll
            diff_total_pitch += diff_pitch

            inference_time = 0.0

            if self.dataset_type == "AirSim":
                inference_time = label_time
            elif self.dataset_type == "Gimbal":
                if infer_count == 0:
                    ros_init_time = rospy.Time(0, label_time)
                    inference_time = (rospy.Time(0, label_time) - ros_init_time).to_sec()
                else:
                    inference_time = (rospy.Time(0, label_time) - ros_init_time).to_sec()

            print("------------------------------------")
            print("Inference: ", infer_count)
            print("Inference Time: ", inference_time)
            print("Infered Roll:  " + str(roll) +  "[deg]")
            print("GT Roll:       " + str(correct_roll) + "[deg]")
            print("Infered Pitch: " + str(pitch) + "[deg]")
            print("GT Pitch:      " + str(correct_pitch) + "[deg]")
            print("Diff Roll: " + str(diff_roll) + " [deg]")
            print("Diff Pitch: " + str(diff_pitch) + " [deg]")

            tmp_result_csv = [roll, pitch, correct_roll, correct_pitch, diff_roll, diff_pitch, inference_time]
            result_csv.append(tmp_result_csv)

            print("Period [s]: ", time.time() - start_clock)
            print("------------------------------------")

            infer_count += 1



        print("Inference Test Has Done....")
        print("Average of Error of Roll : " + str(diff_total_roll/float(infer_count)) + " [deg]")
        print("Average of Error of Pitch: " + str(diff_total_pitch/float(infer_count)) + " [deg]")
        return result_csv

    def save_csv(self, result_csv):
        csv_file = open(self.infer_log_path, 'w')
        csv_w = csv.writer(csv_file)
        for row in result_csv:
            csv_w.writerow(row)
        csv_file.close()
        print("Save Inference Data")

    def prediction(self, img_list):
        roll_inf, pitch_inf = self.net(img_list)
        output_roll_array = roll_inf.to('cpu').detach().numpy().copy()
        output_pitch_array = pitch_inf.to('cpu').detach().numpy().copy()

        return np.array(output_roll_array), np.array(output_pitch_array)

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

    def array_to_value_simple_label(self, label_array):
        max_index = int(np.argmax(label_array))
        plus_index = max_index + 1
        minus_index = max_index - 1
        value = 0.0

        for tmp, label in zip(label_array, self.value_dict):
            value += tmp * float(label[0])

        if max_index == 0:
            value = -31.0
        elif max_index == 62: #361
            value = 31.0

        return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser("./frame_infer.py")

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="../pyyaml/frame_infer_config.yaml",
        required=False,
        help="Infer config file path"
    )

    FLAGS, unparsed = parser.parse_known_args()

    try:
        print("Opening Infer config file", FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Failed to open config file", FLAGS.config)
        quit()


    frame_infer = FrameInfer(CFG, FLAGS)  #CFG:yamlファイルを開いたもの． FLAGS:Namespace(config="hogeohge.yaml")
    result_csv = frame_infer.spin()
    frame_infer.save_csv(result_csv)
