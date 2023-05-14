import os
import cv2
import PIL.Image as PILIMAGE
import math
import numpy as np
import time
import argparse
from numpy.core.fromnumeric import argmin
from torch.utils import data
import yaml
import csv
import random
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import urllib
import pickle
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import scipy.stats as stats

from sklearn.mixture import GaussianMixture
from collections import OrderedDict
from einops import rearrange, reduce, repeat

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as nn_functional

#Grad CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
from models import vit
from models import senet


class GradCam:
    def __init__(self, CFG):
        self.CFG = CFG
        self.infer_dataset_top_directory = str(CFG['infer_dataset_top_directory'])

        self.camera_image_directory = str(CFG['camera_image_directory'])
        self.image_num = int(CFG['image_num'])
        # self.image_format = str(CFG['image_format'])
        self.csv_name = str(CFG['csv_name'])

        self.weights_top_directory = str(CFG['weights_top_directory'])
        self.weights_file_name = str(CFG['weights_file_name'])
        self.weights_path = os.path.join(self.weights_top_directory, self.weights_file_name)

        self.save_image_top_directory = str(CFG['save_image_top_directory'])
        self.save_image_file_format = str(CFG['save_image_file_format'])

        self.index_csv_path = CFG['index_csv_path']

        self.img_size = int(self.CFG['hyperparameters']['img_size'])
        
        self.do_domain_randomization =str(self.CFG['hyperparameters']["transform_params"]['do_domain_randomization'])
        self.resize = int(CFG["hyperparameters"]["transform_params"]["resize"])

        self.num_classes = int(self.CFG['hyperparameters']['num_classes'])
        self.num_frames = int(self.CFG['hyperparameters']['num_frames'])
        self.deg_threshold = int(self.CFG['hyperparameters']['deg_threshold'])
        self.mean_element = float(self.CFG['hyperparameters']['mean_element'])
        self.std_element = float(self.CFG['hyperparameters']['std_element'])
        self.network_type = self.CFG['hyperparameters']['network_type']

        # TimeSformer params
        self.patch_size = int(CFG["hyperparameters"]["timesformer"]["patch_size"])
        self.attention_type = str(CFG["hyperparameters"]["timesformer"]["attention_type"])
        self.depth = int(CFG["hyperparameters"]["timesformer"]["depth"])
        self.num_heads = int(CFG["hyperparameters"]["timesformer"]["num_heads"])

        # SENet params
        self.resnet_model = str(CFG["hyperparameters"]["senet"]["resnet_model"])

        self.image_cv = np.empty(0)
        self.depth_cv = np.empty(0)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ",self.device)

        self.img_transform = self.getImageTransform(self.mean_element, self.std_element, self.resize)

        self.net = self.getNetwork()

        self.target_layer = self.net.feature_extractor
        self.target_layer_roll = self.net.fully_connected.roll_fc[-1]
        self.target_layer_pitch = self.net.fully_connected.pitch_fc[-1]

        self.gradcam_roll = GradCAM(model = self.net, target_layer = self.target_layer_roll, use_cuda = torch.cuda.is_available())
        self.gradcam_pitch = GradCAM(model = self.net, target_layer = self.target_layer_pitch, use_cuda = torch.cuda.is_available())


        self.value_dict = []

        with open(self.index_csv_path) as fd:
            reader = csv.reader(fd)
            for row in reader:
                num = float(row[0])
                self.value_dict.append(num)

        self.index_dict = []
        self.dict_len = 0

        self.index_dict.append([-1*int(self.deg_threshold)-1, 0])

        with open(self.index_csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                tmp_row = [int(row[0]), int(row[1])+1]
                self.index_dict.append(tmp_row)

        self.index_dict.append([int(self.deg_threshold)+1, int(self.num_classes)-1])

        self.dict_len = len(self.index_dict)

        self.image_data_list, self.data_list = self.get_data()

    def getImageTransform(self, mean_element, std_element, resize):

        mean = mean_element
        std = std_element
        size = (resize, resize)

        img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

        return img_transform

    def ImageTransform(self, img_pil, roll_numpy, pitch_numpy):
        ## img: numpy -> tensor
        img_tensor = self.img_transform(img_pil)
        
        ## roll: numpy -> tensor
        roll_numpy = roll_numpy.astype(np.float32)
        roll_tensor = torch.from_numpy(roll_numpy)

        # pitch: numpy -> tensor
        pitch_numpy = pitch_numpy.astype(np.float32)
        pitch_tensor = torch.from_numpy(pitch_numpy)
        
        #return img_tensor, logged_roll_tensor, logged_pitch_tensor
        return img_tensor, roll_tensor, pitch_tensor

    def get_data(self):
        image_data_list = []
        data_list = []

        csv_path = os.path.join(self.infer_dataset_top_directory, self.csv_name)

        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                img_path = self.infer_dataset_top_directory + self.camera_image_directory + "/" +row[1]
                
                gt_roll = float(row[3])/3.141592*180.0
                gt_pitch = float(row[4])/3.141592*180.0

                #print(img_path)

                image_data_list.append(img_path)
                tmp_row = [row[1], gt_roll, gt_pitch]
                data_list.append(tmp_row)

            # print(image_data_list)

        return image_data_list, data_list

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
        image_array = []
        roll_array = []
        pitch_array = []
        cv_array = []

        roll_val_list = []
        pitch_val_list = []

        index = self.image_num
        
        for i in range(self.num_frames):
            img_path = self.image_data_list[index + i]
            # print(img_path)
            img_pil = Image.open(img_path)
            #img_pil = img_pil.convert("L") # convert to grayscale
            img_pil = img_pil.convert("RGB")

            tmp_cv_image = cv2.imread(img_path)
            cv_array.append(tmp_cv_image)

            # arrPIL = np.asarray(img_pil)
            # plt.imshow(arrPIL)
            # plt.show()

            tmp_roll = float(self.data_list[index + i][1])
            tmp_pitch = float(self.data_list[index + i][2])

            roll_val_list.append(tmp_roll)
            pitch_val_list.append(tmp_pitch)

            roll_list = self.float_to_array(tmp_roll)
            pitch_list = self.float_to_array(tmp_pitch)

            roll_numpy = np.array(roll_list)
            pitch_numpy = np.array(pitch_list)
            
            #Convert to tensor
            img_trans, roll_trans, pitch_trans = self.ImageTransform(img_pil, roll_numpy, pitch_numpy)

            image_array.append(img_trans)
            roll_array.append(roll_trans)
            pitch_array.append(pitch_trans)

            #print(roll_trans, i)

        # print(pil_array[0].size())
        label_image_cv = cv_array[self.num_frames-1]
        concated_image = torch.cat(image_array[:self.num_frames+1], dim=0).reshape(self.num_frames, 3, self.resize, self.resize)
        input_tensor = rearrange(concated_image, 't c h w -> c t h w')
        label_roll = roll_array[self.num_frames-1]
        label_pitch = pitch_array[self.num_frames-1]
        label_roll_val = roll_val_list[self.num_frames-1]
        label_pitch_val = pitch_val_list[self.num_frames-1]

        print("Start Inference")
        input_image = input_tensor.unsqueeze(dim=0)
        input_image = input_image.to(self.device)
        roll_output_array, pitch_output_array = self.inference(input_image)

        vis_image = cv2.resize(label_image_cv, (224, 224)) // 255.0

        label = 0

        print(type(input_image))

        grayscale_cam_roll = self.gradcam_roll(input_image)
        grayscale_cam_roll = grayscale_cam_roll[0, :]
        visualization_roll = show_cam_on_image(vis_image, grayscale_cam_roll, use_rgb = True)

        grayscale_cam_pitch = self.gradcam_pitch(input_tensor = input_image)
        grayscale_cam_pitch = grayscale_cam_pitch[0, :]
        visualization_pitch = show_cam_on_image(vis_image, grayscale_cam_pitch, use_rgb = True)

    def inference(self, input_tensor):
        roll_array, pitch_array = self.net(input_tensor)

        output_roll_array = roll_array.to('cpu').detach().numpy().copy()
        output_pitch_array = pitch_array.to('cpu').detach().numpy().copy()

        return np.array(output_roll_array), np.array(output_pitch_array)

    def float_to_array(self, num_float):
        num_deg = float((num_float/3.141592)*180.0)

        num_upper = 0.0
        num_lower = 0.0

        tmp_deg = float(int(num_deg))
        if tmp_deg < num_deg: # 0 < num_deg
            num_lower = tmp_deg
            num_upper = num_lower + 1.0
        elif num_deg < tmp_deg: # tmp_deg < 0
            num_lower = tmp_deg - 1.0
            num_upper = tmp_deg

        dist_low = math.fabs(num_deg - num_lower)
        dist_high = math.fabs(num_deg - num_upper)

        lower_ind = int(self.search_index(num_lower))
        upper_ind = int(self.search_index(num_upper))

        array = np.zeros(self.num_classes)
        
        if upper_ind == lower_ind:
            array[upper_ind] = 1.0
        else:
            array[lower_ind] = dist_high
            array[upper_ind] = dist_low

        return array

    def search_index(self, number):
        index = int(1000000000)
        for row in self.index_dict:
            if float(number) == float(row[0]):
                index = int(row[1])
                break
            elif float(number) < float(self.index_dict[0][0]): ##-31度以下は-31度として切り上げ
                index = self.index_dict[0][1]
                break
            elif float(number) > float(self.index_dict[self.num_classes-1][0]): #+31度以上は+31度として切り上げ
                index = self.index_dict[self.num_classes-1][1]
                break
        
        return index

if __name__ == '__main__':

    parser = argparse.ArgumentParser("./visualize_grad_cam.py")
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default='./visualize_grad_cam_config.yaml',
        help='Grad Cam Config'
    )

    FLAGS, unparsed = parser.parse_known_args()

    #Load yaml file
    try:
        print("Opening grad cam config file %s", FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening grad cam config file %s", FLAGS.config)
        quit()
    
    grad_cam = GradCam(CFG)
    grad_cam.spin()