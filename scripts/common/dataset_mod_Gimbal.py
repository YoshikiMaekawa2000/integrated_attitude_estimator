import sys

import torch.utils.data as data
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat

class AttitudeEstimatorDataset(data.Dataset):
    def __init__(self, data_list, transform, phase, index_dict_path, dim_fc_out, timesteps, deg_threshold, resize, do_white_makeup, do_white_makeup_from_back, whiteup_frame):
        self.data_list = data_list
        self.transform = transform
        self.phase = phase
        self.index_dict_path = index_dict_path
        self.dim_fc_out = dim_fc_out #63
        self.num_frames = timesteps
        self.deg_threshold = deg_threshold #30deg
        self.resize = resize
        self.do_white_makeup = do_white_makeup
        self.do_white_makeup_from_back = do_white_makeup_from_back
        self.whiteup_frame = whiteup_frame

        self.index_dict = []
        self.dict_len = 0

        self.index_dict.append([-1*int(self.deg_threshold)-1, 0])

        with open(index_dict_path) as f:
            reader = csv.reader(f)
            for row in reader:
                tmp_row = [int(row[0]), int(row[1])+1]
                self.index_dict.append(tmp_row)

        self.index_dict.append([int(self.deg_threshold)+1, int(dim_fc_out)-1])

        self.dict_len = len(self.index_dict)


    def search_index(self, number):
        index = int(1000000000)
        for row in self.index_dict:
            if float(number) == float(row[0]):
                index = int(row[1])
                break
            elif float(number) < float(self.index_dict[0][0]): ##-31度以下は-31度として切り上げ
                index = self.index_dict[0][1]
                break
            elif float(number) > float(self.index_dict[self.dim_fc_out-1][0]): #+31度以上は+31度として切り上げ
                index = self.index_dict[self.dim_fc_out-1][1]
                break
        
        return index

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

        array = np.zeros(self.dim_fc_out)
        
        if upper_ind == lower_ind:
            array[upper_ind] = 1.0
        else:
            array[lower_ind] = dist_high
            array[upper_ind] = dist_low

        return array

    def __len__(self):
        return len(self.data_list) - self.num_frames

    def __getitem__(self, index):

        image_array = []
        roll_array = []
        pitch_array = []
        process_time_array = []

        
        for i in range(self.num_frames):
            count = int(self.data_list[index + i][0])
            img_path = self.data_list[index + i][1]
            img_pil = Image.open(img_path)
            #img_pil = img_pil.convert("L") # convert to grayscale
            img_pil = img_pil.convert("RGB")

            if self.do_white_makeup == True and i < self.whiteup_frame and self.do_white_makeup_from_back == False:
                #Convert to white makeup for verification
                img_pil = Image.new('RGB', (self.resize, self.resize), 'white')
                
                # show test
                # arrPIL = np.asarray(img_pil)
                # plt.imshow(arrPIL)
                # plt.show()

            if self.do_white_makeup_from_back == True and self.do_white_makeup == False and i >= (self.num_frames - self.whiteup_frame):
                #Convert to white makeup for verification
                img_pil = Image.new('RGB', (self.resize, self.resize), 'white')
                
                # show test
                # print(i)
                # arrPIL = np.asarray(img_pil)
                # plt.imshow(arrPIL)
                # plt.show()

            tmp_roll = float(self.data_list[index + i][3])
            tmp_pitch = float(self.data_list[index + i][4])
            process_time = float(self.data_list[index + i][2])
            process_time_array.append(process_time)

            # print(self.data_list[index + i])
            # print_roll = tmp_roll / 3.141592 * 180.0
            # print_pitch = tmp_pitch / 3.141592 * 180.0

            # print(print_roll, print_pitch)

            # print(tmp_roll, tmp_pitch)

            # if i == self.num_frames-1:
            #     arrPIL = np.asarray(img_pil)
            #     plt.imshow(arrPIL)
            #     plt.show()


            roll_list = self.float_to_array(tmp_roll)
            pitch_list = self.float_to_array(tmp_pitch)

            roll_numpy = np.array(roll_list)
            pitch_numpy = np.array(pitch_list)
            
            #Convert to tensor
            img_trans, roll_trans, pitch_trans = self.transform(img_pil, roll_numpy, pitch_numpy)

            image_array.append(img_trans)
            roll_array.append(roll_trans)
            pitch_array.append(pitch_trans)

            #print(roll_trans, i)

        concated_image = torch.cat(image_array[:self.num_frames+1], dim=0).reshape(self.num_frames, 3, self.resize, self.resize)
        concated_image = rearrange(concated_image, 't c h w -> c t h w')
        label_roll = roll_array[self.num_frames-1]
        label_pitch = pitch_array[self.num_frames-1]
        label_time = process_time_array[self.num_frames-1]
        
        #print("\n")
        #print(concated_image.size())
        #print("\n")

        # print(roll_array)

        # print(label_roll)
        # print(label_pitch)

        return concated_image, label_roll, label_pitch, label_time