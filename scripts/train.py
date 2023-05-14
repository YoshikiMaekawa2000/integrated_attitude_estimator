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
import csv
import __init__ as booger

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as nn_functional
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

from models import vit
from models import senet
from common import dataset_mod_Gimbal
from common import dataset_mod_AirSim
from common import make_datalist_mod
from common import data_transform_mod
from einops import rearrange
from collections import OrderedDict

class Trainer:
    def __init__(self,
        save_top_path,
        pretrained_weights_path,
        index_dict_path,
        multiGPU,
        img_size,
        mean_element,
        std_element,
        num_classes,
        deg_threshold,
        batch_size,
        num_epochs,
        optimizer_name,
        lr_feature,
        lr_fc,
        alpha,
        num_frames,
        patch_size,
        net,
        train_dataset,
        distort_dataset,
        valid_dataset,
        num_workers,
        save_step,
        distort_epoch,
        train_system):

        print("Activated Training Functions")

        self.save_top_path = save_top_path
        self.pretrained_weights_path = pretrained_weights_path
        self.index_dict_path = index_dict_path
        self.multiGPU = multiGPU
        self.img_size = img_size
        self.mean_element = mean_element
        self.std_element = std_element
        self.num_classes = num_classes
        self.deg_threshold = deg_threshold
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer_name = optimizer_name
        self.lr_feature = lr_feature
        self.lr_fc = lr_fc
        self.alpha = alpha
        self.num_frames = num_frames
        self.patch_size = patch_size

        self.tmp_net = net

        self.train_dataset = train_dataset
        self.distort_dataset = distort_dataset
        self.valid_dataset = valid_dataset

        self.num_workers = num_workers
        self.save_step = save_step
        self.distort_epoch = distort_epoch

        self.train_system = train_system

        if self.multiGPU == 0:
                self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.setRandomCondition()
        self.dataloaders_dict = self.getDataloaders(self.train_dataset, self.distort_dataset, self.valid_dataset, batch_size)
        self.net = self.getNetwork(net)
        self.optimizer = self.getOptimizer(self.optimizer_name, self.lr_feature, self.lr_fc)

        self.value_dict = []

        self.value_dict.append([-1*int(self.deg_threshold)-1, 0])

        with open(self.index_dict_path) as fd:
            reader = csv.reader(fd)
            for row in reader:
                #num = float(row[0])
                tmp_row = [int(row[0]), int(row[1])+1]
                self.value_dict.append(tmp_row)

        self.value_dict.append([int(self.deg_threshold)+1, int(self.num_classes)-1])
        # print(self.value_dict)


    def setRandomCondition(self, keep_reproducibility=False, seed=123456789):
        if keep_reproducibility:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def getDataloaders(self, train_dataset, distort_dataset, valid_dataset, batch_size):

        distort_batch_size = int(batch_size)
        if (batch_size % 2) == 0:
            distort_batch_size = int(batch_size/2)
        else:
            distort_batch_size = int(batch_size/2) + 1


        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle=True,
            num_workers = self.num_workers,
            #pin_memory =True
        )

        distort_dataloader = torch.utils.data.DataLoader(
            distort_dataset,
            batch_size = distort_batch_size,
            shuffle=True,
            num_workers = self.num_workers,
            #pin_memory =True
        )

        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size = batch_size,
            shuffle=True,
            num_workers = self.num_workers,
            #pin_memory = True
        )

        dataloaders_dict = {"train":train_dataloader, "distort":distort_dataloader, "valid":valid_dataloader}

        return dataloaders_dict

    def getOptimizer(self, optimizer_name, lr_feature, lr_fc):

        list_feature_extractor_param_value, list_roll_fc_param_value, list_pitch_fc_param_value = self.tmp_net.getParamValueList()

        if optimizer_name == "SGD":
            #optimizer = optim.SGD(self.net.parameters() ,lr = lr_feature, momentum=0.9, weight_decay=0.0)
            optimizer = optim.SGD([
                {"params": list_feature_extractor_param_value, "lr": lr_feature},
                {"params": list_roll_fc_param_value, "lr": lr_fc},
                {"params": list_pitch_fc_param_value, "lr": lr_fc},
            ], momentum=0.9, weight_decay=0.0)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam([
                {"params": list_feature_extractor_param_value, "lr": lr_feature},
                {"params": list_roll_fc_param_value, "lr": lr_fc},
                {"params": list_pitch_fc_param_value, "lr": lr_fc},
            ], momentum=0.9, weight_decay=0.0)
        elif optimizer_name == "RAdam":
            #optimizer = optim.RAdam(self.net.parameters(), lr = lr_vit, weight_decay=0.0)
            optimizer = optim.RAdam([
                {"params": list_feature_extractor_param_value, "lr": lr_feature},
                {"params": list_roll_fc_param_value, "lr": lr_fc},
                {"params": list_pitch_fc_param_value, "lr": lr_fc},
            ], weight_decay=0.0)

        print("optimizer: {}".format(optimizer_name))
        return optimizer

    def fix_model_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        return new_state_dict

    def getNetwork(self, net):
        print("Loading Pretrained Network")
        pretrained_state_dict = torch.load(self.pretrained_weights_path)
        #print(pretrained_model)
        if train_system == "finetune":
            net.load_state_dict(self.fix_model_state_dict(pretrained_state_dict))
        else:
            net.load_state_dict(pretrained_state_dict, strict=False)
        # print("Model's state_dict:")
        # for param_tensor in net.state_dict():
        #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        net = net.to(self.device)

        if self.multiGPU == 1 and self.device == "cuda":
            net = nn.DataParallel(net)
            cudnn.benchmark = True
            print("Training on multiGPU Device")
        else:
            cudnn.benchmark = True
            print("Training on Single GPU Device")

        return net

    def process(self):
        start_clock = time.time()
        
        #Loss recorder
        writer = SummaryWriter(log_dir = self.save_top_path + "/log")

        record_train_loss = []
        record_distort_loss = []
        record_valid_loss = []

        for epoch in range(self.num_epochs):
            print("--------------------------------")
            print("Epoch: {}/{}".format(epoch+1, self.num_epochs))

            for phase in ["train", "distort", "valid"]:
                if phase == "train":
                    self.net.train()
                elif phase == "distort":
                    self.net.train()
                elif phase == "valid":
                    self.net.eval()

                if phase == "distort" and (epoch%self.distort_epoch)!=0:
                    continue
                elif phase == "distort" and (epoch%distort_epoch)==0:
                    print("Distort Phase in Epoch: {}".format(epoch))
                
                #Data Load
                epoch_loss = 0.0

                for img_list, label_roll, label_pitch, label_time in tqdm(self.dataloaders_dict[phase]):
                    self.optimizer.zero_grad()

                    # # To check image transform
                    # img_show = rearrange(img_list, 'b c t h w -> b t h w c')
                    # arrPIL = np.asarray(img_show[0][0].detach().numpy().copy())
                    # plt.imshow(arrPIL)
                    # plt.show()

                    img_list = img_list.to(self.device)

                    label_roll = label_roll.to(self.device)
                    label_pitch = label_pitch.to(self.device)

                    #Reset Gradient
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=="train" or phase=="distort"):
                        roll_inf, pitch_inf = self.net(img_list)

                        logged_roll_inf = nn_functional.log_softmax(roll_inf, dim=1)
                        logged_pitch_inf = nn_functional.log_softmax(pitch_inf, dim=1)

                        roll_loss = torch.mean(torch.sum(-label_roll*logged_roll_inf, 1))
                        pitch_loss = torch.mean(torch.sum(-label_pitch*logged_pitch_inf, 1))

                        # torch.set_printoptions(edgeitems=1000000)

                        # print_infer_roll = self.array_to_value_simple(roll_inf.to('cpu').detach().numpy().copy())
                        # print_label_roll = self.array_to_value_simple_label_in_train(label_roll.to('cpu').detach().numpy().copy())
                        # print(roll_inf)
                        # print(label_roll)
                        # print("Infer Roll: {}".format(print_infer_roll))
                        # print("Label Roll: {}".format(print_label_roll))
                        # print("\n\n")

                        if self.device == 'cpu':
                            l2norm = torch.tensor(0., requires_grad = True).cpu()
                        else:
                            l2norm = torch.tensor(0., requires_grad = True).cuda()

                        for w in self.net.parameters():
                            l2norm = l2norm + torch.norm(w)**2
                        
                        total_loss = roll_loss + pitch_loss + self.alpha*l2norm

                        if phase == "train":
                            total_loss.backward()
                            self.optimizer.step()

                        epoch_loss += total_loss.item() * img_list.size(0)


                epoch_loss = epoch_loss/len(self.dataloaders_dict[phase].dataset)
                print("{} Loss: {:.4f}".format(phase, epoch_loss))

                if(epoch%self.save_step == 0 and epoch > 0 and epoch != self.num_epochs and phase == "valid"):
                    self.saveWeight_Interval(epoch)

                if phase == "train":
                    record_train_loss.append(epoch_loss)
                    writer.add_scalar("Loss/Train", epoch_loss, epoch)
                elif phase == "valid":
                    record_valid_loss.append(epoch_loss)
                    writer.add_scalar("Loss/Valid", epoch_loss, epoch)
                elif phase == "distort":
                    record_distort_loss.append(epoch_loss)
                    writer.add_scalar("Loss/Distort", epoch_loss, epoch)

            if record_train_loss and record_valid_loss:
                writer.add_scalars("Loss/train_and_val", {"train": record_train_loss[-1], "val": record_valid_loss[-1]}, epoch)

        mins = (time.time() - start_clock) // 60
        secs = (time.time() - start_clock) % 60
        print("Training Time: ", mins, "[min]", secs, "[sec]")
        
        writer.close()
        self.saveParam()
        self.saveGraph(record_train_loss, record_valid_loss)

    def saveParam(self):
        save_path = self.save_top_path + "/weights.pth"
        torch.save(self.net.state_dict(), save_path)
        print("Saved Weight")

    def saveWeight_Interval(self, epoch):
        save_path = self.save_top_path + "/weights" + "_" + str(epoch) + ".pth"
        torch.save(self.net.state_dict(), save_path)
        print("Saved Weight in Epoch: {}".format(epoch))

    def saveGraph(self, record_loss_train, record_loss_val):
        graph = plt.figure()
        plt.plot(range(len(record_loss_train)), record_loss_train, label="Training")
        plt.plot(range(len(record_loss_val)), record_loss_val, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        #plt.title("loss: train=" + str(record_loss_train[-1]) + ", val=" + str(record_loss_val[-1]))
        graph.savefig(self.save_top_path + "/train_log.jpg")
        plt.show()

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

    def array_to_value_simple_label_in_train(self, label_array):
        max_index = int(np.argmax(label_array))
        plus_index = max_index + 1
        minus_index = max_index - 1
        value = 0.0
        
        for tmp, label in zip(label_array[0], self.value_dict):
            value += tmp * float(label[0])

        if max_index == 0:
            value = -31.0
        elif max_index == 62: #361
            value = 31.0

        # print("value: {}".format(value))
        return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train.py")

    parser.add_argument(
        '--train_cfg', '-c',
        type=str,
        required=False,
        default='../pyyaml/train_config.yaml',
        help='Training configuration file'
    )

    FLAGS, unparsed = parser.parse_known_args()

    print("Load YAML file")

    try:
        print("Opening train config file %s", FLAGS.train_cfg)
        CFG = yaml.safe_load(open(FLAGS.train_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening train config file %s", FLAGS.train_cfg)
        quit()

    save_top_path = CFG["save_top_path"]
    yaml_path = save_top_path + "/train_config.yaml"
    shutil.copy(FLAGS.train_cfg, yaml_path)

    train_type = str(CFG["train_type"])
    train_system = str(CFG["train_system"])
    print("Training type: ", train_type)

    pretrained_weights_top_directory = CFG["pretrained_weights_top_directory"]
    pretrained_weights_file_name = CFG["pretrained_weights_file_name"]
    pretrained_weights_path = os.path.join(pretrained_weights_top_directory, pretrained_weights_file_name)
    
    index_csv_path = str(CFG["index_csv_path"])

    train_sequence = CFG["train"]
    distort_sequence = CFG["distort"]
    valid_sequence = CFG["valid"]
    csv_name = CFG["csv_name"]
    index_csv_path = CFG["index_csv_path"]

    multiGPU = int(CFG["multiGPU"])

    img_size = int(CFG["hyperparameters"]["img_size"])

    do_domain_randomization = str(CFG["hyperparameters"]["transform_params"]["do_domain_randomization"])
    resize = int(CFG["hyperparameters"]["transform_params"]["resize"])
    brightness = float(CFG["hyperparameters"]["transform_params"]["brightness"])
    contrast = float(CFG["hyperparameters"]["transform_params"]["contrast"])
    saturation = float(CFG["hyperparameters"]["transform_params"]["saturation"])
    hue = float(CFG["hyperparameters"]["transform_params"]["hue"])
    kernel_size = int(CFG["hyperparameters"]["transform_params"]["kernel_size"])
    sigma_min = float(CFG["hyperparameters"]["transform_params"]["sigma_min"])
    sigma_max = float(CFG["hyperparameters"]["transform_params"]["sigma_max"])
    equalize_p = float(CFG["hyperparameters"]["transform_params"]["equalize_p"])
    elastic_alpha = float(CFG["hyperparameters"]["transform_params"]["elastic_alpha"])
    distort_epoch = int(CFG["hyperparameters"]["transform_params"]["distort_epoch"])
    
    network_type = str(CFG["hyperparameters"]["network_type"])
    num_classes = int(CFG["hyperparameters"]["num_classes"])
    num_frames = int(CFG["hyperparameters"]["num_frames"])
    deg_threshold = float(CFG["hyperparameters"]["deg_threshold"])
    batch_size = int(CFG["hyperparameters"]["batch_size"])
    num_epochs = int(CFG["hyperparameters"]["num_epochs"])
    optimizer_name = str(CFG["hyperparameters"]["optimizer_name"])
    lr_feature = float(CFG["hyperparameters"]["lr_feature"])
    lr_fc = float(CFG["hyperparameters"]["lr_fc"])
    alpha = float(CFG["hyperparameters"]["alpha"])
    num_workers = int(CFG["hyperparameters"]["num_workers"])
    save_step = int(CFG["hyperparameters"]["save_step"])
    mean_element = float(CFG["hyperparameters"]["mean_element"])
    std_element = float(CFG["hyperparameters"]["std_element"])
    do_white_makeup = bool(CFG["hyperparameters"]["do_white_makeup"])
    do_white_makeup_from_back = bool(CFG["hyperparameters"]["do_white_makeup_from_back"])
    whiteup_frame = int(CFG["hyperparameters"]["whiteup_frame"])

    # TimeSformer params
    patch_size = int(CFG["hyperparameters"]["timesformer"]["patch_size"])
    attention_type = str(CFG["hyperparameters"]["timesformer"]["attention_type"])
    depth = int(CFG["hyperparameters"]["timesformer"]["depth"])
    num_heads = int(CFG["hyperparameters"]["timesformer"]["num_heads"])

    # SENet params
    resnet_model = str(CFG["hyperparameters"]["senet"]["resnet_model"])


    print("Load Train Dataset")

    if train_type == "AirSim":
        train_dataset = dataset_mod_AirSim.AttitudeEstimatorDataset(
            data_list = make_datalist_mod.makeMultiDataList(train_sequence, csv_name),
            transform = data_transform_mod.DataTransform(
                resize,
                mean_element,
                std_element,
                brightness,
                contrast,
                saturation,
                hue,
                kernel_size,
                sigma_min,
                sigma_max,
                equalize_p,
                elastic_alpha,
                do_domain_randomization
            ),
            phase = "train",
            index_dict_path = index_csv_path,
            dim_fc_out = num_classes,
            timesteps = num_frames,
            deg_threshold = deg_threshold,
            resize = resize,
            do_white_makeup = do_white_makeup,
            do_white_makeup_from_back = do_white_makeup_from_back,
            whiteup_frame = whiteup_frame
        )
    elif train_type == "Gimbal":
        train_dataset = dataset_mod_Gimbal.AttitudeEstimatorDataset(
            data_list = make_datalist_mod.makeMultiDataList(train_sequence, csv_name),
            transform = data_transform_mod.DataTransform(
                resize,
                mean_element,
                std_element,
                brightness,
                contrast,
                saturation,
                hue,
                kernel_size,
                sigma_min,
                sigma_max,
                equalize_p,
                elastic_alpha,
                do_domain_randomization
            ),
            phase = "train",
            index_dict_path = index_csv_path,
            dim_fc_out = num_classes,
            timesteps = num_frames,
            deg_threshold = deg_threshold,
            resize = resize,
            do_white_makeup = do_white_makeup,
            do_white_makeup_from_back = do_white_makeup_from_back,
            whiteup_frame = whiteup_frame
        )
    else:
        print("Error: train_type is not defined")
        quit()

    print("Load Distort Dataset")

    if train_type == "AirSim":
        distort_dataset = dataset_mod_AirSim.AttitudeEstimatorDataset(
            data_list = make_datalist_mod.makeMultiDataList(distort_sequence, csv_name),
            transform = data_transform_mod.DataTransform(
                resize,
                mean_element,
                std_element,
                brightness,
                contrast,
                saturation,
                hue,
                kernel_size,
                sigma_min,
                sigma_max,
                equalize_p,
                elastic_alpha,
                "distort"
            ),
            phase = "train",
            index_dict_path = index_csv_path,
            dim_fc_out = num_classes,
            timesteps = num_frames,
            deg_threshold = deg_threshold,
            resize = resize,
            do_white_makeup = do_white_makeup,
            do_white_makeup_from_back = do_white_makeup_from_back,
            whiteup_frame = whiteup_frame
        )
    elif train_type == "Gimbal":
        distort_dataset = dataset_mod_Gimbal.AttitudeEstimatorDataset(
            data_list = make_datalist_mod.makeMultiDataList(distort_sequence, csv_name),
            transform = data_transform_mod.DataTransform(
                resize,
                mean_element,
                std_element,
                brightness,
                contrast,
                saturation,
                hue,
                kernel_size,
                sigma_min,
                sigma_max,
                equalize_p,
                elastic_alpha,
                "distort"
            ),
            phase = "train",
            index_dict_path = index_csv_path,
            dim_fc_out = num_classes,
            timesteps = num_frames,
            deg_threshold = deg_threshold,
            resize = resize,
            do_white_makeup = do_white_makeup,
            do_white_makeup_from_back = do_white_makeup_from_back,
            whiteup_frame = whiteup_frame
        )
    else:
        print("Error: distort_type is not defined")
        quit()
        
    

    print("Load Valid Dataset")

    if train_type == "AirSim":
        valid_dataset = dataset_mod_AirSim.AttitudeEstimatorDataset(
            data_list = make_datalist_mod.makeMultiDataList(valid_sequence, csv_name),
            transform = data_transform_mod.DataTransform(
                resize,
                mean_element,
                std_element,
                brightness,
                contrast,
                saturation,
                hue,
                kernel_size,
                sigma_min,
                sigma_max,
                equalize_p,
                elastic_alpha,
                "eval"
            ),
            phase = "valid",
            index_dict_path = index_csv_path,
            dim_fc_out = num_classes,
            timesteps = num_frames,
            deg_threshold = deg_threshold,
            resize = resize,
            do_white_makeup = do_white_makeup,
            do_white_makeup_from_back = do_white_makeup_from_back,
            whiteup_frame = whiteup_frame
        )
    elif train_type == "Gimbal":
        valid_dataset = dataset_mod_Gimbal.AttitudeEstimatorDataset(
            data_list = make_datalist_mod.makeMultiDataList(valid_sequence, csv_name),
            transform = data_transform_mod.DataTransform(
                resize,
                mean_element,
                std_element,
                brightness,
                contrast,
                saturation,
                hue,
                kernel_size,
                sigma_min,
                sigma_max,
                equalize_p,
                elastic_alpha,
                "eval"
            ),
            phase = "valid",
            index_dict_path = index_csv_path,
            dim_fc_out = num_classes,
            timesteps = num_frames,
            deg_threshold = deg_threshold,
            resize = resize,
            do_white_makeup = do_white_makeup,
            do_white_makeup_from_back = do_white_makeup_from_back,
            whiteup_frame = whiteup_frame
        )
    else:
        print("Error: train_type is not defined")
        quit()

    print("Load Network")
    if network_type == "TimeSformer":
        net = vit.TimeSformer(img_size, patch_size, num_classes, num_frames, depth, num_heads, attention_type, pretrained_weights_path, train_system)
    elif network_type == "SENet":
        net = senet.SENet(model=resnet_model, dim_fc_out=num_classes, norm_layer=nn.BatchNorm2d, pretrained_model=pretrained_weights_path, time_step=num_frames, use_SELayer=True)
    else:
        print("Error: Network type is not defined")
        quit()

    print(net)

    trainer = Trainer(
        save_top_path,
        pretrained_weights_path,
        index_csv_path,
        multiGPU,
        img_size,
        mean_element,
        std_element,
        num_classes,
        deg_threshold,
        batch_size,
        num_epochs,
        optimizer_name,
        lr_feature,
        lr_fc,
        alpha,
        num_frames,
        patch_size,
        net,
        train_dataset,
        distort_dataset,
        valid_dataset,
        num_workers,
        save_step,
        distort_epoch,
        train_system
    )

    trainer.process()