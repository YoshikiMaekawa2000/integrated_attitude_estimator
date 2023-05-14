import sys, codecs
from tkinter import W
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout)
sys.dont_write_bytecode = True
#
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

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as nn_functional
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

import sys
sys.path.append('../')
from common import network_mod
from common import make_datalist_mod
from common import data_transform_mod
from common import dataset_mod
from common import make_datalist_mod

class Trainer:
    def __init__(self,
        save_top_path,
        multiGPU,
        img_size,
        mean_element,
        std_element,
        dropout_rate,
        num_classes,
        deg_threshold,
        batch_size,
        num_epochs,
        optimizer_name,
        lr,
        alpha,
        net,
        train_dataset,
        valid_dataset,
        num_workers,
        save_step):
        super(Trainer, self).__init__()
        self.save_top_path = save_top_path
        self.multiGPU = multiGPU
        self.img_size = img_size
        self.mean_element = mean_element
        self.std_element = std_element
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.deg_threshold = deg_threshold
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.alpha = alpha
        self.net = net
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.num_workers = num_workers
        self.save_step = save_step

        if self.multiGPU == 0:
                self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.setRandomCondition()
        self.dataloaders_dict = self.getDataloaders(self.train_dataset, self.valid_dataset, batch_size)
        self.net = self.getNetwork(net)
        self.optimizer = self.getOptimizer(self.optimizer_name, self.lr)

    def setRandomCondition(self, keep_reproducibility=False, seed=123456789):
        if keep_reproducibility:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def getDataloaders(self, train_dataset, valid_dataset, batch_size):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle=False,
            num_workers = self.num_workers,
            #pin_memory =True
        )

        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size = batch_size,
            shuffle=False,
            num_workers = self.num_workers,
            #pin_memory = True
        )

        dataloaders_dict = {"train":train_dataloader, "valid":valid_dataloader}

        return dataloaders_dict

    def getOptimizer(self, optimizer_name, lr):

        if optimizer_name == "SGD":
            optimizer = optim.SGD(self.net.parameters() ,lr = lr, momentum=0.9, 
            weight_decay=0.0)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam(self.net.parameters(), lr = lr, weight_decay=0.0)
        elif optimizer_name == "RAdam":
            optimizer = optim.RAdam(self.net.parameters(), lr = lr, weight_decay=0.0)

        print("optimizer: {}".format(optimizer_name))
        return optimizer


    def getNetwork(self, net):
        print("Loading Network")
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
        record_valid_loss = []

        for epoch in range(self.num_epochs):
            print("--------------------------------")
            print("Epoch: {}/{}".format(epoch+1, self.num_epochs))

            for phase in ["train", "valid"]:
                if phase == "train":
                    self.net.train()
                elif phase == "valid":
                    self.net.eval()
                
                #Data Load
                epoch_loss = 0.0

                for img, label_roll, label_pitch in tqdm(self.dataloaders_dict[phase]):
                    self.optimizer.zero_grad()

                    #print(img_list.size())

                    #img_list = torch.FloatTensor(1, 3, 8, 224, 224)

                    img = img.to(self.device)

                    label_roll = label_roll.to(self.device)
                    label_pitch = label_pitch.to(self.device)

                    #Reset Gradient
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=="train"):
                        roll_inf, pitch_inf = self.net(img)

                        #print(roll_inf, pitch_inf)

                        logged_roll_inf = nn_functional.log_softmax(roll_inf, dim=1)
                        logged_pitch_inf = nn_functional.log_softmax(pitch_inf, dim=1)

                        roll_loss = torch.mean(torch.sum(-label_roll*logged_roll_inf, 1))
                        pitch_loss = torch.mean(torch.sum(-label_pitch*logged_pitch_inf, 1))

                        torch.set_printoptions(edgeitems=1000000)

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

                        epoch_loss += total_loss.item() * img.size(0)

                epoch_loss = epoch_loss/len(self.dataloaders_dict[phase].dataset)
                print("{} Loss: {:.4f}".format(phase, epoch_loss))

                if(epoch%self.save_step == 0 and epoch > 0 and epoch != self.num_epochs and phase == "valid"):
                    self.saveWeight_Interval(epoch)

                if phase == "train":
                    record_train_loss.append(epoch_loss)
                    writer.add_scalar("Loss/Train", epoch_loss, epoch)
                else:
                    record_valid_loss.append(epoch_loss)
                    writer.add_scalar("Loss/Valid", epoch_loss, epoch)

            if record_train_loss and record_valid_loss:
                writer.add_scalars("Loss/train_and_val", {"train": record_train_loss[-1], "val": record_valid_loss[-1]}, epoch)

        mins = (time.time() - start_clock) // 60
        secs = (time.time() - start_clock) % 60
        print("Training Time: ", mins, "[min]", secs, "[sec]")
        
        writer.close()
        self.saveParam()
        self.saveGraph(record_train_loss, record_valid_loss)

    def saveParam(self):
        save_path = self.save_top_path + "/weights_SII2022.pth"
        torch.save(self.net.state_dict(), save_path)
        print("Saved Weight")

    def saveWeight_Interval(self, epoch):
        save_path = self.save_top_path + "/weights_SII2022" + "_" + str(epoch) + ".pth"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train.py")

    parser.add_argument(
        '--train_cfg', '-c',
        type=str,
        required=False,
        default='../pyyaml/train_config_SII2022.yaml',
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
    save_yaml_name = CFG["save_yaml_name"]
    yaml_path = save_top_path + "/" + save_yaml_name
    shutil.copy(FLAGS.train_cfg, yaml_path)

    train_sequence = CFG["train"]
    valid_sequence = CFG["valid"]
    csv_name = CFG["csv_name"]
    index_csv_path = CFG["index_csv_path"]

    multiGPU = int(CFG["multiGPU"])

    img_size = int(CFG["hyperparameters"]["img_size"])
    resize = int(CFG["hyperparameters"]["resize"])
    num_classes = int(CFG["hyperparameters"]["num_classes"])
    deg_threshold = float(CFG["hyperparameters"]["deg_threshold"])
    batch_size = int(CFG["hyperparameters"]["batch_size"])
    num_epochs = int(CFG["hyperparameters"]["num_epochs"])
    optimizer_name = str(CFG["hyperparameters"]["optimizer_name"])
    lr = float(CFG["hyperparameters"]["lr"])
    alpha = float(CFG["hyperparameters"]["alpha"])
    num_workers = int(CFG["hyperparameters"]["num_workers"])
    save_step = int(CFG["hyperparameters"]["save_step"])
    mean_element = float(CFG["hyperparameters"]["mean_element"])
    std_element = float(CFG["hyperparameters"]["std_element"])
    dropout_rate = float(CFG["hyperparameters"]["dropout_rate"])
    do_white_makeup = bool(CFG["hyperparameters"]["do_white_makeup"])
    do_white_makeup_from_back = bool(CFG["hyperparameters"]["do_white_makeup_from_back"])
    whiteup_frame = int(CFG["hyperparameters"]["whiteup_frame"])


    print("Load Train Sequence\n")
    train_dataset = dataset_mod.ClassOriginalDataset(
        data_list = make_datalist_mod.makeMultiDataList(train_sequence, csv_name),
        transform = data_transform_mod.DataTransform(
            resize,
            mean_element,
            std_element,
        ),
        phase = "train",
        index_dict_path = index_csv_path,
        dim_fc_out = num_classes,
        deg_threshold = deg_threshold,
    )

    print("Load Valid Sequence\n")
    valid_dataset = dataset_mod.ClassOriginalDataset(
        data_list = make_datalist_mod.makeMultiDataList(valid_sequence, csv_name),
        transform = data_transform_mod.DataTransform(
            resize,
            mean_element,
            std_element,
        ),
        phase = "valid",
        index_dict_path = index_csv_path,
        dim_fc_out = num_classes,
        deg_threshold = deg_threshold,
    )

    print("Load Network")
    net = network_mod.Network(resize, num_classes, dropout_rate)
    print(net)

    trainer = Trainer(
        save_top_path,
        multiGPU,
        img_size,
        mean_element,
        std_element,
        dropout_rate,
        num_classes,
        deg_threshold,
        batch_size,
        num_epochs,
        optimizer_name,
        lr,
        alpha,
        net,
        train_dataset,
        valid_dataset,
        num_workers,
        save_step
    )

    trainer.process()