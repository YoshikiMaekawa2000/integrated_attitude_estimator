import sys
sys.dont_write_bytecode = True

from models import feature_extractor_high
from models import feature_extractor_low
from models import classification_fc_layer

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as nn_functional

class SENet(nn.Module):
    def __init__(self, model, dim_fc_out, norm_layer, pretrained_model, time_step, use_SELayer=True):
        super(SENet, self).__init__()
        self.dim_fc_out = dim_fc_out
        if model == "resnet18":
            print("Load ResNet18")
            self.feature_extractor = feature_extractor_low.resnet18(pretrained_model, time_step, use_SELayer, norm_layer=norm_layer, bn_eps=1e-5, bn_momentum=0.1, deep_stem=True, stem_width=64)
            self.fully_connected = classification_fc_layer.ClassificationType("low", dim_fc_out, 0.1)
        elif model == "resnet34":
            print("Load ResNet34")
            self.feature_extractor = feature_extractor_low.resnet34(pretrained_model, time_step, use_SELayer, norm_layer=norm_layer, bn_eps=1e-5, bn_momentum=0.1, deep_stem=True, stem_width=64)
            self.fully_connected = classification_fc_layer.ClassificationType("low", dim_fc_out, 0.1)
        elif model == "resnet50":
            print("Load ResNet50")
            self.feature_extractor = feature_extractor_high.resnet50(pretrained_model, time_step, use_SELayer, norm_layer=norm_layer, bn_eps=1e-5, bn_momentum=0.1, deep_stem=True, stem_width=64)
            print("Load Classification Layer")
            self.fully_connected = classification_fc_layer.ClassificationType("high", dim_fc_out, 0.1)        
        elif model == "resnet101":
            print("Load ResNet101")
            self.feature_extractor = feature_extractor_high.resnet101(pretrained_model, time_step, use_SELayer, norm_layer=norm_layer, bn_eps=1e-5, bn_momentum=0.1, deep_stem=True, stem_width=64)
            print("Load Classification Layer")
            self.fully_connected = classification_fc_layer.ClassificationType("high", dim_fc_out, 0.1)        
        elif model == "resnet152":
            print("Load ResNet152")
            self.feature_extractor = feature_extractor_high.resnet152(pretrained_model, time_step, use_SELayer, norm_layer=norm_layer, bn_eps=1e-5, bn_momentum=0.1, deep_stem=True, stem_width=64)
            print("Load Classification Layer")
            self.fully_connected = classification_fc_layer.ClassificationType("high", dim_fc_out, 0.1)
    
    def forward(self, x):
        x = self.feature_extractor(x)

        #print(x.size())
        fc_input = torch.flatten(x, 1)
        #print(fc_input.size())
        #fc_input = x

        roll = self.fully_connected.roll_fc(fc_input)
        pitch = self.fully_connected.pitch_fc(fc_input)

        return roll, pitch

    def getParamValueList(self):
        list_feature_extractor_param_value = []
        list_roll_fc_param_value = []
        list_pitch_fc_param_value = []

        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "feature_extractor" in param_name:
                list_feature_extractor_param_value.append(param_value)
            if "roll_fc" in param_name:
                list_roll_fc_param_value.append(param_value)
            if "pitch_fc" in param_name:
                list_pitch_fc_param_value.append(param_value)

        return list_feature_extractor_param_value, list_roll_fc_param_value, list_pitch_fc_param_value