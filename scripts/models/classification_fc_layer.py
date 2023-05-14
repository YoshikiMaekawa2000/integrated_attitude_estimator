import functools
import torch
import torch.nn as nn
import torch.nn.functional as nn_functional

class ClassificationType(nn.Module):
    def __init__(self, type, dim_fc_out, dropout_rate):
        super(ClassificationType, self).__init__()
        #self.dim_fc_in = 100
        if type=="high":
            self.dim_fc_in = 100352
        elif type=="low":
            self.dim_fc_in = 25088
        
        self.dim_fc_out = dim_fc_out
        self.dropout_rate = dropout_rate

        self.roll_fc = nn.Sequential(
            nn.Linear(self.dim_fc_in, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 150, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 100, self.dim_fc_out),
            nn.Softmax(dim=1)
        )

        self.pitch_fc = nn.Sequential(
            nn.Linear(self.dim_fc_in, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 150, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 100, self.dim_fc_out),
            nn.Softmax(dim=1)
        )

        self.initializeWeights()#no need?
    
    def initializeWeights(self):
        for m in self.roll_fc.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

        for m in self.pitch_fc.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)