import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import rasterio
import yaml
import pickle
import argparse
from typing import Optional
from functools import partial
from lightning import LightningModule

from terratorch.models.model import ModelOutput

#simple decoder to reduce dimensionality of prithvi enc output and flatten to 64D
class SimpleDecoder_comb_v2(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=64):
        super(SimpleDecoder_comb_v2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)# 1024 to 256; shape 10x1024 to 10x256
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        #self.drp = nn.Dropout(p=drp_rate)
        self.hidden_dim_flattened=10*hidden_dim#10 is feature dim+ class token in MAE; 10x256 to 2560
        self.fc2=nn.Linear(self.hidden_dim_flattened, output_dim)# 2560 to 64
        #self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        #self.gelu = nn.GELU()

    def forward(self, x):
        x = self.relu(self.fc1(x))#shape 10x1024 to 10x256 ORG
        x = torch.reshape(x,(x.shape[0], x.shape[1]*x.shape[2]))#10x256 to 2560 
        x = self.fc2(x)  # 2560 to 64 Output shape 
        return x

# Define the convolutional layers for the point1d MERRA input 
class Pt1dConvBranch(nn.Module):
    def __init__(self):
        super(Pt1dConvBranch, self).__init__()
        self.conv1 = nn.Conv2d(10, 32, kernel_size=1)
        #self.bn1 = nn.BatchNorm2d(32)
        #self.drp = nn.Dropout(p=drp_rate)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=1)
        #self.bn2 = nn.BatchNorm2d(16)
        #self.drp = nn.Dropout(p=drp_rate)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=1)
        self.fc = nn.Linear(8, 64)  # Final output matches decoder output

    def forward(self, x):
        x = torch.relu(self.conv1(x)) #ORIGINAL merra [batch_size, 10, 1, 1] to [batch_size, 32, 1, 1]
        x = torch.relu(self.conv2(x))## Output shape [batch_size, 16, 1, 1]
        x = torch.relu(self.conv3(x))#Output shape [batch_size, 8, 1, 1]
        x=torch.reshape(x, (x.shape[0], x.shape[1]))#output reshape [batch_size, 8]
        x = self.fc(x) # Output shape [batch_size, 64]
        return x

# Define the regression model --simple regression to concatenate prithvi merra and regress to gpp lfux
class RegressionModel_flux(LightningModule):
    def __init__(self, prithvi_model):
        super(RegressionModel_flux, self).__init__()
        self.prithvi_model = prithvi_model
        self.decoder = SimpleDecoder_comb_v2(input_dim=1024, hidden_dim=256, output_dim=64)
        self.pt1d_conv_branch = Pt1dConvBranch()
        self.fc_final = nn.Linear(128, 1)  # Regression output
        #self.fc_final2 = nn.Linear(64, 1)  # Regression output

    def forward(self, im2d, pt1d, **kwargs):
        # Pass HLS im2d through the pretrained prithvi MAE encoder (with frozen weights)
        #pri_enc = self.prithvi_model(im2d, temporal_coords=None, location_coords=None)#.output#batch x 6x1x1x50; none, none for loc, temporal, 0--mask; output: batch x 10 x 1024
        pri_enc = self.prithvi_model(im2d, None, None, 0)#batch x 6x1x1x50; none, none for loc, temporal, 0--mask; output: batch x 10 x 1024

        # Pass pri_enc through the simple decoder
        dec_out = self.decoder(pri_enc)  # op Shape [batch_size, 64]
        # Pass MERRA pt1d through the convolutional layers
        pt1d_out = self.pt1d_conv_branch(pt1d)  # Shape [batch_size, 64]
        # Concatenate decoder output and pt1d output
        combined = torch.cat((dec_out[:, :], pt1d_out), dim=1) # op: [batch x 128]
        # Final regression output
        output1 = self.fc_final(combined)  # Shape [batch_size, 1]
        #output2 = self.fc_final2(output1)  # Shape [batch_size, 1]
        output = ModelOutput(output=output1)
        
        return output
