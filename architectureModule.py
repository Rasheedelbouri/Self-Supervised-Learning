# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:07:49 2021

@author: rashe
"""
from torch import nn
import torch.nn.functional as F
import tensorflow as tf

class customNet(nn.Module):
    def __init__(self):
        super(customNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 224, 3)
        self.zpad = nn.ZeroPad2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(224, 112, 3)
        self.batchnorm = nn.BatchNorm2d(112)
        self.conv3 = nn.Conv2d(112, 66, 3)
        self.conv4 = nn.Conv2d(66, 30, 3)
        self.fc1 = nn.Linear(417720, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 100)
        self.fc5 = nn.Linear(100, 10)

        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.zpad(x)
        x = F.relu(self.conv2(x))
        x = self.zpad(x)
        x = self.pool(x)
        x = self.batchnorm(x)
        x = self.dropout(F.relu(self.conv3(x)))
        x = self.dropout(F.relu(self.conv4(x)))
        x = x.view(x.size(0),-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)
        
        return x

# =============================================================================
#         x = self.pool(self.zpad(F.relu(self.conv1(x))))
#         x = self.pool(self.dropout(F.relu(self.conv2(x))))
#         x = self.pool(self.dropout(F.relu(self.conv3(x))))
#         x = self.pool(self.dropout(F.relu(self.conv4(x))))
#         x = x.view(x.size(0),-1)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.dropout(F.relu(self.fc2(x)))
#         x = self.fc3(x)
# =============================================================================
    
# =============================================================================
# feature_extractor_model = tf.keras.applications.ResNet50V2(include_top=True, weights='imagenet')    
# last_conv_layer_name = 'post_relu'
# last_non_classification_layer = 'avg_pool'
# =============================================================================
