# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:07:49 2021

@author: rashe
"""
import torch
from torch import nn
import torch.nn.functional as F
import tensorflow as tf
from torchvision import models


class customNet(nn.Module):
    def __init__(self):
        super(customNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.zpad = nn.ZeroPad2d(8)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 20, 3)
        self.batchnorm = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 15, 3)
        self.conv4 = nn.Conv2d(15, 10, 3)
        self.fc1 = nn.Linear(90, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(250, 100)
        self.fc5 = nn.Linear(100, 10)

        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        
# =============================================================================
#         x = F.relu(self.conv1(x))
#         x = self.zpad(x)
#         x = F.relu(self.conv2(x))
#         x = self.zpad(x)
#         x = self.pool(x)
#         x = self.batchnorm(x)
#         x = self.dropout(F.relu(self.conv3(x)))
#         x = self.dropout(F.relu(self.conv4(x)))
#         x = x.view(x.size(0),-1)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.dropout(F.relu(self.fc2(x)))
#         x = self.dropout(F.relu(self.fc3(x)))
#         x = self.dropout(F.relu(self.fc4(x)))
#         x = self.fc5(x)
#         
#         return x
# =============================================================================

        x = self.pool(self.zpad(F.relu(self.conv1(x))))
        x = self.pool(self.dropout(F.relu(self.conv2(x))))
        x = self.pool(self.dropout(F.relu(self.conv3(x))))
        x = self.pool(self.dropout(F.relu(self.conv4(x))))
        x = x.view(x.size(0),-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
    
        return x
    

class loadedPreTrained(nn.Module):
    
    def __init__(self, model, finetune=True):
        super(loadedPreTrained, self).__init__()
        
        assert isinstance(finetune, bool)
        assert isinstance(model, str)
        
        
        self.finetune = finetune
        
        if model.lower() == "resnet":
            model = models.resnet18
        elif model.lower() == "vgg":
            model = models.vgg16
        elif model.lower() == "densenet":
            model = models.densenet161
            
        
        self.model_ft = model(pretrained=True)
        
        
        if self.finetune:
            for param in self.model_ft.parameters():
                param.requires_grad = False
                
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 5)
        self.model_ft = self.model_ft.to("cpu")
        
        
        
    def forward(self, x):
        y = self.model_ft(x)
        #x = self.lin(self.resnet(x))

        return y



        
#model = tf.keras.applications.ResNet50V2(include_top=True, weights='imagenet')    
#x = Dense()(model.layers[-1].output)
#o = Activation(None, name='loss')(x)