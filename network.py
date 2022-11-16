# import shutil, time, os, requests, random, copy

import torch
import torch.nn as nn
#import torch.optim as optim
from torchvision import models #datasets, transforms, model

# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()

#     def forward(self, x):
#         return x

# class LinearLayer(nn.Module):
#     def __init__(self, in_features, out_features, use_bias = True, **kwargs):
#         super(LinearLayer, self).__init__(**kwargs)

#         self.in_features = in_features
#         self.out_features = out_features
#         self.use_bias = use_bias
        
#         self.linear = nn.Linear(self.in_features, 
#                                 self.out_features, 
#                                 bias = self.use_bias and not self.use_bn)
        
#         self.bn = nn.BatchNorm1d(self.out_features)

#     def forward(self,x):
#         x = self.linear(x)
#         x = self.bn(x)
#         return x

class ProjectionHead(nn.Module):
    #Linear, Batch Norm, ReLU, Linear, Batch Norm
    def __init__(self,in_features,hidden_features,out_features,**kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.layers = nn.Sequential(
            nn.Linear(self.in_features,self.hidden_features,True),
            nn.BatchNorm1d(self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features,self.out_features,False),
            nn.BatchNorm1d(self.out_features))
        
    def forward(self,x):
        x = self.layers(x)
        return x

class PreModel(nn.Module):
    def __init__(self,base_model):
        super().__init__()
        self.base_model = base_model
        
        #Change pretrained model
        self.pretrained = models.resnet50(pretrained=False)
        # self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        # self.pretrained.maxpool = Identity()
        # self.pretrained.fc = Identity()
        
        for p in self.pretrained.parameters():
            p.requires_grad = True
        
        self.projector = ProjectionHead(2048, 2048, 128)

    def forward(self,x):
        out = self.pretrained(x)
        
        x_projection = self.projector(torch.squeeze(out))
        
        return x_projection

