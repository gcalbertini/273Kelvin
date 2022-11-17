# import shutil, time, os, requests, random, copy

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from torchvision import models #datasets, transforms, model

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
    def __init__(self,base_model,labeled):
        super().__init__()
        self.base_model = base_model
        self.labeled = labeled
        #Add the pretrained model
        self.pretrained = models.resnet50(pretrained=False)
        for p in self.pretrained.parameters():
                p.requires_grad = True

        if labeled:
            layers = list(self.pretrained.children())[:8]
            self.features1 = nn.Sequential(*layers[:6])
            self.features2 = nn.Sequential(*layers[6:])
            self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
            self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        else:  
            self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
            self.pretrained.maxpool = nn.Identity()
            self.pretrained.fc = nn.Identity()
            self.projector = ProjectionHead(2048, 2048, 128)

    def forward(self,x):
        out = self.pretrained(x)
        if self.labeled:
            x = self.features1(x)
            x = self.features2(x)
            x = F.relu(x)
            x = nn.AdaptiveAvgPool2d((1,1))(x)
            x = x.view(x.shape[0], -1)
            classes = self.classifier(x)
            bbox = self.bb(x)
            return classes, bbox
        else:
            x_projection = self.projector(torch.squeeze(out))
            return x_projection

# class BB_model(nn.Module):
#     def __init__(self):
#         super(BB_model, self).__init__()
#         resnet = models.resnet34(pretrained=True)
#         layers = list(resnet.children())[:8]
#         self.features1 = nn.Sequential(*layers[:6])
#         self.features2 = nn.Sequential(*layers[6:])
#         self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
#         self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        
#     def forward(self, x):
#         x = self.features1(x)
#         x = self.features2(x)
#         x = F.relu(x)
#         x = nn.AdaptiveAvgPool2d((1,1))(x)
#         x = x.view(x.shape[0], -1)
#         return self.classifier(x), self.bb(x)

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

