# import shutil, time, os, requests, random, copy

import torch
import torch.nn as nn
from torchvision import models 

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

        self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.pretrained.maxpool = nn.Identity()
        self.pretrained.fc = nn.Identity()
        self.projector = ProjectionHead(2048, 2048, 128)

    def forward(self,x):
        out = self.pretrained(x)
        x_projection = self.projector(torch.squeeze(out))
        return x_projection

model_SimCLR = PreModel('resnet50').to('cuda:0')

class DSModel(nn.Module):
    def __init__(self,premodel,num_classes):
        super().__init__()
        
        self.premodel = premodel
        self.num_classes = num_classes
        
        for p in self.premodel.parameters():
            p.requires_grad = False
            
        for p in self.premodel.projector.parameters():
            p.requires_grad = False
        
        self.lastlayer = nn.Linear(2048,self.num_classes)
        
    def forward(self,x):
        out = self.premodel.pretrained(x)
        out = self.lastlayer(out)
        return out

model_final = DSModel(model_SimCLR, 100)


