import sys
import torch
import torchvision.models as models
import torch.nn as nn

from lightning import train_backbone

class Backbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.out_channels = 512
        self.premodel = backbone

    def forward(self,x):
        out = self.premodel(x)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        return out

def get_backbone(args, train):

    if train:
        print('Pretraining backbone...')
        train_backbone(args)

    backbone = models.__dict__[args.arch](weights=None)
    backbone.fc = nn.Identity()

    # Assuming resnet18 as default
    try:
        path = './' + str(args.arch)+'_backbone_weights.ckpt'
        checkpoint = torch.load(path)
        backbone.load_state_dict(checkpoint['model_state_dict'])
    except:
        print('You need to have a trained backbone first!!')
        sys.exit()

    return Backbone(backbone)

