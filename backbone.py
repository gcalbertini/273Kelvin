import sys
import torch
import torchvision.models as models
import torch.nn as nn

from lightning import train_backbone

class Backbone(nn.Module):
    def __init__(self, freeze, backbone):
        super().__init__()
        self.out_channels = 512
        self.premodel = backbone
        self.freeze = freeze

        if self.freeze:
            for p in self.premodel.parameters():
                p.requires_grad = False

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

    return Backbone(args.freeze, backbone)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()