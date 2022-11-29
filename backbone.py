import torch
import torchvision.models as models
import torch.nn as nn

from lightning import train_backbone

class Backbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.out_channels = 512
        self.premodel = backbone

        # change this later
        for p in self.premodel.parameters():
            p.requires_grad = False

    def forward(self,x):
        out = self.premodel(x)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        return out

def get_backbone(train=False):

    if train:
        train_backbone()

    backbone = models.resnet18(pretrained=None)
    backbone.fc = nn.Identity()
    checkpoint = torch.load('./resnet18_backbone_weights.ckpt')
    backbone.load_state_dict(checkpoint['model_state_dict'])

    return Backbone(backbone)