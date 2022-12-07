import torch
import torchvision.models as models
import torch.nn as nn

from lightning import train_backbone
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class Backbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.out_channels = 256
        self.premodel = backbone

        # change this later
        for p in self.premodel.parameters():
            p.requires_grad = True

    def forward(self,x):
        #print(x.size())
        out = self.premodel(x)
        #print(out.size())
        return out

def get_backbone(train=False):

    if train:
        train_backbone()

    #resnet = models.resnet18(pretrained=None)
    #resnet.fc = nn.Identity()
    checkpoint = torch.load('./resnet18_backbone_weights_final.ckpt')
    #resnet.load_state_dict(checkpoint['model_state_dict'])
    #req_layers = list(resnet.children())[:8]
    #backbone = nn.Sequential(*req_layers)

    #return Backbone(backbone)
    #return Backbone(resnet_fpn_backbone('resnet18', weights=checkpoint, trainable_layers=5))
    return resnet_fpn_backbone('resnet18', weights=checkpoint, trainable_layers=5)
