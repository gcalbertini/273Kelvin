import torch
import torchvision.models as models
import torch.nn as nn

#from lightning import train_backbone
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

def get_backbone(train=False):

    #if train:
        #train_backbone()

    resnet = models.resnet18(weights=None)
    resnet.maxpool = nn.Identity()
    resnet.fc = nn.Identity()
    checkpoint = torch.load('./resnet18_backbone_weights_final.ckpt')
    resnet.load_state_dict(checkpoint['model_state_dict'])

    returned_layers=[1, 2, 3, 4]
    in_channels_stage2 = resnet.inplanes // 8 # resnet.inplanes=512
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
    extra_blocks = LastLevelMaxPool()

    return BackboneWithFPN(backbone=resnet, in_channels_list=in_channels_list, return_layers=return_layers, out_channels=resnet.inplanes, extra_blocks=extra_blocks)