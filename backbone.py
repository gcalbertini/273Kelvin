import sys
import torch
import torchvision.models as models
import torch.nn as nn

#from lightning import train_backbone
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

def get_backbone(args, train):

    #if train:
        #train_backbone()

    resnet = models.resnet18(weights=None)
    resnet.maxpool = nn.Identity()
    resnet.fc = nn.Identity()
    checkpoint = torch.load('./resnet18_backbone_weights_final.ckpt')
    resnet.load_state_dict(checkpoint['model_state_dict'])

    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1", "bn1"]
    #layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1", "bn1"]
    #layers_to_train = ["layer4", "layer3", "layer2"]
    for name, parameter in resnet.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    returned_layers=[1, 2, 3, 4]
    in_channels_stage2 = resnet.inplanes // 8 # resnet.inplanes=512
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
    extra_blocks = LastLevelMaxPool()

    return BackboneWithFPN(backbone=resnet, in_channels_list=in_channels_list, return_layers=return_layers, out_channels=256, extra_blocks=extra_blocks)
