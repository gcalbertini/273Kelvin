import warnings
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from backbone import get_backbone

def get_model(backbone=None, num_classes=100):

    if backbone is None:
        warnings.warn("Using pretrained mobilenet_v2 backbone instead of simclr, deafult weights")
        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        backbone.out_channels = 1280
        output_size = 7

    else:
        backbone = get_backbone(train=False)
        backbone.out_channels = 2048
        output_size = 1

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))  # (1:2, 1:1, 2:1); 3 AR + 3 scales scales shown to work best in Faster RCNN paper wrt mAP; probably keep same


    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=output_size,  # TODO change this as *ResNet*-based RCNN adopts RoI pooling to extract 7×7 features, with prev backbone was 7
                                                    sampling_ratio=2,
                                                    canonical_level=4,
                                                    canonical_scale=224)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    

    return model
    

    




