import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from backbone import get_backbone

def get_model(args, backbone=None, num_classes=100):

    if backbone is None:
        print("!!!! Using pretrained mobilenet_v2 backbone instead of SimCLR, default weights !!!!")
        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        backbone.out_channels = 1280

    else:
        backbone = get_backbone(args, train=args.train_backbone)
        #out_channels already set

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,)) 
    anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                    aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes))  # (1:2, 1:1, 2:1); 3 AR + 3 scales scales shown to work best in Faster RCNN paper wrt mAP; probably keep samem

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0','1','2','3'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone=backbone,
                        num_classes=num_classes,
                        rpn_anchor_generator=anchor_generator,
                        box_roi_pool=roi_pooler)
    

    return model