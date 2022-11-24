import warnings
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from eval import evaluate
from labeled_dataloaders import *
from utils import *



class tuned_FasterRCNN():

    def __init__(self):
        self.backbone = None
        self.out_channels = None
        # using "canonical ImageNet pre-training size" RoI width x height = 224x224; same as ours so ok to use
        self.anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                                aspect_ratios=((0.5, 1.0, 2.0),))  # (1:2, 1:1, 2:1); 3 AR + 3 scales scales shown to work best in Faster RCNN paper wrt mAP; probably keep same
        self.training = True
        self.roi_pooler = None
        self.num_classes = 100

        # load a pre-trained model for classification and return
        # only the features

    def get_model(self, backbone=None, out_channels=None, pretrained=False):
        # FasterRCNN needs to know the number of
        # output channels in a backbone. For mobilenet_v2, it's 1280
        if backbone is None or out_channels is None:
            warnings.warn("Using mobilenet_v2 backbone by default")
            self.backbone = torchvision.models.mobilenet_v2(
                weights="DEFAULT").features
            self.backbone.out_channels = 1280
        else:
            # TODO This will ultimately be SimCLR().features and # of out chan modified
            pass

        '''
		NOTE: Params below configured using ResNet-based Faster RCNN:
		 FPN paper (Section 4): https://arxiv.org/pdf/1612.03144.pdf
		 ResNet (Table 1): https://arxiv.org/pdf/1512.03385.pdf
		 Faster RCNN: https://arxiv.org/pdf/1506.01497.pdf

		 Helpful links:
		 What is RoI align: https://towardsdatascience.com/understanding-region-of-interest-part-2-roi-align-and-roi-warp-f795196fc193
		 Changing canonical values: https://github.com/pytorch/vision/issues/3129
		 FPN Review: https://towardsdatascience.com/review-fpn-feature-pyramid-network-object-detection-262fc7482610


		Canonical scale and canonical level:
		Here, 224 is the canonical ImageNet pre-training size, and k0 is the target level on which an RoI with w-by-h = 224x224 should be mapped into. [...] we set k0 to 4
		as *RESNET-BASED* Faster R-CNN uses C4 as the single-scale feature map; if the RoI scale becomes smaller (say, 1/2 of 224),
		it should be mapped into a finer-resolution level (say, k = 3):
		k = ⌊k0 + log2(√wh/224)⌋

		Sampling ratio:
		is number of sampling points in the interpolation grid used to compute the output value of each pooled output bin.
		If > 0, then exactly sampling_ratio x sampling_ratio sampling points per bin are used for RoI alignment. I'd say keep to 2x2.

		Featmap_names:
		if your backbone returns a Tensor, featmap_names is expected to
		be [0]. More generally, the backbone should return an
		OrderedDict[Tensor], and in featmap_names you can choose which
		feature maps to use.

		'''
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                             output_size=7,  # TODO change this as *ResNet*-based RCNN adopts RoI pooling to extract 7×7 features
                                                             sampling_ratio=2,
                                                             canonical_level=4,
                                                             canonical_scale=224)

        # put the pieces together inside a FasterRCNN model
        model = FasterRCNN(backbone=self.backbone,
                           num_classes=self.num_classes,
                           rpn_anchor_generator=self.anchor_generator,
                           box_roi_pool=self.roi_pooler)
        return model

    def train_tuned(self, model, VALID_DATASET_PATH, EPOCHS=1, LR=0.001, MOM=0.9, DECAY=0.0005, BATCH_SIZE=4, NUM_WORKERS=2, SHUFFLE=False, print_freq=50, verbose=True):

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        train_dataset = LabeledDataset(
            root=VALID_DATASET_PATH,
            split="training",
            transforms=transform  # albumentations transformation
            )

        valid_dataset = LabeledDataset(
            root=VALID_DATASET_PATH,
            split="validation",
            transforms=transform  # albumentations transformation
            )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
        )

        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=LR, momentum=MOM, weight_decay=DECAY)

        if verbose:
            print("Model's state_dict:")
            for param_tensor in model.state_dict():
                print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            print("Optimizer's state_dict:")
            for var_name in optimizer.state_dict():
                print(var_name, "\t", optimizer.state_dict()[var_name])
            #TODO defintely change this or where this be
            print('Model Summary:')
            print(model)
        else:
            print_freq = 0

        # Use a learning rate scheduler: this means that we will decay the learning rate every <step_size> epoch by a factor of <gamma>
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)

        for epoch in range(EPOCHS):
            train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq)
        # update the learning rate
            lr_scheduler.step()
        # Evaluate 
            evaluate(model, valid_loader, device=device)
        # Save model weights
        torch.save({model.state_dict()}, f"scratch_tmp/$USER/model__mom_{MOM}_decay_{DECAY}_epoch_{epoch+1}_lr_{LR}_backbone_DEFAULT.pt")

        return model



def sample_main():
    # NOTE For demo purposes only; will have class imported elsewhere so then drop the tuned_FasterRCNN() calls

    # TODO Get this damn labeled dataset copied over from Drive to Greene to GCP
    model_standin_import_temporary = tuned_FasterRCNN()
    model_default = model_standin_import_temporary.get_model()
    model_trained = model_standin_import_temporary.train_tuned(model=model_default, VALID_DATASET_PATH='../nyu_dl/datasets/labeled_data/')
    

if __name__ == "__main__":
    sample_main()