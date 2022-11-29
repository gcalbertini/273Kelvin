import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import yaml

from helper_data import class_dict, collate_fn

#from torchvision.utils import draw_bounding_boxes
#from helper_data import show, unnormalize

class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, img_size):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /labeled
            split: The split you want to used, it should be training or validation
            transform: the transform you want to applied to the images.
        """
        self.IMAGE_SIZE = img_size
        self.split = split
        self.transforms = A.Compose([
                            A.augmentations.geometric.resize.SmallestMaxSize(max_size=self.IMAGE_SIZE , interpolation=cv2.INTER_CUBIC, always_apply=False, p=1),
                            A.RandomSizedBBoxSafeCrop(height=self.IMAGE_SIZE , width=self.IMAGE_SIZE , erosion_rate=0.0),
                            A.HorizontalFlip(p=0.5),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2(),  # convert PIL to Pytorch Tensor
                        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        
        self.image_dir = os.path.join(root, split, "images")
        self.label_dir = os.path.join(root, split, "labels")

        self.num_images = len(os.listdir(self.image_dir))

    def __len__(self):
        return self.num_images  # self.num_images

    def __getitem__(self, idx):
        # the idx of training image is from 1 to 30000
        # the idx of validation image is from 30001 to 50000

        if self.split == "training":
            offset = 1
        if self.split == "validation":
            offset = 30001

        with open(os.path.join(self.image_dir, f"{idx + offset}.JPEG"), "rb") as f:
            img = Image.open(f).convert("RGB")
        with open(os.path.join(self.label_dir, f"{idx + offset}.yml"), "rb") as f:
            yamlfile = yaml.load(f, Loader=yaml.FullLoader)
        
        num_objs = len(yamlfile["labels"])
        # xmin, ymin, xmax, ymax
        boxes = torch.as_tensor(yamlfile["bboxes"], dtype=torch.float32)
        labels = []
        for label in yamlfile["labels"]:
            labels.append(class_dict[label])
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            #img, target = self.transforms(img, target)
            img = np.array(img)
            transformed = self.transforms(image=img, bboxes=target["boxes"], class_labels=target["labels"])
            img = transformed['image']
            target["boxes"] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            target["labels"] = torch.as_tensor(transformed['class_labels'], dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["area"] = area

        return img, target

def labeled_dataloader(BATCH_SIZE=2, NUM_WORKERS=2, SHUFFLE=True, DATASET_PATH="./labeled_data/", SPLIT="training", IMAGE_SIZE=224):

    labeled_dataset = LabeledDataset(
        root=DATASET_PATH,
        split=SPLIT,
        img_size=IMAGE_SIZE
    )

    labeled_dataloader = torch.utils.data.DataLoader(
        labeled_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True
    )

    return labeled_dataset, labeled_dataloader

#if __name__ == "__main__":
    #dataset, loader = labeled_dataloader()
    #batch = next(iter(loader))
    #drawn_boxes = draw_bounding_boxes((unnormalize(batch[0][1]) * 255).to(torch.uint8), batch[1][1]['boxes'], colors="red")
    #show(drawn_boxes)