import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import yaml
from yaml.loader import SafeLoader
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from torchvision.utils import draw_bounding_boxes

import albumentations as A
from albumentations.pytorch import ToTensorV2

VALID_DATASET_PATH = "./labeled_data/" # this is the path where our images and labels are
BATCH_SIZE = 1
NUM_WORKERS = 2
SHUFFLE = False # whether we want to shiffle the dataset
SPLIT = "training" # whether we are doing training or validation (the dataset changes)

plt.rcParams["savefig.bbox"] = "tight"
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def collate_fn(batch):
    return tuple(zip(*batch))

# fmt: off
class_dict = {
    "cup or mug": 0, "bird": 1, "hat with a wide brim": 2, "person": 3, "dog": 4, "lizard": 5, "sheep": 6, "wine bottle": 7,
    "bowl": 8, "airplane": 9, "domestic cat": 10, "car": 11, "porcupine": 12, "bear": 13, "tape player": 14, "ray": 15, "laptop": 16,
    "zebra": 17, "computer keyboard": 18, "pitcher": 19, "artichoke": 20, "tv or monitor": 21, "table": 22, "chair": 23,
    "helmet": 24, "traffic light": 25, "red panda": 26, "sunglasses": 27, "lamp": 28, "bicycle": 29, "backpack": 30, "mushroom": 31,
    "fox": 32, "otter": 33, "guitar": 34, "microphone": 35, "strawberry": 36, "stove": 37, "violin": 38, "bookshelf": 39,
    "sofa": 40, "bell pepper": 41, "bagel": 42, "lemon": 43, "orange": 44, "bench": 45, "piano": 46, "flower pot": 47, "butterfly": 48,
    "purse": 49, "pomegranate": 50, "train": 51, "drum": 52, "hippopotamus": 53, "ski": 54, "ladybug": 55, "banana": 56, "monkey": 57,
    "bus": 58, "miniskirt": 59, "camel": 60, "cream": 61, "lobster": 62, "seal": 63, "horse": 64, "cart": 65, "elephant": 66,
    "snake": 67, "fig": 68, "watercraft": 69, "apple": 70, "antelope": 71, "cattle": 72, "whale": 73, "coffee maker": 74, "baby bed": 75,
    "frog": 76, "bathing cap": 77, "crutch": 78, "koala bear": 79, "tie": 80, "dumbbell": 81, "tiger": 82, "dragonfly": 83, "goldfish": 84,
    "cucumber": 85, "turtle": 86, "harp": 87, "jellyfish": 88, "swine": 89, "pretzel": 90, "motorcycle": 91, "beaker": 92, "rabbit": 93,
    "nail": 94, "axe": 95, "salt or pepper shaker": 96, "croquet ball": 97, "skunk": 98, "starfish": 99,
}
# fmt: on

"""
# Below is just to keep in mind

# These numbers are mean and std values for channels of natural images. 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(hue=.1, saturation=.1, contrast=.1),
                                    transforms.RandomRotation(20, Image.BILINEAR),
                                    transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
                                    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
                                    normalize,
                                ])

validation_transforms = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor(), 
                                    normalize,
                                ])

"""

# Inverse transformation: needed for plotting.
unnormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


# Albumentations library
transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(hue=.1, saturation=.1, contrast=.1),
    A.SafeRotate(90, Image.BILINEAR, p=1),
    A.GaussNoise(9),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),  # convert PIL to Pytorch Tensor
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Below is the one actually used during fine-tuning of SimCLRv1
transform2 = A.Compose([
    A.RandomSizedBBoxSafeCrop(width=224, height=224, erosion_rate=0.2),
    A.HorizontalFlip(p=0.5),
    #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),  # convert PIL to Pytorch Tensor
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Below is the one actually used during fine-tuning of SimCLRv1, however, need to make sure that the transformations keep the bounding boxes
transform3 = A.Compose([
    A.resize.SmallestMaxSize(max_size=224, interpolation=1, always_apply=False, p=1),
    A.CenterCrop(height=224, width=224),
    A.HorizontalFlip(p=0.5),
    #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),  # convert PIL to Pytorch Tensor
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /labeled
            split: The split you want to used, it should be training or validation
            transform: the transform you want to applied to the images.
        """

        self.split = split
        self.transforms = transforms

        
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

def main():
  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 100

    valid_dataset = LabeledDataset(
        root=VALID_DATASET_PATH,
        split=SPLIT,
        transforms=transform, # albumentations transformation
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    return iter(valid_loader)

    #model = get_model().to(device)
    #evaluate(model, valid_loader, device=device)

if __name__ == "__main__":
    data_loader = main()
    img = next(data_loader)
    drawn_boxes = draw_bounding_boxes((unnormalize(img[0][0]) * 255).to(torch.uint8), img[1][0]['boxes'], colors="red")
    # uncomment below to print image with bounding boxes
    #show(drawn_boxes) 