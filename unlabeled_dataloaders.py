import torch
import os
from PIL import Image
import torchvision
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import math

#VALID_DATASET_PATH = "./unlabeled_data/" # this is the path where our images and labels are
#BATCH_SIZE = 2
#NUM_WORKERS = 2
#SHUFFLE = False # whether we want to shiffle the dataset, should change to True as in SimCLRv1
IMAGE_SIZE = 112
S = 1.0 # strength of color distortion

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

transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.8*S, hue=.5*S, saturation=.8*S, contrast=.2*S)], p=0.8), # hue should be between [-0.5, 0.5]
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(math.floor(0.1*IMAGE_SIZE), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
])

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /unlabeled
            transform: the transform you want to applied to the images.
        """
        self.transform = transform
        self.image_dir = root
        self.num_images = len(os.listdir(self.image_dir))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # the idx of labeled image is from 0
        with open(os.path.join(self.image_dir, f"{idx}.PNG"), "rb") as f:
            img = Image.open(f).convert("RGB")

        return transform(img), transform(img)

def data_loader(BATCH_SIZE = 2, NUM_WORKERS = 2, SHUFFLE = False, VALID_DATASET_PATH="./unlabeled_data/"):
  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    unlabeled_dataset = UnlabeledDataset(
        VALID_DATASET_PATH, 
        transform=transform,
    )

    loader = torch.utils.data.DataLoader(
            unlabeled_dataset,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            num_workers=NUM_WORKERS,
            collate_fn=None,
            drop_last=True
    )

    return loader

#if __name__ == "__main__":
    #data_loader = main()
    # if batch = 2 then
    # 1st image -> has 2 augmentation, to get each augmentation index liek this: batch_images[0][0], batch_images[0][1]
    # 2nd image -> has 2 augmentation, to get each augmentation index liek this: batch_images[1][0], batch_images[1][1]
    #batch_images = next(data_loader) 
    # to display uncomment below
    #show((batch_images[0][0] * 255).to(torch.uint8)) # x_i
    #show((batch_images[0][1] * 255).to(torch.uint8)) # x_j
    #show((batch_images[1][0] * 255).to(torch.uint8)) # x_i
    #show((batch_images[1][1] * 255).to(torch.uint8)) # x_j