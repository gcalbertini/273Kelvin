import torch
import os
from PIL import Image
import torchvision
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

VALID_DATASET_PATH = "./unlabeled_data/" # this is the path where our images and labels are
BATCH_SIZE = 2
NUM_WORKERS = 2
SHUFFLE = False # whether we want to shiffle the dataset

def collate_fn(batch):
    return tuple(zip(*batch))

transform1 = transforms.Compose([
          transforms.RandomCrop(56),
          transforms.Resize(112),
          transforms.ColorJitter(brightness=0.5, hue=.2, saturation=.3, contrast=.2),
          transforms.GaussianBlur(5, sigma=(0.6, 1.0)),
          transforms.ToTensor(),
])

transform2 = transforms.Compose([
          transforms.RandomCrop(56),
          transforms.Resize(112),
          transforms.ColorJitter(brightness=0.9, hue=.2, saturation=.3, contrast=.9),
          transforms.GaussianBlur(3, sigma=(0.2, 1.0)),
          transforms.ToTensor(),
])

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform1, transform2):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /unlabeled
            transform: the transform you want to applied to the images.
        """
        self.transform1 = transform1
        self.transform2 = transform2
        self.image_dir = root
        self.num_images = len(os.listdir(self.image_dir))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # the idx of labeled image is from 0
        with open(os.path.join(self.image_dir, f"{idx}.PNG"), "rb") as f:
            img = Image.open(f).convert("RGB")

        return self.transform1(img), self.transform2(img)

def main():
  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    unlabeled_dataset = UnlabeledDataset(
        VALID_DATASET_PATH, 
        transform1=transform1,
        transform2=transform2,
    )

    loader = torch.utils.data.DataLoader(
            unlabeled_dataset,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
    )

    return iter(loader)

if __name__ == "__main__":
    data_loader = main()
    # if batch = 2 then
    # 1st image -> has 2 augmentation, to get each augmentation index liek this: batch_images[0][0], batch_images[1][0]
    # 2nd image -> has 2 augmentation, to get each augmentation index liek this: batch_images[0][1], batch_images[1][1]
    batch_images = next(data_loader) 