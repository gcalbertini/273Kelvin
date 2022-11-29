import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Subset

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, IMAGE_SIZE):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /unlabeled
            transform: the transform you want to applied to the images.
        """
        self.image_dir = root
        self.num_images = len(os.listdir(self.image_dir))
        self.IMAGE_SIZE = IMAGE_SIZE

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # the idx of labeled image is from 0
        with open(os.path.join(self.image_dir, f"{idx}.PNG"), "rb") as f:
            img = Image.open(f).convert("RGB")
        return img

def create_subset(DATASET_PATH="/unlabeled/"):
    trainset = UnlabeledDataset(DATASET_PATH, IMAGE_SIZE=224)
    train_indices = list(range(20000))
    train_subset = Subset(trainset, train_indices)
    torch.save(train_subset, '/train_subset.pt')

def main():
    create_subset()

if __name__ == '__main__':
    main()
