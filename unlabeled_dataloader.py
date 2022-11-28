import os
import torch
import numpy as np
import math
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# use if you want to display images
def show(imgs):
    plt.rcParams["savefig.bbox"] = "tight"
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, IMAGE_SIZE, S):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /unlabeled
            transform: the transform you want to applied to the images.
        """
        self.image_dir = root
        #self.num_images = len(os.listdir(self.image_dir))
        self.num_images = 512000
        self.IMAGE_SIZE = IMAGE_SIZE
        self.S = S # this is colour distortion, applied to ColorJitter
        self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(self.IMAGE_SIZE, scale=(0.08, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.8*self.S, hue=.5*self.S, saturation=.8*self.S, contrast=.2*self.S)], p=0.8), # hue should be between [-0.5, 0.5]
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(math.floor(0.1*self.IMAGE_SIZE), sigma=(0.1, 2.0)),
                    transforms.ToTensor()])
                    # should we normalise too?

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # the idx of labeled image is from 0
        print("idx: ", idx)
        with open(os.path.join(self.image_dir, f"{idx}.PNG"), "rb") as f:
            img = Image.open(f).convert("RGB")
        print("idx:", idx)
        return self.transform(img), self.transform(img)

def unlabeled_dataloader(BATCH_SIZE=2, NUM_WORKERS=2, SHUFFLE=False, DATASET_PATH="./unlabeled_data/", IMAGE_SIZE=112, S=1.0):

    unlabeled_dataset = UnlabeledDataset(
        DATASET_PATH,
        IMAGE_SIZE,
        S
    )

    unlabeled_dataloader = torch.utils.data.DataLoader(
            unlabeled_dataset,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            num_workers=NUM_WORKERS,
            collate_fn=None,
            drop_last=True
    )

    return unlabeled_dataset, unlabeled_dataloader

'''
# if you want to display images
if __name__ == "__main__":
    dataset, loader = unlabeled_dataloader()
    batch_images = next(iter(loader)) 

    # if batch = 2 then
        # 1st image -> has 2 augmentation, to get each augmentation index liek this: batch_images[0][0], batch_images[0][1]
        # 2nd image -> has 2 augmentation, to get each augmentation index liek this: batch_images[1][0], batch_images[1][1]

    show((batch_images[0][0] * 255).to(torch.uint8)) # x_i
    show((batch_images[0][1] * 255).to(torch.uint8)) # x_j
    show((batch_images[1][0] * 255).to(torch.uint8)) # x_i
    show((batch_images[1][1] * 255).to(torch.uint8)) # x_j
'''