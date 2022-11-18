import torch
import os
from PIL import Image
import torchvision
from torchvision import transforms

VALID_DATASET_PATH = "./unlabeled_data/" # this is the path where our images and labels are
BATCH_SIZE = 1
NUM_WORKERS = 2
SHUFFLE = False # whether we want to shiffle the dataset

def collate_fn(batch):
    return tuple(zip(*batch))

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

        return self.transform(img)

def main():
  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    unlabeled_dataset = UnlabeledDataset(
        VALID_DATASET_PATH, 
        transform=lambda x : transforms.ToTensor()(x).unsqueeze_(0),
    )

    loader = torch.utils.data.DataLoader(
            unlabeled_dataset,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
    )

    return iter(loader)

    #model = get_model().to(device)
    #evaluate(model, valid_loader, device=device)

if __name__ == "__main__":
    data_loader = main()
    img = next(data_loader)
    # uncomment to display img
    #show((img[0][0] * 255).to(torch.uint8))