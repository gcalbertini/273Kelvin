import torch
import torchvision
from torchvision import transforms

# These numbers are mean and std values for channels of natural images. 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Inverse transformation: needed for plotting.
unnormalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

train_transforms = transforms.Compose([
                                    #transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(hue=.1, saturation=.1, contrast=.1),
                                    #transforms.RandomRotation(20, Image.BILINEAR),
                                    transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
                                    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
                                    normalize,
                                ])

validation_transforms = transforms.Compose([
                                    #transforms.Resize((224, 224)),
                                    transforms.ToTensor(), 
                                    normalize,
                                ])

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import yaml
from yaml.loader import SafeLoader

class MyLabelledDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels"))))
    
    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        with open(os.path.join(self.root, "labels", self.labels[idx])) as f:
            data = yaml.load(f, Loader=SafeLoader)
        
        target = {}
        target["boxes"] = torch.as_tensor(data['bboxes'], dtype=torch.float32)
        target["labels"] = data['labels'] # modify text -> ints
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

#root_training = 'labeled_data/training/'
#train_dataset = MyLabelledDataset('labeled_data/training/', train_transforms)

#oot_training = 'labeled_data/validation/'
#validation_dataset = MyLabelledDataset('labeled_data/training/', validation_transforms)

#data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)

#example = next(iter(data_loader_train))