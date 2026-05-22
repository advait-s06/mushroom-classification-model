import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image
from torchvision import datasets
from torchvision.transforms import v2
import splitfolders
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(127)

train_transforms = v2.Compose([
    v2.ToTensor(), # Applied to all images
    v2.Resize(size=(100, 100)), # Applies to all images
    v2.RandomHorizontalFlip(0.15), # Randomly applied to images with 0.15 probability
    v2.RandomPerspective(0.3, 0.15), # Randomly applied distortion to images with 0.15 probability
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalizes all inputs
])

val_test_transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize(size=(100, 100)),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

splitfolders.ratio('organized_mushroom_data', output='split_mushroom_data', seed=1337, ratio=(.8, .1, .1))

train_dataset = datasets.ImageFolder('split_mushroom_data/train', transform=train_transforms)
val_dataset = datasets.ImageFolder('split_mushroom_data/val', transform=val_test_transforms)
test_dataset = datasets.ImageFolder('split_mushroom_data/test', transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Visualization
for idx, (images, labels) in enumerate(train_loader):
    break

plt.figure(figsize=(20,20))

for idx, image in enumerate(images):
    if idx < 100:
        plt1 = plt.subplot(10, 10, idx + 1)
        image = image.permute(1, 2, 0)
        plt1.imshow(image)
        plt1.set_title(train_dataset.classes[labels[idx].item()])
        plt1.axis('off')
plt.tight_layout()
plt.show()

NUM_EPOCHS = 10
# Will make loop traversing through the loaders and printing intput and output values later 