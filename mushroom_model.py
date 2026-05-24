import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import splitfolders

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

train_dataset = ImageFolder('split_mushroom_data/train', transform=train_transforms)
val_dataset = ImageFolder('split_mushroom_data/val', transform=val_test_transforms)
test_dataset = ImageFolder('split_mushroom_data/test', transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(train_loader)

NUM_EPOCHS = 1
# Iterating through training data
for i in range(NUM_EPOCHS):
    for train_X, train_y in train_loader:
        print((train_X, train_y))
    for val_X, val_y in val_loader:
        print((val_X, val_y))

with torch.no_grad():
    for test_X, test_y in test_loader:
        print((test_X, test_y))