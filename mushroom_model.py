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

train_dataset = ImageFolder('split_mushroom_data/train', transform=train_transforms)
val_dataset = ImageFolder('split_mushroom_data/val', transform=val_test_transforms)
test_dataset = ImageFolder('split_mushroom_data/test', transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Visualization
# for idx, (images, labels) in enumerate(train_loader):
    # break

# plt.figure(figsize=(20,20))

# for idx, image in enumerate(images):
    # if idx < 100:
       # plt1 = plt.subplot(10, 10, idx + 1)
       # image = image.permute(1, 2, 0)
       # plt1.imshow(image)
        #plt1.set_title(train_dataset.classes[labels[idx].item()])
        #plt1.axis('off')
#plt.tight_layout()
#plt.show()

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1000, 3, 1, 1)
        self.conv2 = nn.Conv2d(1000, 2000, 3, 1, 1)
        self.conv3 = nn.Conv2d(2000, 3000, 5, 1, 2)
        self.conv4 = nn.Conv2d(3000, 3500, 5, 1, 2)
        self.linear1 = nn.Linear(3500*6*6, 30000)
        self.linear2 = nn.Linear(30000, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.relu(self.conv1(input))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.relu(self.linear1(x))
        output = self.linear2(x)
        return output

model = MyModel()
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
NUM_EPOCHS = 1
# Iterating through training, val, and test data
for i in range(NUM_EPOCHS):
    total_correct = 0
    for idx, (train_X, train_y) in enumerate(train_loader):
        train_preds = model(train_X)
        loss = criterion(train_preds, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        class_train_preds = torch.max(train_preds, axis=1)[1]
        total_correct += (class_train_preds == train_y).sum()
        print("finished batch")
        if idx == 4:
            print("5 train batches done\n")
            break
    print(f"Training Accuracy: {total_correct / len(train_loader)}")
    total_correct = 0
    for idx, (val_X, val_y) in enumerate(val_loader):
        val_preds = model(val_X)
        loss = criterion(val_preds, val_y)
        class_val_preds = torch.max(val_preds, axis=1)[1]
        total_correct += (class_val_preds == val_y).sum()
        if idx == 4:
            print("5 val batches done\n")
            break
    print(f"Validation Accuracy: {total_correct / len(val_loader)}")

with torch.no_grad():
    total_correct = 0
    for idx, (test_X, test_y) in enumerate(test_loader):
        test_preds = model(test_X)
        loss = criterion(test_preds, test_y)
        class_test_preds = torch.max(test_preds, axis=1)[1]
        total_correct += (class_test_preds == test_y).sum()
        if idx == 4:
            print("5 test batches done\n")
            break
    print(f"Test Accuracy: {total_correct / len(test_loader)}")
