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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from torch.optim.lr_scheduler import ExponentialLR as ExpLR

torch.manual_seed(127)
device = "cuda" if torch.cuda.is_available() else "cpu"

train_transforms = v2.Compose([
    v2.ToTensor(), # Applied to all images
    # v2.Resize(size=(100, 100), antialias=True), # Applies to all images
    v2.RandomResizedCrop(size=(200, 200), antialias=True),
    v2.RandomRotation(15),
    v2.RandomHorizontalFlip(0.5), # Randomly applied to images with 0.5 probability
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalizes all inputs
])

val_test_transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize(size=(200, 200)),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

splitfolders.ratio('organized_mushroom_data', output='split_mushroom_data', seed=1337, ratio=(.8, .1, .1))

train_dataset = ImageFolder('split_mushroom_data/train', transform=train_transforms)
val_dataset = ImageFolder('split_mushroom_data/val', transform=val_test_transforms)
test_dataset = ImageFolder('split_mushroom_data/test', transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=16, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=16, pin_memory=True)

# Visualization
visualize = input("Do you want to visualize the images (y/n)? ")
if visualize == "y":
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
    plt.savefig('image-visualize.png')
    plt.show()

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, 1, 1)
        self.batchNorm1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, 3, 1, 1)
        self.batchNorm2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 30, 5, 1, 2)
        self.batchNorm3 = nn.BatchNorm2d(30)
        self.conv4 = nn.Conv2d(30, 35, 5, 1, 2)
        self.batchNorm4 = nn.BatchNorm2d(35)
        self.linear1 = nn.Linear(35*12*12, 300)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(300, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.relu(self.batchNorm1(self.conv1(input)))
        # x = self.relu(self.conv1(input))
        x = self.pool(x)
        x = self.relu(self.batchNorm2(self.conv2(x)))
        # x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.batchNorm3(self.conv3(x)))
        # x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.batchNorm4(self.conv4(x)))
        # x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        output = self.linear2(x)
        return output

model = MyModel()
model = model.to(device)
model.train()

class_weights = torch.tensor([1.1, 1.3, 1.0, 1.2])
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.000001)
# scheduler = ExpLR(optimizer, gamma=0.9)
NUM_EPOCHS = 35

for images, labels in train_loader:
    break

# Iterating through training, val, and test data
if __name__ == "__main__":
    train_losses = []
    val_losses = []
    for i in range(NUM_EPOCHS):
        model.train()
        total_correct = 0
        for idx, (train_X, train_y) in enumerate(train_loader):
            train_X = train_X.to(device)
            train_y = train_y.to(device)
            train_preds = model(train_X)
            pNorm = sum(param.abs().pow(2).sum() for param in model.parameters())
            loss = criterion(train_preds, train_y) + 0.0001*pNorm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            class_train_preds = torch.max(train_preds, axis=1)[1]
            total_correct += (class_train_preds == train_y).sum().item()
        # scheduler.step()
        train_losses.append(loss.detach().cpu().numpy())
        print(f"Epoch: {i + 1} \nTraining Accuracy: {total_correct / len(train_dataset)} and Loss: {loss}")
        total_correct = 0
        model.eval()
        for idx, (val_X, val_y) in enumerate(val_loader):
            val_X = val_X.to(device)
            val_y = val_y.to(device)
            val_preds = model(val_X)
            loss = criterion(val_preds, val_y)
            class_val_preds = torch.max(val_preds, axis=1)[1]
            total_correct += (class_val_preds == val_y).sum().item()
        val_losses.append(loss.detach().cpu().numpy())
        print(f"Validation Accuracy: {total_correct / len(val_dataset)} and Loss: {loss}")

    # train/val loss graph
    plt.title('Train Loss vs. Epochs')
    plt.plot(range(NUM_EPOCHS), train_losses)
    plt.xlabel("Epoch #")
    plt.ylabel("Train Loss")
    plt.savefig('Train-LossVSEpoch.png')
    plt.show()

    plt.title('Val Loss vs. Epochs')
    plt.plot(range(NUM_EPOCHS), val_losses)
    plt.xlabel("Epoch #")
    plt.ylabel("Val Loss")
    plt.savefig('Val-LossVSEpoch.png')
    plt.show()

    model.eval()
    total_targets = []
    total_preds = []
    test_losses = []
    with torch.no_grad():
        total_correct = 0
        for idx, (test_X, test_y) in enumerate(test_loader):
            test_X = test_X.to(device)
            test_y = test_y.to(device)
            test_preds = model(test_X)
            loss = criterion(test_preds, test_y)
            class_test_preds = torch.max(test_preds, axis=1)[1]
            total_correct += (class_test_preds == test_y).sum().item()

            total_targets.extend(test_y.tolist())
            total_preds.extend(class_test_preds.squeeze().tolist())
        test_losses.append(loss.detach().cpu().numpy())

        print(f"Test Accuracy: {total_correct / len(test_dataset)} and Loss: {loss}")

# Calculate precision, recall, and create Confusion Matrix
precision = precision_score(total_targets, total_preds, average=None)
recall = recall_score(total_targets, total_preds, average=None)
print(f"Precision: Conditionally Edible - {precision[0]}, Deadly - {precision[1]}, Edible - {precision[2]}, Poisonous - {precision[3]}")
print(f"Recall: Conditionally Edible - {recall[0]}, Deadly - {recall[1]}, Edible - {recall[2]}, Poisonous - {recall[3]}")

cm = confusion_matrix(total_targets, total_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot()
plt.savefig('confusion-matrix.png')
plt.show() 

# Saving the model to a file path
torch.save(model.state_dict(), "model.pt")
