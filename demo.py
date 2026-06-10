import torch
import torch.nn as nn
from PIL import Image
from mushroom_model import MyModel
import torchvision.transforms as transforms

input_img = Image.open("poisonous_mushroom_demo.png")
input_tensor = transforms.ToTensor(input_img)

model = MyModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))

out = model(input_tensor)
probabilities = nn.Softmax(out)
print(probabilities.tolist())