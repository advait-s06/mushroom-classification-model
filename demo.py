import torch
import torch.nn as nn
from PIL import Image
from mushroom_model import MyModel
from torchvision.transforms import v2

input_img = Image.open("poisonous_mushroom_demo.png")

transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize(size=(200, 200), antialias=True), 
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(input_img).unsqueeze(0)
model = MyModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))

out = model(torch.tensor(input_tensor))
softmax = nn.Softmax(dim=1)
probabilities = softmax(out)
print(probabilities.squeeze().tolist())