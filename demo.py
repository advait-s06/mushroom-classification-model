import torch
from mushroom_model import MyModel

model = MyModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))

# out = model(input)