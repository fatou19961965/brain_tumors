import torch.nn as nn
from torchvision import models

def get_pytorch_model():
    resnet = models.resnet18(weights='DEFAULT')
    for param in resnet.parameters():
        param.requires_grad = False
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 4)  # 4 classes for breast cancer dataset
    return resnet