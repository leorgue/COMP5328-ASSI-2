import torch.nn as nn
from torchvision import models

def get_model(num_classes=3, input_channels=3):
    model = models.resnet18()
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model