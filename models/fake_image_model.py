import torch.nn as nn
from torchvision import models

class FakeImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        # freeze feature extractor
        for p in self.base_model.features.parameters():
            p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def build_model(unfreeze_at_epoch: int = None):
    model = FakeImageModel()
    model.unfreeze_at = unfreeze_at_epoch
    return model

