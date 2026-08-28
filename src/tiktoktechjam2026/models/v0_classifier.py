import torch.nn as nn


class V0Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        return self.classifier(x)