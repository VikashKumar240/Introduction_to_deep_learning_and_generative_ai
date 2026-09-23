import torch
import torch.nn as nn


class ANNClassifier(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)

        )

    def forward(self, x):
        return self.model(x)