import torch.nn as nn
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()

        # width = ((width - kernel_size + 2 * padding) / strides) + 1

        # input shape (64, 3, 128, 128)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=3, padding=1)

        # shape (64, 64, 64, 64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32,
                               kernel_size=3, padding=1)

        # shape (64, 32, 32, 32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16,
                               kernel_size=3, padding=1)

        # shape (64, 16, 16, 16)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=8,
                               kernel_size=3, padding=1)

        # shape (64, 8, 8, 8)
        self.fc1 = nn.Linear(in_features=8*8*8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 8*8*8)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)

        return X
