import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn(nn.Module):
    def __init__(self, class_num=10, input_channel=3):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc = nn.Linear(180, class_num)
        pass

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        f = x
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc(x)
        return x, f
        pass
