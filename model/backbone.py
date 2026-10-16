#!/home/client/Documents/fun/venv/bin/python3
import torch.nn as nn

class BACKBONEModel(nn.Module):
    def __init__(self, in_chan, out_features=18):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_chan, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(4, 4), # [B, 28, 28, 5*B+C] => [B, 7, 7, 5*B+C]
        )
        self.head = nn.Conv2d(128, out_features, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x.permute(0, 2, 3, 1).contiguous()