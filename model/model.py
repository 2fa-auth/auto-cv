
import torch
import torch.nn as nn


class DSResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))


class FastResidualDetector(nn.Module):
    def __init__(self, in_chan=3, out_features=11):
        super().__init__()

        self.backbone = nn.Sequential(
            # 448 -> 224
            nn.Conv2d(
                in_chan, 24, kernel_size=3,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True),
            # 224 -> 112
            DSResidualBlock(24, 32, stride=2),
            DSResidualBlock(32, 32),
            # 112 -> 56
            DSResidualBlock(32, 48, stride=2),
            DSResidualBlock(48, 48),
            # 56 -> 28
            DSResidualBlock(48, 80, stride=2),
            DSResidualBlock(80, 80),
            # 28 -> 14
            DSResidualBlock(80, 128, stride=2),
        )

        self.head = nn.Sequential(
            # 14 -> 7
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                128, out_features,
                kernel_size=1
            ),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x.permute(0, 2, 3, 1).contiguous()
