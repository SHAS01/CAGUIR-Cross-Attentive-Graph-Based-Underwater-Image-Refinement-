import torch
import torch.nn as nn
import torch.nn.functional as F


##############--- CBAM Attention ---################
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.fc(x)
        return x * weight

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = SEBlock(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

##############--- Dual-attention Module (DAM) ---################
class DAM(nn.Module):
    def __init__(self, bc, r, dim):
        super(DAM, self).__init__()
        self.cbam = CBAM(bc, reduction=r)

    def forward(self, x):
        return self.cbam(x)

##############--- Multi-blocks ---################
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

##############--- Residual Block (RB) ---################
class RB(nn.Module):
    def __init__(self, bc=64):
        super(RB, self).__init__()
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(bc, bc, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(bc, bc, 3, 1, 1)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(bc, bc, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(bc, bc, 3, 1, 1)
        )
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(bc, bc, 1),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(bc, bc, 1)
        # )

    def forward(self, x):
        x1 = self.conv3_2(self.conv3_1(x))
        x = x + x1
        return x

##############--- Residual Skip Block (RSB) ---################
class RSB(nn.Module):
    def __init__(self, bc=64):
        super(RSB, self).__init__()
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(bc, bc, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(bc, bc, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(bc, bc, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(bc, bc, kernel_size=1, bias=False)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(bc, bc, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(bc, bc, kernel_size=1, bias=False)
        )

    def forward(self, x, skip_x):
        x1 = self.conv1(self.conv3_1(x))
        x1 = x1 + self.conv3_2(skip_x)
        x = x + x1
        return x
