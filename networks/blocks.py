import torch
import torch.nn as nn
import torch.nn.functional as F



##############--- Cross-Attention Module ---################
class CrossAttention(nn.Module):
    def __init__(self, channels):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        """
        x1: Feature map from branch 1 (e.g., reflectance)
        x2: Feature map from branch 2 (e.g., illumination)
        """
        batch_size, C, H, W = x1.size()

        # Query, Key, and Value for cross-attention
        query = self.query_conv(x1).view(batch_size, -1, H * W).permute(0, 2, 1)  # (B, HW, C//8)
        key = self.key_conv(x2).view(batch_size, -1, H * W)  # (B, C//8, HW)
        value = self.value_conv(x2).view(batch_size, -1, H * W)  # (B, C, HW)

        # Attention map
        attention = self.softmax(torch.bmm(query, key))  # (B, HW, HW)

        # Cross-attention output
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(batch_size, C, H, W)  # Reshape to original feature map size

        # Fuse with original feature map
        out = x1 + out  # Residual connection
        return out

##############--- Dual-attention Module (DAM) ---################
class DAM(nn.Module):
    def __init__(self, bc):
        super(DAM, self).__init__()
        self.cross_attention = CrossAttention(bc)

    def forward(self, x1, x2):
        """
        x1: Feature map from branch 1 (e.g., reflectance)
        x2: Feature map from branch 2 (e.g., illumination)
        """
        return self.cross_attention(x1, x2)

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
