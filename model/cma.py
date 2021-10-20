import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicConvBlock(nn.Module):
    def __init__(self, channels):
        super(DynamicConvBlock, self).__init__()
        self.convk1d1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, groups=channels, bias=False)
        self.convk3d1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.convk5d1 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, groups=channels, bias=False)
        self.convk7d1 = nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3, groups=channels, bias=False)
        self.convk3d3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3, dilation=3, groups=channels, bias=False)
        self.convk3d5 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=5, dilation=5, groups=channels, bias=False)
        self.convk3d7 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=7, dilation=7, groups=channels, bias=False)
        self.convk1 = nn.Conv2d(channels, channels // 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, a):
        c1 = self.convk1d1(x)
        c2 = self.convk3d1(x)
        c3 = self.convk5d1(x)
        c4 = self.convk7d1(x)
        c5 = self.convk3d1(x)
        c6 = self.convk3d5(x)
        c7 = self.convk3d7(x)

        out = self.relu(x*a[0] + c1*a[1] + c2*a[2] + c3*a[3] + c4*a[4] + c5*a[5] + c6*a[6] + c7*a[7])
        return self.convk1(out)


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv2d(out)
        # out = self.sigmoid(out)
        out = self.tanh(out) + 1
        return out


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(CrossModalAttentionBlock, self).__init__()
        self.ca = ChannelAttentionModule(channels*3, 8)
        self.sa = SpatialAttentionModule()
        self.dc = DynamicConvBlock(channels*3)

    def forward(self, x1, x2):
        out = torch.cat((x1, x2, x1 + x2), dim=1)
        a = self.ca(out).view(-1)
        out = self.dc(out, a)
        out = self.sa(out) * out
        return out
        