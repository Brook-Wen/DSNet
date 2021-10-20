import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class CrossModalDynamicPoolingBlock(nn.Module):
    def __init__(self, channels, pool_size=[2, 4, 8]):
        super(CrossModalDynamicPoolingBlock, self).__init__()
        self.att = ChannelAttentionModule(channels, 7)

        self.pool = nn.ModuleList()
        for i in pool_size:
            self.pool.append(nn.Sequential(nn.MaxPool2d(kernel_size=i, stride=i), 
                            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)))
            self.pool.append(nn.Sequential(nn.AvgPool2d(kernel_size=i, stride=i), 
                            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)))

        self.conv = nn.Conv2d(channels*7, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        att = self.att(x).view(-1)

        out = x * att[-1]
        for i in range(len(self.pool)):
            p = F.interpolate(self.pool[i](x) * att[i], x.size()[2:], mode='bilinear', align_corners=True)
            out = torch.cat([out, p], dim=1)
        
        return self.conv(self.relu(out))
