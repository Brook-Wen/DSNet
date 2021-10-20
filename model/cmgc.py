import torch
import torch.nn as nn
import torchvision


class CrossModalGlobalContextBlock(nn.Module):
    def __init__(self, inplanes, ratio):
        super(CrossModalGlobalContextBlock, self).__init__()
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes // ratio)
        self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.channel_mul_conv1 = nn.Sequential(
                                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        self.channel_mul_conv2 = nn.Sequential(
                                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_channels=self.inplanes*2, out_channels=self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.relu = nn.ReLU(inplace=True)

    def spatial_pool(self, x1, x2):
        batch, channel, height, width = x1.size()
        # [N, C, H * W]
        x1 = x1.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        x1 = x1.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x2)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(x1, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x1, x2):
        # [N, C, 1, 1]
        context1 = self.spatial_pool(x1, x2)
        context2 = self.spatial_pool(x2, x1)

        out1 = x1 * torch.sigmoid(self.channel_mul_conv1(context1))
        out2 = x2 * torch.sigmoid(self.channel_mul_conv2(context2))
        out = torch.cat([out1, out2], dim=1)

        avgout = torch.mean(out, dim=1, keepdim=True)
        maxout, _ = torch.max(out, dim=1, keepdim=True)
        mask = self.conv2d(torch.cat([avgout, maxout], dim=1))
        mask = self.sigmoid(mask)
        out = self.conv(out) * mask

        return out
