import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

from .resnet import resnet50
from .cmgc import CrossModalGlobalContextBlock
from .cma import CrossModalAttentionBlock
from .cmdp import CrossModalDynamicPoolingBlock


config_resnet = {'convert': [[64,256,512,1024,2048], [64,128,256,512,512], [32,64,128,256,512]]}


class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        self.convert = nn.ModuleList()
        for i in range(len(list_k[0])):
            self.convert.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert[i](list_x[i]))
        return resl


# Bidirectional Gated Module
class BGModule(nn.Module):
    def __init__(self, channel):
        super(BGModule, self).__init__()
        self.reset_gate = nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1)
        self.update_gate = nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1)
        self.out_gate_1 = nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1)
        self.out_gate_2 = nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        update = torch.sigmoid(self.update_gate(x))
        reset = torch.sigmoid(self.reset_gate(x))
        out1 = torch.tanh(self.out_gate_1(torch.cat([x1, x2 * reset], dim=1)))
        out2 = torch.tanh(self.out_gate_2(torch.cat([x2, x1 * reset], dim=1)))
        x = (x1 + x2) * (1 - update) + (out1 + out2) * update
        return x


class MergeLayer(nn.Module):
    def __init__(self, x_channel, t_channel):
        super(MergeLayer, self).__init__()

        self.bgm = nn.ModuleList()
        self.cmdp = nn.ModuleList()
        self.conv_t = nn.ModuleList()
        self.conv_acm = nn.ModuleList()
        self.score = nn.ModuleList()
        self.edge = nn.ModuleList()
        for k in range(len(x_channel)):
            self.bgm.append(BGModule(x_channel[k]))
            self.cmdp.append(CrossModalDynamicPoolingBlock(x_channel[k]))
            self.conv_t.append(nn.Sequential(nn.Conv2d(x_channel[k], t_channel[k], kernel_size=1, stride=1, padding=0, bias=False), 
                                            nn.ReLU(inplace=True), 
                                            nn.Conv2d(t_channel[k], t_channel[k], kernel_size=3, stride=1, padding=1, bias=False)))
            self.conv_acm.append(nn.Sequential(nn.Conv2d(t_channel[k], 32, kernel_size=3, stride=1, padding=1, bias=False), 
                                            nn.ReLU(inplace=True)))
            self.score.append(nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False))
            self.edge.append(nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False))

        self.out = nn.Sequential(nn.Conv2d(32*5, 64, kernel_size=1, stride=1, padding=0, bias=False), 
                                nn.ReLU(inplace=True), 
                                nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, sal_size):
        x = input[-1]
        acm, score, edge = [], [], []
        for i in range(len(input)-2, -1, -1):
            h = input[i]
            x = self.bgm[i](x, h)
            x = self.cmdp[i](x)
            if i == 0:
                x = self.conv_t[i](x)
            else:
                x = self.conv_t[i](F.interpolate(x, input[i-1].size()[2:], mode='bilinear', align_corners=True))
            
            conv_acm = self.conv_acm[i](self.relu(x))
            acm.append(F.interpolate(conv_acm, sal_size[2:], mode='bilinear', align_corners=True))
            conv_score = self.score[i](conv_acm)
            conv_edge = self.edge[i](conv_acm)
            score.append(F.interpolate(conv_score, sal_size[2:], mode='bilinear', align_corners=True))
            edge.append(F.interpolate(conv_edge, sal_size[2:], mode='bilinear', align_corners=True))
            x = x * (self.sigmoid(conv_score) + self.sigmoid(conv_edge))

        out = torch.cat(acm, dim=1)
        out = self.out(out)
        score.append(out)

        return score[::-1], edge[::-1]


# extra part
def extra_layer(base_model_cfg, rgb, depth):
    if base_model_cfg == 'resnet':
        config = config_resnet
    else:
        raise AssertionError
    merge_layers = MergeLayer(config['convert'][1], config['convert'][2])
    return rgb, depth, merge_layers


# network
# Dynamic Selectiv Network
class Model_bone(nn.Module):
    def __init__(self, base_model_cfg, base_rgb, base_depth, merge_layers):
        super(Model_bone, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base_rgb = base_rgb
        self.base_depth = base_depth
        self.merge = merge_layers

        self.cma = nn.ModuleList()
        if self.base_model_cfg == 'resnet':
            self.convert1 = ConvertLayer(config_resnet['convert'])
            self.convert2 = ConvertLayer(config_resnet['convert'])
            for i in config_resnet['convert'][1]:
                self.cma.append(CrossModalAttentionBlock(i))
        else:
            raise AssertionError

        self.cmgc = CrossModalGlobalContextBlock(inplanes=512, ratio=16)

    def forward(self, x, h, use_gc=False):
        sal_size = x.size()
        cm = []
        out1 = self.base_rgb(x)
        out2 = self.base_depth(h)
        if self.base_model_cfg == 'resnet':
            out1 = self.convert1(out1)
            out2 = self.convert2(out2)
            for i in range(len(config_resnet['convert'][1])):
                cm.append(self.cma[i](out1[i], out2[i]))
        else:
            raise AssertionError

        if use_gc:
            gc = self.cmgc(out1[-1], out2[-1])
            cm.append(gc)
        else:
            cm.append(cm[-1])

        return self.merge(cm, sal_size)


# build the whole network
def build_model(base_model_cfg='resnet'):
    if base_model_cfg == 'resnet':
        return Model_bone(base_model_cfg, *extra_layer(base_model_cfg, resnet50(), resnet50()))
    else:
        raise AssertionError


# weight init
def xavier(param):
    # init.xavier_uniform(param)
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        # m.weight.data.normal_(0.0, 0.01)
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.zero_()
        # m.eval()
    elif isinstance(m, nn.GroupNorm):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.01)
        # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


