import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam, SGD, lr_scheduler
from torch.autograd import Variable
from torch.backends import cudnn
from model.model import build_model, weights_init, inplace_relu
from model.loss_functions import *
import torch.nn as nn
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import math
import time


import visdom
vis = visdom.Visdom(env='DS')


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [35,]
        self.pretrained = True
        self.build_model()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            if self.config.cuda:
                self.net.load_state_dict(torch.load(self.config.model), strict=False)
            else:
                self.net.load_state_dict(torch.load(self.config.model, map_location='cpu'), strict=False)
            self.net.eval()

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = build_model(base_model_cfg=self.config.arch)
        self.criterion = BCELoss(reduction='sum')

        if self.config.cuda:
            self.net = self.net.cuda()
            self.criterion = self.criterion.cuda()

        if self.config.mode == 'train':
            self.net.train()
            # self.net.eval()  # use_global_stats = True
        elif self.config.mode == 'test':
            self.net.eval()
        
        self.net.apply(weights_init)
        # self.net.apply(inplace_relu)

        self.lr = self.config.lr
        self.betas = self.config.betas
        self.eps = self.config.eps
        self.wd = self.config.wd
        if self.pretrained:
            if self.config.arch == 'resnet':
                if self.config.load == '':
                    pretrained_dict = torch.load(self.config.pretrained_model)

                    model_dict = self.net.base_rgb.resnet.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    self.net.base_rgb.resnet.load_state_dict(model_dict, strict=False)

                    model_dict = self.net.base_depth.resnet.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    self.net.base_depth.resnet.load_state_dict(model_dict, strict=False)
                else:
                    self.net.load_state_dict(torch.load(self.config.load), strict=False)

                base_params = list(map(id, self.net.base_rgb.resnet.parameters())) + list(map(id, self.net.base_depth.resnet.parameters()))
                sub_params = filter(lambda p: id(p) not in base_params, self.net.parameters())
                sub_params = filter(lambda p: p.requires_grad, sub_params)
                self.optimizer = Adam([{'params': self.net.base_rgb.resnet.parameters()},
                                    {'params': self.net.base_depth.resnet.parameters()},
                                    {'params': sub_params, 'lr': self.lr*10}],
                                    lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.wd)
            else:
                raise AssertionError

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_decay_epoch, gamma=0.1)

        self.print_network(self.net, 'Net Structure')

    def test(self):
        time_t = 0
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, hhas, depths, name, im_size = data_batch['image'], data_batch['hha'], data_batch['depth'], data_batch['name'][0], np.asarray(data_batch['size'])
            with torch.no_grad():
                images = Variable(images)
                hhas = Variable(hhas)
                if self.config.cuda:
                    images = images.cuda()
                    hhas = hhas.cuda()

                time_s = time.time()
                sal_pred, _ = self.net(images, hhas, use_gc=True)
                time_e = time.time()
                time_t += (time_e-time_s)
                pred = np.squeeze(torch.sigmoid(sal_pred[0]).cpu().data.numpy())
                pred = 255 * pred
                if not os.path.exists(self.config.test_fold): os.mkdir(self.config.test_fold)
                cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '.png'), pred)

        print('Speed: %f FPS' % (img_num/time_t))
        print('Test Done!')

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        for epoch in range(self.config.epoch):
            s_loss = 0
            self.optimizer.zero_grad()
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_hha, sal_depth, sal_label, sal_edge = data_batch['sal_image'], data_batch['sal_hha'], data_batch['sal_depth'], data_batch['sal_label'], data_batch['sal_edge']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                sal_image, sal_hha, sal_label, sal_edge = Variable(sal_image), Variable(sal_hha), Variable(sal_label), Variable(sal_edge)
                if self.config.cuda:
                    sal_image, sal_hha = sal_image.cuda(), sal_hha.cuda()
                    sal_label, sal_edge = sal_label.cuda(), sal_edge.cuda()

                sal_pred, edge_pred = self.net(sal_image, sal_hha, use_gc=True)

                for l in range(len(sal_pred)):
                    if l==0:
                        sal_loss = F.binary_cross_entropy_with_logits(sal_pred[l], sal_label, reduction='sum')
                    else:
                        sal_loss += (self.criterion(edge_pred[l-1], sal_edge) * 0.5)
                        sal_loss += (F.binary_cross_entropy_with_logits(sal_pred[l], sal_label, reduction='sum') * 0.5)
                loss = sal_loss / (self.iter_size * self.config.batch_size)
                s_loss += loss.item()
                loss.backward()

                aveGrad += 1

                # accumulate gradients as done in DSS
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if i % (self.show_every // self.config.batch_size) == 0:
                    if i == 0:
                        x_showEvery = 100
                    print('Epoch: [%2d/%2d], Iter: [%5d/%5d]  ||  Loss : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num, s_loss/x_showEvery))
                    print('Learning rate: ' + str(self.lr))
                    s_loss = 0

                if i % 1000 == 0:
                    tmp_path = 'TMP/' + 'tmp' + str(epoch)
                    if not os.path.exists(tmp_path):
                        os.mkdir(tmp_path)

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            self.scheduler.step()
            self.lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([s_loss]), win='train', update='append' if epoch > 1 else None, opts={'title': 'Train Loss'})

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)
