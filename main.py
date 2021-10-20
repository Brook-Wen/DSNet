import argparse
import torch
import os
import numpy as np
from dataset.dataset import get_loader
from solver import Solver

def get_test_info(sal_mode='nju2k'):
    if sal_mode == 'nju2k':
        image_root = './data/test/NJU2K/'
        image_source = './data/test/NJU2K/test.lst'
    elif sal_mode == 'nlpr':
        image_root = './data/test/NLPR/'
        image_source = './data/test/NLPR/test.lst'
    elif sal_mode == 'des':
        image_root = './data/test/DES/'
        image_source = './data/test/DES/test.lst'
    elif sal_mode == 'ssd':
        image_root = './data/test/SSD/'
        image_source = './data/test/SSD/test.lst'
    elif sal_mode == 'lfsd':
        image_root = './data/test/LFSD/'
        image_source = './data/test/LFSD/test.lst'
    elif sal_mode == 'sip':
        image_root = './data/test/SIP/'
        image_source = './data/test/SIP/test.lst'
    elif sal_mode == 'stere':
        image_root = './data/test/STERE/'
        image_source = './data/test/STERE/test.lst'
    elif sal_mode == 'redwebs':
        image_root = './data/test/ReDWeb-S/'
        image_source = './data/test/ReDWeb-S/test.lst'

    return image_root, image_source

def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_folder, run)):
            run += 1
        os.mkdir("%s/run-%d" % (config.save_folder, run))
        os.mkdir("%s/run-%d/models" % (config.save_folder, run))
        config.save_folder = "%s/run-%d" % (config.save_folder, run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        config.test_root, config.test_list = get_test_info(config.sal_mode)
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")

if __name__ == '__main__':

    vgg_path = './pretrained/vgg16_bn.pth'
    resnet_path = './pretrained/resnet50.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5) # Learning rate
    parser.add_argument('--betas', type=float, default=[0.9, 0.999])
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--wd', type=float, default=0.0001) # Weight decay
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')

    # Training settings
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_thread', type=int, default=12)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='./results')
    parser.add_argument('--epoch_save', type=int, default=2)
    parser.add_argument('--iter_size', type=int, default=8)
    parser.add_argument('--show_every', type=int, default=100)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--cpu_num_thread', type=str, default='8')

    # Train data
    parser.add_argument('--train_root', type=str, default='./data/train')
    parser.add_argument('--train_list', type=str, default='./data/train/train.lst')

    # Testing settings
    parser.add_argument('--model', type=str, default=None) # Snapshot
    parser.add_argument('--test_fold', type=str, default=None) # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='nju2k') # Test image dataset

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    os.environ["OMP_NUM_THREADS"] = config.cpu_num_thread

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    # Get test set info
    test_root, test_list = get_test_info(config.sal_mode)
    config.test_root = test_root
    config.test_list = test_list

    main(config)
