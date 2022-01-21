import torch
import os

#####################################
#
# ADD ROOT DIRECTORY TO PATH
#
#####################################
import sys
abs_root_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
sys.path.insert(1, os.path.join(abs_root_dir))
from functions import parse_configuration

#####################################

import time
import numpy as np
from model.model import parsingNet
from utils.common import merge_config

# torch.backends.cudnn.deterministic = False

args, cfg = merge_config()

torch.backends.cudnn.benchmark = True
backbone_cfg = parse_configuration(cfg)
net = parsingNet(pretrained = False, backbone='38',cls_dim = (100+1,56,4),use_aux=False, backbone_cfg=backbone_cfg).cuda()
# net = parsingNet(pretrained = False, backbone='18',cls_dim = (200+1,18,4),use_aux=False).cuda()

net.eval()

x = torch.zeros((1,3,288,800)).cuda() + 1
for i in range(10):
    y = net(x)

t_all = []
for i in range(100):
    t1 = time.time()
    y = net(x)
    t2 = time.time()
    t_all.append(t2 - t1)

print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:',1 / max(t_all))