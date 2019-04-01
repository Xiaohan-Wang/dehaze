# -*- coding: utf-8 -*-

import torch
from DehazeNet import RankingFunc
import ipdb


x = torch.rand(2, 3, 4, 4) * 100
y = RankingFunc.apply(x, 3)
a = y + 1
y_grad = torch.rand(2, 3, 4, 4) * 100
y.backward(y_grad)
ipdb.set_trace()
