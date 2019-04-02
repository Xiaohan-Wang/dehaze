# -*- coding: utf-8 -*-

import torch
from DehazeNet import RankingFunc
import ipdb


x = torch.rand((2, 2, 4, 4))
x.requires_grad = True
y = RankingFunc.apply(x, 3)

y_grad = torch.rand(2, 2, 4, 4) * 100
y.backward(y_grad)
ipdb.set_trace()

