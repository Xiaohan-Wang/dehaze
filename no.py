# -*- coding: utf-8 -*-

import torch
from DehazeNet import RankingFunc
import ipdb


x = torch.rand((1, 1, 3, 3))
x.requires_grad = True
y = RankingFunc.apply(x, 3)

y_grad = torch.rand(1, 1, 3, 3) * 100
y.backward(y_grad)
ipdb.set_trace()

#x = torch.tensor(([1, 2]),dtype = torch.float32, requires_grad=True)
#y = x / 2
#z = y / 2
#hook_handle = y.register_hook(variable_hook)
#z.backward(torch.tensor([2, 1], dtype = torch.float32))
