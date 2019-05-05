#!user/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Function

#%%
class BReLU(Function):
    @staticmethod
    def forward(ctx, x):
        x[x > 1] = 1
        x[x < 0] = 0
        ctx.save_for_backward(x)
        return x
    
    @staticmethod
    def backward(ctx, grad_y):
        x = ctx.saved_variables[0]
        grad = torch.ones(x.size())
        if torch.cuda.is_available():
            grad = grad.cuda()
        grad[x == 1] = 0
        grad[x == 0] = 0
        return grad_y * grad

#%%
class DehazePyramid(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_num, kernel_size):
        """
        INPUT PARAMETERS
        ---------------------------------------
        out_channels: out channels for each kernel
        kernel_size: it should correspond to kernel_num
        """
        super().__init__()
        self.kernel_num = kernel_num
        self.net = nn.ModuleList()
        for i in range(kernel_num):
            ks = kernel_size[i] #kernel_size
            self.net.append(nn.Conv2d(in_channels, out_channels, ks, padding = (ks - 1) // 2))
    
    def forward(self, x):
        y = []
        for i in range(self.kernel_num):
            y.append(self.net[i](x))
        return torch.cat(y, dim = 1)

#%%
#class DehazeBlock(nn.Module):
#    def __init__(self, basic_input_channel, basic_output_channel, kernel_size, has_aux = False, \
#                 aux_in_channels = 0, aux_out_channels = 0, aux_kernel_num = 0, aux_kernel_size = None):
#        super().__init__()
#        if has_aux:
#            self.has_aux = True
#            self.dp = DehazePyramid(aux_in_channels, aux_out_channels, aux_kernel_num, aux_kernel_size)
#        aux_total_output = aux_out_channels * aux_kernel_num
#        basic_total_input = aux_total_output + basic_input_channel
#        self.conv = nn.Conv2d(basic_total_input, basic_output_channel, kernel_size, 1, (kernel_size - 1) // 2)
#    
#    def forward(self, x, aux_x):
#        if self.has_aux:
#            aux_out = self.dp(aux_x)
#            x = torch.cat([x, aux_out], dim = 1)
#        return self.conv(x)
   
#%%
class DehazeNet(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size_num, kernel_size):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(layer_num - 1):
            in_channel = in_channels[i]
            out_channel = out_channels[i]
            #TODO: change kernel size?
            ks = kernel_size[i]
            ksn = kernel_size_num[i]
            self.net.append(DehazePyramid(in_channel, out_channel // ksn, ksn, ks))
    
    def forward(self, x):
        #TODO: try to change aux
#        aux1 = torch.tensor([]).cuda()
#        aux1 = x.cuda()
#        aux1 = x[:, 0:3, :, :].cuda()
        aux1 = x[:, 3, :, :].unsqueeze(1).cuda()
        aux2 = x[:, 3, :, :].unsqueeze(1).cuda()
        aux3 = x[:, 3, :, :].unsqueeze(1).cuda()
        aux4 = x[:, 3, :, :].unsqueeze(1).cuda()
        x1 = nn.functional.relu(self.net[0](x))   
        x2 = nn.functional.relu(self.net[1](torch.cat([x1, aux1], dim = 1)))
        x3 = nn.functional.relu(self.net[2](torch.cat([x2, aux2], dim = 1)))
        x4 = nn.functional.relu(self.net[3](torch.cat([x3, aux3], dim = 1)))
#        x5 = BReLU.apply(self.net[4](torch.cat([x4, aux4], dim = 1)))
        x5 = self.net[4](torch.cat([x4, aux4], dim = 1))
        return x5
        

            
        
        
        
        