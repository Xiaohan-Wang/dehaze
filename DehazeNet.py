#!user/bin/env python
#-*-coding:utf-8-*-

import time
import torch
import torch.nn as nn
from torch.autograd import Function

#%%
class BasicModule(nn.Module):
    """ A class which encapsulates nn.Module and provides interfaces for quick model loading and saving."""
    def __init__(self):
        super().__init__()
    
    def load(self, path):
        """Load model from specified path."""
        self.load_state_dict(torch.load(path))
    
    def save(self, name):
        """Save a model to checkpoints/"""
        prefix = 'checkpoints/'
        if name == None:
            name = time.strftime("%m%d_%H:%M:%S.pth")
        name = prefix + name
        torch.save(self.state_dict(), name)
        return name         

#%%
class RankingFunc(Function): 
    # TODO: check whether it's correct or not
    
    @staticmethod
    def forward(ctx, x, ranking_size): 
        p, c, h, w = x.size()
        y = x.clone()
        ind = torch.arange(0, h*w).view(h, w).unsqueeze(0).repeat(p, c, 1, 1)
        if torch.cuda.is_available():
            ind = ind.cuda()
        
        h_num = h // ranking_size
        w_num = w // ranking_size
        
        #TODO: what is the dimension of x????
        for i in range(h_num):
            for j in range(w_num):
                temp1 = x[:, :, i * ranking_size : (i + 1) * ranking_size, j * ranking_size : (j + 1) * ranking_size].clone()
                temp2 = ind[:, :, i * ranking_size : (i + 1) * ranking_size, j * ranking_size : (j + 1) * ranking_size].clone()
                value, index = torch.sort(temp1.view(p, c, ranking_size ** 2))
                temp1 = value.view(p, c, ranking_size, ranking_size)
                y[:, :, i * ranking_size : (i + 1) * ranking_size, j * ranking_size : (j + 1) * ranking_size] = temp1
                temp2 = temp2.view(p, c, ranking_size ** 2).view(-1)
                index = index.view(-1)
                temp2 = temp2[index].view(p, c, ranking_size, ranking_size)
                ind[:, :, i * ranking_size : (i + 1) * ranking_size, j * ranking_size : (j + 1) * ranking_size] = temp2
        ctx.save_for_backward(ind)
        return y
    
    @staticmethod
    def backward(ctx, grad_y):
        index = ctx.saved_variables[0]
        grad_x = torch.empty(grad_y.size())
        if torch.cuda.is_available():
            grad_x = grad_x.cuda()
        p, c, h, w = grad_y.size()
        for i in range(p):
            for j in range(c):
                temp = grad_y[i, j, :, :].view(-1)
                _, temp_index = index[i, j, :, :].view(-1).sort()
                grad_x[i, j, :, :] = temp[temp_index].view(h,w)
                
        return grad_x, None
    
#def ranking_func(x, ranking_size):      
#    h_num = x.size()[2] // ranking_size
#    w_num = x.size()[3] // ranking_size
#    for i in range(h_num):
#        for j in range(w_num):
#            temp = x[:, :, i * ranking_size : (i + 1) * ranking_size, j * ranking_size : (j + 1) * ranking_size]
#            value = torch.sort(temp.resize(x.size()[0], x.size()[1], ranking_size * ranking_size))[0]
#            temp = value.resize(x.size()[0], x.size()[1], ranking_size, ranking_size)
#    return x

#%%
class DehazeBlock(BasicModule):
    def __init__(self, in_channel, out_channel, kernel_size, dilation = 1, conv = True, ranking = False):
        """
        INPUT PARAMETERS
        ----------------------------------------------
        dilation: atrous rate
        : atrous olution path
        ranking: ranking path
        ranking_size: the size of each ranking region
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, 1, (kernel_size - 1)// 2 * dilation, dilation, bias = True)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace = True)
        
        self.has_conv = conv
        self.has_ranking = ranking
        #TODO：should we change the size?
        self.ranking_size = 2 * dilation  

           
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.has_ranking:
            y = RankingFunc.apply(x, self.ranking_size)
#            y = RankingFunc.apply(x, self.ranking_size)
        if self.has_ranking and self.has_conv:
            return torch.cat((x, y), dim = 1)
        elif self.has_ranking:
            return y
        elif self.has_conv:
            return x
            
#%%
class DehazePyramid(BasicModule):
    def __init__(self, in_channel, out_channel, kernel_size, num, conv = True, ranking = False, dilation = False):
        """
        INPUT PARAMETERS
        ----------------------------------------------
        num: number of DehazeBlock in each pyramid
        conv: atrous convolution path
        ranking: ranking path
        """
        super().__init__()
        if dilation:
            self.dilation = [1, 2, 5, 11, 23]
        else:
            self.dilation = [1, 1, 1, 1, 1]
        self.db = nn.ModuleList()
        for i in range(num):
            self.db.append(DehazeBlock(in_channel, out_channel, kernel_size, dilation = self.dilation[i], 
                                       conv = conv, ranking = ranking))
        if conv and ranking:
            self.conv = nn.Conv2d(out_channel * num * 2, out_channel, (1, 1), 1, bias = True)
        else:
            self.conv = nn.Conv2d(out_channel * num, out_channel, (1, 1), 1, bias = True)
        self.num = num
        
    def forward(self, x):
        y = []
        for i in range(self.num):
            y.append(self.db[i](x) + y[i-1] if i != 0 else self.db[i](x))
        y = torch.cat(y, dim = 1)
        y = self.conv(y)
        return y

#%%
class DehazeNet(BasicModule):
    def __init__(self, kernel_size, rate_num, pyramid_num, conv = True, ranking = False, dilation = False):
        """
        INPUT PEREMETERS
        -------------------------------------------------
        rate_num: number of DehazeBlock in each DehazePyramid
        conv: atrous convolution path
        ranking: ranking path
        """
        super().__init__()
        self.net = nn.Sequential()
        if conv and ranking:
            self.net.add_module('DP1', DehazePyramid(6, 8, kernel_size, rate_num, conv, ranking, dilation))
        else:
            self.net.add_module('DP1', DehazePyramid(3, 8, kernel_size, rate_num, conv, ranking, dilation))
        for i in range(2, pyramid_num + 1):          
            self.net.add_module('BN' + str(i - 1), nn.BatchNorm2d(8))
            self.net.add_module('ReLU' + str(i - 1), nn.ReLU(inplace = True))
            if i == pyramid_num - 1:
                self.net.add_module('DP' + str(i), DehazePyramid(8, 3, kernel_size, rate_num, conv, ranking, dilation))
            else:
                self.net.add_module('DP' + str(i), DehazePyramid(8, 8, kernel_size, rate_num, conv, ranking, dilation))

        
    def forward(self, x):
        return self.net(x)