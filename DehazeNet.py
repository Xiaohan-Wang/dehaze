import torch
import torch.nn as nn
from torch.autograd import Function

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
            name = strftime("%m%d_%H:%M:%S.pth")
        name = prefix + name
        torch.save(self.state_dict(), name)
        return name         

class RankingFunc(Function): 
    # TODO: check whether it's correct or not
    
    @staticmethod
    def forward(ctx, x, ranking_size): 
        c, h, w = x.size()
        ctx.y_index = torch.arange(0, h*w).resize(h, w).unsqueeze(0).repeat(c, 1, 1)
        ctx.ranking_size = ranking_size
        
        h_num = h // ranking_size
        w_num = w // ranking_size
        
        ipdb.set_trace()
        #TODO: what is the dimension of x????
        for i in range(h_num):
            for j in range(w_num):
                temp1 = x[:, :, i * ranking_size : (i + 1) * ranking_size, j * ranking_size : (j + 1) * ranking_size]
                temp2 = ctx.y_index[:, :, i * ranking_size : (i + 1) * ranking_size, j * ranking_size : (j + 1) * ranking_size]
                value, index = torch.sort(temp1.resize(c, ranking_size ** 2))
                temp1 = value.resize(c, ranking_size, ranking_size)
                temp2 = temp2.resize(c, ranking_size ** 2)[index].resize(c, ranking_size, ranking_size)
        return x
    
    @staticmethod
    def backward(ctx, grad_y):
        c, h, w = grad_y.size()[0], grad_y.size()[1], grad_y.size()[2]
        grad_y = grad_y.resize(c, ctx.ranking_size ** 2)
        _, index = ctx.y_index.resize(c, ctx.ranking_size ** 2).sort()
        for i in range(c):
            grad_y[c] = grad_y[c][index[c]]
        grad_y = grad_y.resize(c, h, w)
        return grad_y, None
    
def ranking_func(x, ranking_size):      
    h_num = x.size()[2] // ranking_size
    w_num = x.size()[3] // ranking_size
    for i in range(h_num):
        for j in range(w_num):
            temp = x[:, :, i * ranking_size : (i + 1) * ranking_size, j * ranking_size : (j + 1) * ranking_size]
            value = torch.sort(temp.resize(y.size()[0], y.size()[1], ranking_size * ranking_size))[0]
            temp = value.resize(y.size()[0], y.size()[1], ranking_size, ranking_size)
    return x

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
            y = RankingFunc.Apply(x, self.raning_size)
        if self.has_ranking and self.has_conv:
            return torch.cat((x, y), dim = 1)
        elif self.has_ranking:
            return y
        elif self.has_conv:
            return x
            

class DehazePyramid(BasicModule):
    def __init__(self, in_channel, out_channel, kernel_size, num, conv = True, ranking = False):
        """
        INPUT PARAMETERS
        ----------------------------------------------
        num: number of DehazeBlock in each pyramid
        conv: atrous convolution path
        ranking: ranking path
        """
        super().__init__()
        self.db = nn.ModuleList()
        for i in range(num):
            self.db.append(DehazeBlock(in_channel, out_channel, kernel_size, dilation = 2**i, 
                                       conv = conv, ranking = ranking))
        self.conv = nn.Conv2d(out_channel * num, out_channel, (1, 1), 1, bias = True)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace = True)
        
        self.num = num
        
    def forward(self, x):
        y = []
        for i in range(self.num):
            y.append(self.db[i](x))
        y = torch.cat(y, dim = 1)
        y = self.relu(self.bn(self.conv(y)))
        return y

class DehazeNet(BasicModule):
    def __init__(self, kernel_size, rate_num, conv = True, ranking = False):
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
            self.net.add_module('DP1', DehazePyramid(6, 16, kernel_size, rate_num, conv, ranking))
        else:
            self.net.add_module('DP1', DehazePyramid(3, 16, kernel_size, rate_num, conv, ranking))
        self.net.add_module('DP2', DehazePyramid(16, 32, kernel_size, rate_num, conv, ranking))
        self.net.add_module('DP3', DehazePyramid(32, 64, kernel_size, rate_num, conv, ranking))
        self.net.add_module('DP4', DehazePyramid(64, 64, kernel_size, rate_num, conv, ranking))
        self.net.add_module('DP5', DehazePyramid(64, 3, kernel_size, rate_num, conv, ranking))
        
    def forward(self, x):
        return self.net(x)