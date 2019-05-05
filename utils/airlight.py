# -*- coding: utf-8 -*-

import torch

def estimate_airlight(img, t, method):
    img = img.detach()
    t = t.detach()
    c, h, w = img.size()
    img = img.view(c, -1)
    max_I_3c, _ = torch.max(img, dim = 1)
    min_max = torch.min(max_I_3c)
    
    if method == "min_I":
        img = img.view(c, -1)
        t = t.view(1, -1)
        min_I, _ = torch.min(img, dim = 0)
#        _, index = torch.sort(min_I / torch.max(t, torch.Tensor([0.1]).cuda()))
        _, index = torch.sort(min_I)
#        _, index = torch.sort(min_I)
        index = index[0 : h * w // 50000]
#        index = index[0]
        atmosphere = min_I.unsqueeze(0)[:, index] / (1 - t[:, index]).cuda()
        atmosphere = torch.mean(atmosphere)
        at = torch.tensor([atmosphere, atmosphere, atmosphere]).cuda()
        at[at > min_max] = min_max
        at[at < 0.7] = 0.7
        return at
#    
    elif method == "max_min":
        img = img.view(c, -1)
        t = t.view(1, -1)
        min_I, _ = torch.min(img, dim = 0)
        max_I, _ = torch.max(img, dim = 0)
        diff = max_I - min_I
#        diff = diff / torch.max(t, torch.Tensor([0.1]).cuda())
#        diff = diff / t
#        diff = torch.log(diff) * min_I
        _, index = torch.sort(diff, descending = True)
        index = index[0 : h * w // 50000]
#        index = index[0]
#        atmosphere = min_I.unsqueeze(0)[:, index] /(1 - torch.max(t[:, index], torch.Tensor([0.1]).cuda()))
        atmosphere = min_I.unsqueeze(0)[:, index] /(1 - t[:, index].cuda())
        atmosphere = torch.mean(atmosphere)
        at = torch.tensor([atmosphere, atmosphere, atmosphere]).cuda()
        at[at > min_max] = min_max
        at[at < 0.7] = 0.7
        return at
#        max_I_3c[max_I_3c > atmosphere] = atmosphere
#        max_I_3c[max_I_3c < 0.75] = 0.75
#        return max_I_3c
#        if atmosphere > max_glo:
#            return torch.tensor([max_glo, max_glo, max_glo]).cuda()
#        elif atmosphere > 0.7:
#            return torch.tensor([atmosphere, atmosphere, atmosphere]).cuda()
#        else:
#            return torch.tensor([0.7, 0.7, 0.7]).cuda()
#        
    elif method == "dark_channel":
        min_I, _ = torch.min(img, dim = 0)
        dark_channel = torch.zeros((h, w))
        for i in range(h):
            for j in range(w):
                h_s = i - 2 if (i - 2) >= 0 else 0
                h_e = i + 2 if (i + 2) <= h else h
                w_s = j - 2 if (j - 2) >= 0 else 0
                w_e = j + 2 if (j + 2) <= w else w
                win = min_I[h_s:h_e, w_s:w_e]
                dark_channel[i][j] = win.contiguous().view(1, -1).min()
        _, index = dark_channel.view(-1).sort(descending = True)
        index = index[0 : h * w // 1000]
        img = img.view(c, -1)
        atmosphere = img[:, index]
        return torch.mean(atmosphere, dim = 1)
        
    elif method == "max_I":
#        img = img.view(c, -1)
#        max_I, _ = torch.max(img, dim = 1)
        return max_I_3c
    
    elif method == "min_t":
        img = img.view(c, -1)
        _, index = torch.sort(t.view(1, -1))
        index = index[0 : h * w // 1000]
        atmosphere = img[:, index]
        atmosphere = torch.mean(atmosphere)
        return torch.tensor([atmosphere, atmosphere, atmosphere]).cuda()
        
        
                
                
                
        
        
    