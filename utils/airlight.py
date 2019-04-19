# -*- coding: utf-8 -*-

import torch

def estimate_airlight(img, t, method):
    c, h, w = img.size()
    
    if method == "min_I":
        img = img.view(c, -1)
        t = t.view(1, -1)
        min_I = torch.min(img, dim = 0)
        _, index = torch.sort(min_I)
        index = index[0 : h * w // 1000]
        atmosphere = min_I[index] / t[index]
        atmosphere = torch.mean(atmosphere)
        return torch.tensor([atmosphere, atmosphere, atmosphere])
    
    elif method == "max_min":
        img = img.view(c, -1)
        t = t.view(1, -1)
        min_I = torch.min(img, dim = 0)
        max_I = torch.max(img, dim = 0)
        diff = max_I - min_I
        _, index = torch.sort(diff, descending = True)
        index = index[0 : h * w // 1000]
        atmosphere = min_I[index] / t[index] 
        atmosphere = torch.mean(atmosphere)
        return torch.tensor([atmosphere, atmosphere, atmosphere])
        
    elif method == "dark_channel":
        min_I = torch.min(img, dim = 0)
        dark_channel = torch.tensor(h, w)
        for i in range(h):
            for j in range(w):
                h_s = i - 2 if (i - 2) >= 0 else 0
                h_e = i + 2 if (i + 2) <= h else h
                w_s = j - 2 if (j - 2) >= 0 else 0
                w_e = j + 2 if (j + 2) <= w else w
                win = min_I[h_s:h_e, w_s:w_e]
                dark_channel[i][j] = win.view(1, -1).min()
        _, index = dark_channel.view(-1).sort(descending = True)
        img = img.view(c, -1)
        atmosphere = img(:, index)
        return torch.mean(atmosphere, dim = 1)
        
                
                
                
        
        
    