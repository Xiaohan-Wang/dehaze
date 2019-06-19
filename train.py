# -*- coding: utf-8 -*-

import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from DehazeNet import DehazeNet
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torch import optim
import torchvision.utils
import torch
from config import Config
from DehazingSet import DehazingSet
import visdom
from utils import save_load as sl
from utils.airlight import estimate_airlight
import cv2
import numpy as np


#%%
def train(opt, vis):
    #step1: model
    model = DehazeNet(opt.layer_num, opt.in_channels,opt.out_channels, opt.kernel_size_num, opt.kernel_size)
    if torch.cuda.is_available():
        model = model.cuda()

    #step2: dataset
    train_val_set = DehazingSet(opt.train_data_root, opt.transform, True)
    sampler_train = SubsetRandomSampler(torch.randperm(opt.train_num))
    sampler_val = SubsetRandomSampler(torch.randperm(opt.val_num) + opt.train_num)
    train_dataloader = DataLoader(train_val_set, sampler = sampler_train, batch_size = opt.batch_size, num_workers = opt.num_workers, drop_last = True)
    val_dataloader = DataLoader(train_val_set, sampler = sampler_val, batch_size = opt.val_batch_size, num_workers = opt.num_workers)
    
    #step3: Loss function and Optimizer
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr = opt.lr, weight_decay = opt.weight_decay)
    
    if opt.load_model_path:
        model, optimizer, epoch_s, step_s = sl.load_state(opt.load_model_path, model, optimizer)
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr
    
    if not os.path.exists(opt.output_sample):
        os.mkdir(opt.ouput_sample)
    # metrics
    total_loss = 0
    previous_loss = 1
    
    model.train()  #train mode
    (globle_step, step) = (epoch_s + 1, step_s) if opt.load_model_path != None else (0, 0)   
    
    #step5: train
    for epoch in range(globle_step, opt.max_epoch):
        total_loss = 0

        for iteration, (hazy_img, trans_img, seg_img, _) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                hazy_img = hazy_img.cuda()
                trans_img = trans_img.cuda()
                if opt.has_seg:
                    seg_img = seg_img.cuda()
                    input_data = torch.cat([hazy_img, seg_img], dim = 1)
                else:
                    input_data = hazy_img
                
            output_result = model(input_data)
            loss = criterion(output_result, trans_img)
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.detach()
            
            step += 1
            
            if step % opt.display_iter == 0:
                print("Loss at epoch {} iteration {}: {}".format(epoch, iteration + 1, loss))
                vis.line(X = torch.tensor([step]), Y = torch.tensor([loss]), win = 'step train loss', update = 'append' if step > 0 else None)
            if step % opt.sample_iter == 0:
                temp_result = output_result.detach()
                for i in range(opt.batch_size):
                    guide = hazy_img[i, :, :, :].cpu().detach().numpy()
                    guide = np.transpose(guide, [1, 2, 0])
                    temp = cv2.ximgproc.guidedFilter(guide = guide, src = temp_result[i, :, :, :].squeeze().unsqueeze(2).cpu().detach().numpy(), radius = 10, eps = 0.001)
                    temp_result[i, :, :, :] = torch.tensor(temp).cuda().unsqueeze(0)
                torchvision.utils.save_image(torch.cat((trans_img, temp_result, output_result), dim = 0), \
                                             opt.output_sample + '/epoch{}_iteration{}.jpg'.format(epoch, iteration + 1), nrow = opt.batch_size)
            if os.path.exists(opt.debug_file):
                import ipdb
                ipdb.set_trace()
        
        training_loss = total_loss / (opt.train_num // opt.batch_size)
        sl.save_state(epoch, step, model.state_dict(), optimizer.state_dict())
        
        val_loss, val_dehazing_loss1, val_dehazing_loss2, val_dehazing_loss3, val_dehazing_loss4 = val(model, val_dataloader, opt, epoch)

#        vis.line(X = torch.tensor([globle_step]), Y = torch.tensor([training_loss]), win = 'train loss', update = 'append' if globle_step > 0 else None, name = 'without transmission similarity', opts = dict(xlable= 'number of epoch', ylable= 'MSE on training set', xtick = True, ytick = True, showlegend = True))
#        vis.line(X = torch.tensor([globle_step]), Y = torch.tensor([val_loss]), win = 'val loss', update = 'append' if globle_step > 0 else None, name = 'without transmission similarity', opts = dict(xlable= 'number of epoch', ylable= 'MSE on test set', xtick = True, ytick = True, showlegend = True))
        vis.line(X = torch.tensor([globle_step]), Y = torch.tensor([training_loss]), win = 'train loss', update = 'append', name = 'with transmission similarity', opts = dict(xlabel= 'number of epoch', ylabel= 'MSE on training set', xtick = True, ytick = True, showlegend = True))
        vis.line(X = torch.tensor([globle_step]), Y = torch.tensor([val_loss]), win = 'val loss', update = 'append', name = 'with transmission similarity', opts = dict(xlabel= 'number of epoch', ylabel= 'MSE on test set', xtick = True, ytick = True, showlegend = True))      
        vis.line(X = torch.tensor([globle_step]), Y = torch.tensor([val_dehazing_loss1]), win = "val dehazing result loss", update = 'append' if globle_step > 0 else None, name = opt.at_method1)
        vis.line(X = torch.tensor([globle_step]), Y = torch.tensor([val_dehazing_loss2]), win = "val dehazing result loss", update = 'append', name = opt.at_method2)
        vis.line(X = torch.tensor([globle_step]), Y = torch.tensor([val_dehazing_loss3]), win = "val dehazing result loss", update = 'append', name = opt.at_method3)
        vis.line(X = torch.tensor([globle_step]), Y = torch.tensor([val_dehazing_loss4]), win = "val dehazing result loss", update = 'append', name = opt.at_method4)
        globle_step += 1
        
        #if loss does not decrease, decrease learning rate
        if training_loss >= previous_loss:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_decay
        previous_loss = training_loss

#%%       
def val(model, dataloader, opt, epoch):
    model.eval() #evaluation mode
    
    loss_total = 0
    dehazing_loss_total1 = 0
    dehazing_loss_total2 = 0
    dehazing_loss_total3 = 0
    dehazing_loss_total4 = 0
    img_num = 0
    for iteration, (hazy_img, trans_img, seg_img, gt_img) in enumerate(dataloader):
        if torch.cuda.is_available():
            hazy_img = hazy_img.cuda()
            trans_img = trans_img.cuda()
            gt_img = gt_img.cuda()
            if opt.has_seg:
                seg_img = seg_img.cuda()
                input_data = torch.cat([hazy_img, seg_img], dim = 1)
            else:
                input_data = hazy_img
        
        output_result = model(input_data)
        guide = hazy_img.squeeze().cpu().detach().numpy()
        guide = np.transpose(guide, [1, 2, 0])
        output_result = cv2.ximgproc.guidedFilter(guide = guide, src = output_result.squeeze().cpu().detach().numpy(), radius = 20, eps = 0.000001)
        output_result = torch.tensor(output_result).cuda().unsqueeze(0)
        loss = nn.MSELoss()(output_result, trans_img)
        loss_total += loss.detach()
        
        atmosphere1 = estimate_airlight(hazy_img.squeeze(0), output_result.squeeze(0), opt.at_method1)
        atmosphere2 = estimate_airlight(hazy_img.squeeze(0), output_result.squeeze(0), opt.at_method2)
        atmosphere3 = estimate_airlight(hazy_img.squeeze(0), output_result.squeeze(0), opt.at_method3)
        atmosphere4 = estimate_airlight(hazy_img.squeeze(0), output_result.squeeze(0), opt.at_method4)
        a_t1 = torch.mm(atmosphere1.unsqueeze(1), (1 - output_result.squeeze(0).view(1, -1))).view(hazy_img.size())
        dehazing_result1 = (hazy_img - a_t1) / output_result
        a_t2 = torch.mm(atmosphere2.unsqueeze(1), (1 - output_result.squeeze(0).view(1, -1))).view(hazy_img.size())
        dehazing_result2 = (hazy_img - a_t2) / output_result
        a_t3 = torch.mm(atmosphere3.unsqueeze(1), (1 - output_result.squeeze(0).view(1, -1))).view(hazy_img.size())
        dehazing_result3 = (hazy_img - a_t3) / output_result
        a_t4 = torch.mm(atmosphere4.unsqueeze(1), (1 - output_result.squeeze(0).view(1, -1))).view(hazy_img.size())
        dehazing_result4 = (hazy_img - a_t4) / output_result
        dehazing_loss1 = nn.MSELoss()(dehazing_result1, gt_img)
        dehazing_loss_total1 += dehazing_loss1.detach()
        dehazing_loss2 = nn.MSELoss()(dehazing_result2, gt_img)
        dehazing_loss_total2 += dehazing_loss2.detach()
        dehazing_loss3 = nn.MSELoss()(dehazing_result3, gt_img)
        dehazing_loss_total3 += dehazing_loss3.detach()
        dehazing_loss4 = nn.MSELoss()(dehazing_result4, gt_img)
        dehazing_loss_total4 += dehazing_loss4.detach()
        img_num += 1
        if iteration % opt.result_sample_iter == 0:
            torchvision.utils.save_image(torch.cat((hazy_img, gt_img, dehazing_result1, dehazing_result2, dehazing_result3, dehazing_result4), dim = 0), \
                                         opt.dehazing_result_sample + '/epoch{}_iteration{}.jpg'.format(epoch, iteration), nrow = 6)            
    
    loss_avg = loss_total / img_num
    dehazing_loss_avg1 = dehazing_loss_total1 / img_num
    dehazing_loss_avg2 = dehazing_loss_total2 / img_num
    dehazing_loss_avg3 = dehazing_loss_total3 / img_num
    dehazing_loss_avg4 = dehazing_loss_total4 / img_num
    model.train() #back to train mode
    
    return loss_avg, dehazing_loss_avg1, dehazing_loss_avg2, dehazing_loss_avg3, dehazing_loss_avg4

if __name__ == '__main__':
    opt = Config()
    vis = visdom.Visdom(env = " with/without transmission similarity")
    train(opt, vis)