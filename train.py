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


#%%
def train(opt, vis):
    #step1: model
    model = DehazeNet(opt.layer_num, opt.channels, opt.kernel_size_num, opt.kernel_size)
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
                if opt.seg:
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
                print("Loss at epoch {} iteration {}: {}".format(epoch, iteration, loss))
                vis.line(X = torch.tensor([step]), Y = torch.tensor([loss]), win = 'train loss', update = 'append' if step > 0 else None)
            if step % opt.sample_iter == 0:
                torchvision.utils.save_image(torch.cat((trans_img, output_result), dim = 0), \
                                             opt.output_sample + '/epoch{}_iteration{}.jpg'.format(epoch, iteration), nrow = opt.batch_size)
            if os.path.exists(opt.debug_file):
                import ipdb
                ipdb.set_trace()
        
        training_loss = total_loss / (opt.train_num // opt.batch_size)
        sl.save_state(epoch, step, model.state_dict(), optimizer.state_dict())
        
        val_loss, val_dehazing_loss = val(model, val_dataloader, opt.seg, opt.at_method)

        vis.line(X = torch.tensor([globle_step]), Y = torch.tensor([training_loss]), win = 'val and train loss', update = 'append' if globle_step > 0 else None, name = 'train loss')
        vis.line(X = torch.tensor([globle_step]), Y = torch.tensor([val_loss]), win = 'val and train loss', update = 'append', name = 'Val loss')
        
        vis.line(X = torch.tensor([globle_step]), Y = torch.tensor([val_dehazing_loss]), win = "val dehazing result loss", updata = 'append' if globle_step > 0 else None)
        globle_step += 1
        
        #if loss does not decrease, decrease learning rate
        if training_loss >= previous_loss:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_decay
        previous_loss = training_loss

#%%       
def val(model, dataloader, has_seg, at_method):
    model.eval() #evaluation mode
    
    loss_total = 0
    dehazing_loss_total = 0
    img_num = 0
    for iteration, (hazy_img, trans_img, seg_img, gt_img) in enumerate(dataloader):
        if torch.cuda.is_available():
            hazy_img = hazy_img.cuda()
            trans_img = trans_img.cuda()
            if has_seg:
                seg_img = seg_img.cuda()
                input_data = torch.cat([hazy_img, seg_img], dim = 1)
            else:
                input_data = hazy_img
        
        output_result = model(input_data)
        loss = nn.MSELoss()(output_result, trans_img)
        loss_total += loss.detach()
        
        atmosphere = estimate_airlight(hazy_img.squeeze(0), output_result.squeeze(0), at_method)
        dehazing_result = (hazy_img - atmosphere * (1 - output_result)) / output_result
        dehazing_loss = nn.MSELoss()(dehazing_result, gt_img)
        dehazing_loss_total += dehazing_loss.detach()
        img_num += 1
    
    loss_avg = loss_total / img_num
    dehazing_loss_avg = dehazing_loss_total / img_num
    model.train() #back to train mode
    
    return loss_avg, dehazing_loss_avg
