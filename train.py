#!user/bin/env python3
#-*-coding:utf-8-*-

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from DehazeNet import DehazeNet
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torchvision.utils
import torch
from config import Config
from DehazingSet import DehazingSet
from torchvision import transforms as T
import time
import visdom
from utils import save_load as sl

#%%
def train(opt, vis):
    #step1: model
    model = DehazeNet(opt.kernel_size, opt.rate_num, opt.pyramid_num, opt.conv, opt.ranking)
    if torch.cuda.is_available():
        model = model.cuda()
    
    #step2: dataset
    train_set = DehazingSet(opt.train_data_root, opt.transform)
    val_set = DehazingSet(opt.val_data_root, opt.transform)
    train_dataloader = DataLoader(train_set, opt.batch_size, shuffle = True, num_workers = opt.num_workers)
    val_dataloader = DataLoader(val_set, opt.val_batch_size, shuffle = True, num_workers = opt.num_workers)
    
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

        for iteration, (hazy_img, gt_img) in enumerate(train_dataloader):

            input_data = hazy_img
            target_data = gt_img
            
            if torch.cuda.is_available():
                input_data = input_data.cuda()
                target_data = target_data.cuda()
                
            output_result = model(input_data)
            loss = criterion(output_result, target_data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.detach()
            
            step += 1
            
            if step % opt.display_iter == 0:
                print("Loss at epoch {} step {}: {}".format(epoch + 1, step, loss))
                vis.line(X = torch.tensor([step]), Y = torch.tensor([loss]), win = 'train loss', update = 'append' if step > 0 else None)
            if step % opt.sample_iter == 0:
                torchvision.utils.save_image(torch.cat((input_data / 2 + 0.5, target_data / 2 + 0.5, output_result / 2 + 0.5), dim = 0), \
                                             opt.output_sample + '/epoch{}_step{}.jpg'.format(epoch + 1, step), nrow = 4)
            if os.path.exists(opt.debug_file):
                import ipdb
                ipdb.set_trace()
        
        training_loss = total_loss / (len(train_set) // opt.batch_size)
#        print("Training Set Loss at Epoch {}: {}".format(epoch, training_loss))
        sl.save_state(epoch, step, model.state_dict(), optimizer.state_dict())
        
        val_loss = val(model, val_dataloader)
#        print("Val Set Loss at epoch {} : {}".format(epoch + 1, val_loss))

        vis.line(X = torch.tensor([globle_step]), Y = torch.tensor([training_loss]), win = 'val and train loss', update = 'append' if globle_step > 0 else None, name = 'train loss')
        vis.line(X = torch.tensor([globle_step]), Y = torch.tensor([val_loss]), win = 'val and train loss', update = 'append' if globle_step > 0 else None, name = 'Val loss')

        globle_step += 1
        
        #if loss does not decrease, decrease learning rate
        if training_loss >= previous_loss:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_decay
        previous_loss = training_loss
                
#%%       
def val(model, dataloader):
    model.eval() #evaluation mode
    
    loss_total = 0
    img_num = 0
    for iteration, (hazy_img, gt_img) in enumerate(dataloader):
        input_data = hazy_img
        target_data = gt_img
        if torch.cuda.is_available():
            input_data = input_data.cuda()
            target_data = target_data.cuda()
        
        output_result = model(input_data)
        
        loss = nn.MSELoss()(output_result, target_data)
        loss_total += loss.detach()
        img_num += 1
    
    loss_avg = loss_total / img_num
    model.train() #back to train mode
    
    return loss_avg


#%%
if __name__ == '__main__':
    config = Config()
    vis = visdom.Visdom(env = "0402")
    train(config, vis)