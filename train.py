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

#%%
def train(opt):
    #step1: model
    model = DehazeNet(opt.kernel_size, opt.rate_num, opt.conv, opt.ranking)
    if torch.cuda.is_available():
        model = model.cuda()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    
    #step2: dataset
    transform = T.Compose([
                    T.CenterCrop(opt.img_size),
                    T.ToTensor(),
                    T.Normalize(mean = [.5, .5, .5], std = [.5, .5, .5])
                ])
    train_set = DehazingSet(opt.train_data_root, transform)
    val_set = DehazingSet(opt.val_data_root, transform)
    train_dataloader = DataLoader(train_set, opt.batch_size, shuffle = True, num_workers = opt.num_workers)
    val_dataloader = DataLoader(val_set, opt.val_batch_size, shuffle = True, num_workers = opt.num_workers)
    
    #step3: Loss function and Optimizer
    criterion = nn.MSELoss().cuda()
    lr = opt.lr #current learning rate
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = opt.weight_decay)
    
    # metrics
    total_loss = 0
    previous_loss = 1e100
    
    model.train()  #train mode
    
    
    #step5: train
    for epoch in range(opt.max_epoch):
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
            
#            if (iteration + 1) % opt.display_iter == 0:
#                print("Loss at iteration {}: {}".format(iteration, loss))
#            if (iteration + 1) % opt.sample_iter == 0:
#                torchvision.utils.save_image(torch.cat((input_data / 2 + 0.5, target_data / 2 + 0.5, output_result / 2 + 0.5), dim = 0),
#                                             'output_sample/epoch{}.jpg'.format(epoch))
        
        if (epoch + 1) % opt.display_iter == 0:
            print("Loss at epoch {}: {}".format(epoch + 1, loss))
        if (epoch + 1) % opt.sample_iter == 0:
            torchvision.utils.save_image(torch.cat((input_data / 2 + 0.5, target_data / 2 + 0.5, output_result / 2 + 0.5), dim = 0),
                                         'output_sample/epoch{}.jpg'.format(epoch + 1))
            val_loss = val(model, val_dataloader)
            print("Val Set Loss at epoch {}: {}".format(epoch + 1, val_loss))
        #print("Training Set Loss at Epoch {}: {}".format(epoch, total_loss))
        #model.save(time.strftime('%m%d_%H:%M:%S') + '_Epoch:' + str(epoch) + '.pth')
        
        

        
#        #if loss does not decrease, decrease learning rate
#        if loss_meter.value()[0] > previous_loss:
#            lr = lr * opt.lr_decay
#            for param_group in optimizer.param_groups:
#                param_group['lr'] = lr
                
#%%       
def val(model, dataloader):
    model.eval() #evaluation mode
    
    loss_total = 0
    for iteration, (hazy_img, gt_img) in enumerate(dataloader):
        input_data = hazy_img
        target_data = gt_img
        if torch.cuda.is_available():
            input_data = input_data.cuda()
            target_data = target_data.cuda()
        
        output_result = model(input_data)
        
        #TODO: SSIM and PSNR test
        loss = nn.MSELoss()(output_result, target_data)
        loss_total += loss.detach()
    
    model.train() #back to train mode
    
    return loss_total


#%%
if __name__ == '__main__':
    config = Config()
    train(config)