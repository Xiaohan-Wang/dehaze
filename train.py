
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import ipdb

from DehazeNet import DehazeNet
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import config
import torchvision.utils
import torch
from config import Config
from DehazingSet import DehazingSet

def train(opt):
    #step1: model
    model = DehazeNet(opt.kernel_size, opt.rate_num, opt.conv, opt.ranking).cuda()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    
    #step2: dataset
    train_set = DehazingSet(opt.train_data_root)
    val_set = DehazingSet(opt.val_data_root)
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

            input_data = hazy_img.cuda()
            target_data = gt_img.cuda()
            
            print("iteration {} (before back): {}".format(iteration, torch.cuda.memory_allocated()/10e6))
            output_result = model(input_data)
            loss = criterion(output_result, target_data)
            
            optimizer.zero_grad()
            loss.backward()
            print("iteration {} (after back): {}".format(iteration, torch.cuda.memory_allocated()/10e6))
            ipdb.set_trace()
            optimizer.step()
            
            total_loss += loss.detach()
            
            if (iteration + 1) % opt.display_iter == 0:
                print("Loss at iteration {}: {}".format(iteration, loss))
            if (iteration + 1) == len(train_dataloader):
                torchvision.utils.save_image(torch.cat((input_data.data, target_data.data, output_result.data), dim = 0))
                
        print("Training Set Loss at Epoch {}: {}".format(epoch, total_loss))
        model.save(strftime('%m%d_%H:%M:%S') + '_Epoch:' + epoch)
        
        
        val_loss = val(model, val_dataloader)
        print("Val Set Loss at Epoch {}: {}".format(epoch, val_loss))
        
        #if loss does not decrease, decrease learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
       
                  

def val(model, dataloader):
    model.eval() #evaluation mode
    
    loss_meter = meter.AverageValueMeter()
    for iteration, (hazy_img, gt_img) in enumerate(dataloader):
        input_data = hazy_img.cuda()
        target_data = gt_img.cuda()
        
        output_result = model(input_data)
        
        #TODO: SSIM and PSNR test
        loss = nn.MSELoss()(input_data, target_data)
        loss_meter.add(loss.data[0])
    
    model.train() #back to train mode
    
    return loss_meter.value()[0]

if __name__ == '__main__':
    config = Config()
    train(config)