# -*- coding: utf-8 -*-

from torchvision import transforms as T

class Config(object):
    debug_file = 'debug_file'
    
    train_data_root = '/home/ws/datasets/ITS(training set)'
    test_data_root = '/home/ws/datasets/SOTS(Testing Set)/(indoor)nyuhaze500'
    output_sample = 'output_sample'
    load_model_path = None
    
    train_num = 12600
    val_num = 1390
    
    layer_num = 6 
    channels = [3, 8, 11, 11, 8, 1]
    kernel_size_num = [1, 1, 1, 1, 1]
    kernel_size = [3, 3, 3, 3, 3]
    
    seg = False
    at_method = "dark_channel"
    
    ori_img_size = (460, 620)
    batch_size = 4
    val_batch_size = 1
    num_workers = 4
    lr = 0.0003 # initial learning rate
    weight_decay = 0.0001 
    
    max_epoch = 1000
    display_iter = 10
    sample_iter = 200
    
    lr_decay = 0.95
    
    transform = T.Compose([
#                T.CenterCrop(ori_img_size),
                T.ToTensor()
            ])
     