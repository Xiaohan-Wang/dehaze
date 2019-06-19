# -*- coding: utf-8 -*-

from torchvision import transforms as T

class Config(object):
    debug_file = 'debug_file'
    
    train_data_root = '/home/ws/datasets/ITS(training set)'
    test_data_root = '/home/ws/datasets/SOTS(Testing Set)/(indoor)nyuhaze500'
    output_sample = 'output_sample'
    dehazing_result_sample = 'dehazing_result_sample'
    test_result_mm = 'test_result_mm'
    test_result_mI = 'test_result_mI'
    test_result_maxI = 'test_result_maxI'
    trans = "trans"
    refined_trans = "refined_trans"
    load_model_path = None
#    load_model_path = 'checkpoints/0503_13:06:07_epoch70_step110250.pth'
    
    train_num = 12600
    val_num = 1390
#    train_num = 100
#    val_num = 40
    
    layer_num = 6 
    in_channels = [4, 4, 4, 4, 4]
#    in_channels = [3, 3, 3, 3, 3]
    out_channels = [3, 3, 3, 3, 1]
    kernel_size_num = [1, 1, 1, 1, 1]
    kernel_size = [[7], [7], [7], [7], [7]]
    
    has_seg = True
    at_method1 = "max_I"
    at_method2 = "min_I"
    at_method3 = "max_min"
    at_method4 = "min_t"
    
    ori_img_size = (460, 620)
    batch_size = 8
    val_batch_size = 1
    num_workers = 0
    
    lr = 0.0001 # initial learning rate
    weight_decay = 0.0001 
    
    max_epoch = 20
    display_iter = 10
    sample_iter = 100
    result_sample_iter = 50
    
    lr_decay = 0.95
    
    transform = T.Compose([
#                T.CenterCrop(ori_img_size),
                T.ToTensor()
            ])
     