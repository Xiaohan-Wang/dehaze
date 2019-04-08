from torchvision import transforms as T

class Config(object):
    debug_file = 'debug_file'
    
    train_data_root = '/home/ws/datasets/ITS(training set)'
#    val_data_root = '/home/ws/datasets/ITS(training set)'
    test_data_root = '/home/ws/datasets/SOTS(Testing Set)/(indoor)nyuhaze500'
    output_sample = 'output_sample'
    load_model_path = None
    
    train_num = 12600
    val_num = 1390
    
    kernel_size = 3
    rate_num = 1
    pyramid_num = 5
    conv = True
    ranking = False
    dilation = False
    
    ori_img_size = (460, 620)
    resized_img_size = (int(460/4), int(620/4))
    batch_size = 4
    val_batch_size = 1
    num_workers = 4
    lr = 0.0001 # initial learning rate
    weight_decay = 0.0001 
#    grad_clip_norm = 0.1
#    momentum = 0.9
    
    max_epoch = 1000
    display_iter = 10
    sample_iter = 20
    
    lr_decay = 0.95
    
    transform = T.Compose([
#                T.CenterCrop(ori_img_size),
#                T.Resize(resized_img_size),
                T.ToTensor(),
#                T.Normalize(mean = [.5, .5, .5], std = [.5, .5, .5])
            ])
     