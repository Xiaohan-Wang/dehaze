class Config(object):
    debug_file = 'debug_file'
    
    train_data_root = '/home/ws/datasets/ITS(training set)'
    val_data_root = '/home/ws/datasets/SOTS(Testing Set)/(indoor)nyuhaze500'
    test_data_root = 'dataset/test_set'
    output_sample = 'output_sample'
    load_model_path = 'checkpoints/0402_22:27:10_Epoch1.pth'
    
    kernel_size = 3
    rate_num = 5
    pyramid_num = 4
    conv = True
    ranking = False
    
    img_size = 512
    batch_size = 4
    val_batch_size = 1
    num_workers = 1
    lr = 0.001 # initial learning rate
    weight_decay = 0.0001 
    
    max_epoch = 100
    display_iter = 50
    sample_iter = 50
    
    lr_decay = 0.95
     