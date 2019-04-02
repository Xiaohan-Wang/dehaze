class Config(object):
    debug_file = True
    
    train_data_root = 'dataset/training_set'
    val_data_root = 'dataset/val_set'
    test_data_root = 'dataset/test_set'
    output_sample = 'output_sample'
    load_model_path = None
    
    kernel_size = 3
    rate_num = 5
    pyramid_num = 4
    conv = True
    ranking = False
    
    img_size = 2
    batch_size = 4
    val_batch_size = 1
    num_workers = 4
    lr = 0.003 # initial learning rate
    weight_decay = 0.0001 
    
    max_epoch = 1500
    display_iter = 1
    sample_iter = 5
    
    lr_decay = 0.95
     