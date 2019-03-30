class Config(object):
    
    train_data_root = 'dataset/training_set'
    val_data_root = 'dataset/val_set'
    test_data_root = 'dataset/test_set'
    output_sample = 'output_sample'
    load_model_path = None
    
    kernel_size = 3
    rate_num = 5
    conv = True
    ranking = False
    
    batch_size = 1
    val_batch_size = 1
    num_workers = 0
    lr = 0.1 # initial learning rate
    weight_decay = 0.95 
    
    max_epoch = 100
    display_iter = 2
    sample_iter = 5
    
    lr_decay = 0.95
     