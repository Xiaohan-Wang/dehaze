class Config(object):
    
    train_data_root = 'dataset/training_set'
    val_data_root = 'dataset/val_set'
    test_data_root = 'dataset/test_set'
    output_sample = 'output_sample'
    load_model_path = None
    
    kernel_size = 3
    rate_num = 5
    conv = True
    ranking = True
    
    img_size = 4
    batch_size = 1
    val_batch_size = 1
    num_workers = 4
    lr = 0.003 # initial learning rate
    weight_decay = 0.0001 
    
    max_epoch = 1500
    display_iter = 25
    sample_iter = 50
    
    lr_decay = 0.95
     