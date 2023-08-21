open_clip_dict = {
    #################################
    # about model
    #################################
    "model": "RN50",
    "pretrained": "openai",
    "precision": "amp",
    "device": "cuda:0",  ## TODO: you should consider device and rank.
    "torchscript": False,
    "frozen": False,     # Unfrozen
    "proj_type": "mlp",
    "force_quick_gelu": False,  #
    "pretrained_image": False,  #
    "image_mean": None,  # https://git.projectbro.com/deep-learning/project-clap/open_clip/-/blame/dev_clif_wandb_exp/src/training/params.py#L202
    "image_std": None,
    "trace": False,
    ###################################
    # about learning
    ###################################
    # "wd": 0.1,
    # "warmup": 200,  # This is not used. (We use ArgParse warmup-epochs)
    # "lr": 5e-3,     # This is not used.   (We use ArgParse lr)
    # "epochs": 30,   # This is not used.  
    "horovod": False,
    # "batch_size": 200,  # This is not used.
    # "workers": 8,   # This is not used.
    ##################################
    # about data
    ##################################
    "train_data": "/data/clap/cooked_data/COCO/Sep14/train_one_caption.csv",
    "val_data": "/data/clap/cooked_data/COCO/Sep14/val_one_caption.csv",
    "val_on_train_data": "/data/clap/cooked_data/COCO/Sep14/train_one_caption_5000.csv",
    "csv_img_key": "image_path",
    "csv_caption_key": "caption",
    "csv_separator": "|",
    "dataset_type": "auto",
    "max_dataset_size": None,
    "distributed": False,
    "fixed_batch": False,
    ###################################
    # Infonce Loss
    ###################################
    "local_loss": False,
    "gather_with_grad": False,
    "rank": 0,
    "world_size": 1,
    "horovod": False,
    "bigbatch": True
}