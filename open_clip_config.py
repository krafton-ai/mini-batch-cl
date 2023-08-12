open_clip_dict = {
    # about model
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
    # about learning
    "wd": 0.1,
    "warmup": 200,
    #"batch_size": 200,
    "lr": 3e-3,
    "epochs": 30,
    "horovod": False
}