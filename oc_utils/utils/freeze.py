import torch


def freeze_all(parameters, named=True):
    if named:
        for name, param in parameters:
            param.requires_grad = False
    else:
        for param in parameters:
            param.requires_grad = False


def unfreeze_all(parameters, named=True):
    if named:
        for name, param in parameters:
            param.requires_grad = True
    else:
        for param in parameters:
            param.requires_grad = True
