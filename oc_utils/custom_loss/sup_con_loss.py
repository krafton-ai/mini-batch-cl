from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        # NOTE: temperature is already captured using logit_scale
        # But we still set it to 0.1 because it seems to help convergence
        self.temperature = temperature

    def forward(self, logits, labels=None, mask=None):
        device = (torch.device('cuda')
                  if logits.is_cuda
                  else torch.device('cpu'))

        batch_size = logits.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(f'Num of labels does not match num of features {labels.shape[0]} / {batch_size}')
            mask = torch.eq(labels, labels.T).float().to(device)  # mask : 1 as same label, 0 as different label
        else:
            mask = mask.float().to(device)

        logits = normalize(logits) / self.temperature

        # project logits down to (-100, 100) to prevent them from exploding
        with torch.no_grad():
            logits.data = torch.clamp(logits.data, min=-100.0, max=100.0)

        # compute log_prob
        # according to pytorch: https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html
        # this is more numerically stable than manually computing softmax and taking log
        log_prob = F.log_softmax(logits, dim=-1)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos.mean()

        return loss


# normalize rows of given tensor
def normalize(x):
    return F.normalize(x, dim=-1)
