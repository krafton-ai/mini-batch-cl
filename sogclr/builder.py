# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .simclr_utils import NTXent
import numpy as np

class SimCLR(nn.Module):
    """
    Build a SimCLR-based model with a base encoder, and two MLPs
   
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=2048, T=0.1, loss_type='dcl', N=50000, num_proj_layers=2, device=None):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(SimCLR, self).__init__()
        self.T = T
        self.N = N
        self.loss_type = loss_type
        
        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        
        # build non-linear projection heads
        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        # simclr
        self.simclr_criteria = NTXent(tau=T)

        # sogclr 
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device 
        
        # for DCL
        self.u = torch.zeros(N).reshape(-1, 1) #.to(self.device) 
        self.LARGE_NUM = 1e9


    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass
    
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def dynamic_contrastive_loss(self, hidden1, hidden2, index=None, gamma=0.9, distributed=True):
        # Get (normalized) hidden1 and hidden2.
        hidden1, hidden2 = F.normalize(hidden1, p=2, dim=1), F.normalize(hidden2, p=2, dim=1)
        batch_size = hidden1.shape[0]
        
        # Gather hidden1/hidden2 across replicas and create local labels.
        if distributed:  
           hidden1_large = torch.cat(all_gather_layer.apply(hidden1), dim=0) # why concat_all_gather()
           hidden2_large =  torch.cat(all_gather_layer.apply(hidden2), dim=0)
           enlarged_batch_size = hidden1_large.shape[0]

           labels_idx = (torch.arange(batch_size, dtype=torch.long) + batch_size  * torch.distributed.get_rank()).to(self.device) 
           labels = F.one_hot(labels_idx, enlarged_batch_size*2).to(self.device) 
           masks  = F.one_hot(labels_idx, enlarged_batch_size).to(self.device) 
           batch_size = enlarged_batch_size
        else:
           hidden1_large = hidden1
           hidden2_large = hidden2
           labels = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size * 2).to(self.device) 
           masks  = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size).to(self.device) 

        logits_aa = torch.matmul(hidden1, hidden1_large.T)
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.T)
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.T)
        logits_ba = torch.matmul(hidden2, hidden1_large.T)

        #  SogCLR
        neg_mask = 1-labels
        logits_ab_aa = torch.cat([logits_ab, logits_aa ], 1)
        logits_ba_bb = torch.cat([logits_ba, logits_bb ], 1)
      
        neg_logits1 = torch.exp(logits_ab_aa /self.T)*neg_mask   #(B, 2B)
        neg_logits2 = torch.exp(logits_ba_bb /self.T)*neg_mask

        # u init    
        if self.u[index.cpu()].sum() == 0:
            gamma = 1
            
        u1 = (1 - gamma) * self.u[index.cpu()].cuda() + gamma * torch.sum(neg_logits1, dim=1, keepdim=True)/(2*(batch_size-1))
        u2 = (1 - gamma) * self.u[index.cpu()].cuda() + gamma * torch.sum(neg_logits2, dim=1, keepdim=True)/(2*(batch_size-1))

        # this sync on all devices (since "hidden" are gathering from all devices)
        if distributed:
           u1_large = concat_all_gather(u1)
           u2_large = concat_all_gather(u2)
           index_large = concat_all_gather(index)
           self.u[index_large.cpu()] = (u1_large.detach().cpu() + u2_large.detach().cpu())/2 
        else:
           self.u[index.cpu()] = (u1.detach().cpu() + u2.detach().cpu())/2 

        p_neg_weights1 = (neg_logits1/u1).detach()
        p_neg_weights2 = (neg_logits2/u2).detach()

        def softmax_cross_entropy_with_logits(labels, logits, weights):
            expsum_neg_logits = torch.sum(weights*logits, dim=1, keepdim=True)/(2*(batch_size-1))
            normalized_logits = logits - expsum_neg_logits
            return -torch.sum(labels * normalized_logits, dim=1)

        loss_a = softmax_cross_entropy_with_logits(labels, logits_ab_aa, p_neg_weights1)
        loss_b = softmax_cross_entropy_with_logits(labels, logits_ba_bb, p_neg_weights2)
        loss = (loss_a + loss_b).mean()

        return loss

    def hcl_loss(self, out_1, out_2, temperature, batch_size, tau_plus=0.1, beta=1.0, estimator='hard'):
        """Reference : https://github.com/joshr17/HCL/blob/main/image/main.py#L29"""

        device = out_1.device

        # Get (normalized) out_1, out_2
        out_1, out_2 = F.normalize(out_1, p=2, dim=1), F.normalize(out_2, p=2, dim=1)

        def get_negative_mask(batch_size):
            negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
            for i in range(batch_size):
                negative_mask[i, i] = 0
                negative_mask[i, i + batch_size] = 0

            negative_mask = torch.cat((negative_mask, negative_mask), 0)
            return negative_mask

        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        old_neg = neg.clone()
        mask = get_negative_mask(batch_size).to(device)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)
        
        # negative samples similarity scoring
        if estimator=='hard':
            N = batch_size * 2 - 2
            imp = (beta* neg.log()).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        elif estimator=='easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
            
        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()

        return loss

    def get_features(self, x1, x2):
        return self.base_encoder(x1), self.base_encoder(x2)

    def forward(self, x1, x2, index, gamma, args, wo_loss=False):
        """
        Input:
            x1: first views of images
            x2: second views of images
            index: index of image
            gamma: moving average of sogclr 
        Output:
            loss
        """
        # compute features
        h1, h2 = self.get_features(x1, x2)
        if wo_loss:
            return h1, h2
        if args.objective_type == 'sim':
            loss = self.simclr_criteria(torch.cat([h1, h2], dim=0))
        elif args.objective_type == 'sog':
            loss = self.dynamic_contrastive_loss(h1, h2, index, gamma)
        elif args.objective_type == 'hcl':
            loss = self.hcl_loss(h1, h2, temperature=args.t, batch_size=args.batch_size,
                                 tau_plus=args.hcl_tau_plus, beta=args.hcl_beta, estimator='hard')
        else:
            raise NotImplemented

        return loss


class SimCLR_ResNet(SimCLR):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim, num_proj_layers=2):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc  # remove original fc layer
            
        # projectors
        # TODO: increase number of mlp layers 
        self.base_encoder.fc = self._build_mlp(num_proj_layers, hidden_dim, mlp_dim, dim)


class SimCLR_CLIP(SimCLR):
    def __init__(self, open_clip_args, device=None):
        super(SimCLR, self).__init__()

        from open_clip import create_model_and_transforms, trace_model

        model, preprocess_train, preprocess_val = create_model_and_transforms(
            open_clip_args.model,
            open_clip_args.pretrained,
            precision=open_clip_args.precision,
            device=open_clip_args.device,
            jit=open_clip_args.torchscript,
            frozen=open_clip_args.frozen,
            proj_type=open_clip_args.proj_type,
            force_quick_gelu=open_clip_args.force_quick_gelu,
            pretrained_image=open_clip_args.pretrained_image,
            image_mean=open_clip_args.image_mean,
            image_std=open_clip_args.image_std,
        )
        self.base_encoder = model
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val

        # sogclr 
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device 

    # def get_features(self, x1, x2):


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class all_gather_layer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return grad_out
