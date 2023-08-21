import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
from datetime import datetime
from time_utils import *
from memory_utils import *
import copy

from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from collections import defaultdict

#@title Convert a flat list out of a list of lists
def flatten(l):
    return [item for sublist in l for item in sublist]



def cosine_distance(A, B):
    # https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
    assert A.shape[1] == B.shape[1]
    A = A / np.linalg.norm(A, axis=-1, keepdims=True)
    B = B / np.linalg.norm(B, axis=-1, keepdims=True)
    cos_sim = A @ B.T
    return 1 - cos_sim




#@title CLIP loss (negative log likelihood)
def CLIP_loss(z): # negative log likelihood
    z_exp = np.exp(z)
    tmp = 0 # (positive) log likelihood
    for i in range(z.shape[0]):
        tmp += 2*z[i,i] # numerator
        tmp -= np.log(np.sum(z_exp[i,:])) # x to y classification
        tmp -= np.log(np.sum(z_exp[:,i])) # y to x classification
    tmp = tmp / z.shape[0] / 2  # average
    return -tmp # make it negative


#@title CLIP loss (negative log likelihood)
def CLIP_loss_torch(z): # negative log likelihood
  z_exp = torch.exp(z)
  tmp = 0 # (positive) log likelihood
  for i in range(z.shape[0]):
    tmp += 2*z[i,i] # numerator
    tmp -= torch.log(torch.sum(z_exp[i,:])) # x to y classification
    tmp -= torch.log(torch.sum(z_exp[:,i])) # y to x classification
  tmp = tmp / z.shape[0] / 2  # average
  return -tmp # make it negative


#@title CLIP loss on a chosen subset
def CLIP_loss_on_filtered_set(z_full, chosen_idx):
    tmp = z_full[chosen_idx, :]
    z = tmp[:, chosen_idx]
    if isinstance(z_full, torch.Tensor):
        return CLIP_loss_torch(z)
    else:
        return CLIP_loss(z)


def cosine_distance(A, B):
    # https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
    assert A.shape[1] == B.shape[1]
    A = A / np.linalg.norm(A, axis=-1, keepdims=True)
    B = B / np.linalg.norm(B, axis=-1, keepdims=True)
    cos_sim = A @ B.T
    return 1 - cos_sim


#@logging_time
def CLIP_loss_dp(z, z_exp, numerator_sum, row_exp_sum, col_exp_sum): # time complexity : O(B)
    """ CLIP loss with pre-calculated numerator, denominator """

    tmp = 0 # (positive) log likelihood
    tmp += 2*numerator_sum # previous numerators' sum
    tmp += 2*z[-1, -1]     # newcomer's numerator

    for i in range(z.shape[0] - 1):  # previous denominators + newcomer
        tmp -= np.log(row_exp_sum[i, :] + z_exp[i, -1])
        tmp -= np.log(col_exp_sum[:, i] + z_exp[-1, i])
    tmp -= np.log(np.sum(z_exp[-1, :]))  # newcomer's denominator
    tmp -= np.log(np.sum(z_exp[:, -1])) 

    return -tmp  # make it negative


def CLIP_loss_on_filtered_set_dp(z, z_exp, chosen_inds, numerator_sum, row_exp_sum, col_exp_sum):
    """CLIP loss on filtered set (using chosen_indices) with memoization """

    # Get chosen_z, chosen_z_exp  # time complexity : O(B) * one indexing access time
    tmp = z[chosen_inds, :]
    chosen_z = tmp[:, chosen_inds]
    tmp = z_exp[chosen_inds, :]
    chosen_z_exp = tmp[:, chosen_inds]

    # Get CLIP loss on chosen_z, chosen_z_exp  # time complextiy : O(B)
    start_time = datetime.now()
    loss = CLIP_loss_dp(chosen_z, chosen_z_exp, numerator_sum, row_exp_sum, col_exp_sum)
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    return loss, elapsed_time.total_seconds(), elapsed_time.total_seconds()




# @logging_time
def create_inverse_greedy_batches_with_K(N: int, B: int, 
    x_features: torch.Tensor, y_features: torch.Tensor, exp_logit_scale: torch.Tensor, 
    num_rand: int = None,
    z: torch.Tensor = None, z_exp: torch.Tensor = None, 
    epoch: int = None, 
    device: str = 'cuda',
    K: int = 1,
    D: int = 1024,
    max_B: int = None,
    max_n_batch: int = None):
    """
    Time complexity : O ( N * (N // B) * (B * f(N)) ) = O (N^2 f(N))
    """

    n_batch = N // B
    if max_n_batch is not None:
        n_batch = max_n_batch
        assert max_B is not None
    if num_rand != None and num_rand != 0:
        K_pivot = num_rand # 초기에 채워진 개수
    else:
        K_pivot = K
    K_loop = K # 채울 개수

    pivots = np.random.choice(N, n_batch * K_pivot, replace=False).reshape(n_batch, K_pivot)  # (N//B, K)
    batch_idx_list = [idx.tolist() for idx in pivots]                                         # (N//B, K)

    if num_rand == B:
        return batch_idx_list

    X = x_features[pivots.reshape(-1)].reshape(n_batch, K_pivot, -1)  # (N//B, K_pivot, D) --eventually--> (N//B, B, D)
    Y = y_features[pivots.reshape(-1)].reshape(n_batch, K_pivot, -1)  # (N//B, K_pivot, D) --eventually--> (N//B, B, D)
    # remain_idx = list(np.setdiff1d(np.arange(N), pivots.reshape(-1)))
    remain_idx = np.setdiff1d(np.arange(N), pivots.reshape(-1))
                                    
    pivot_z = exp_logit_scale * torch.einsum('nkd,nmd->nkm', X, Y)  # (N//B, K_pivot, K_pivot)  
    pivot_z_diagonal = pivot_z.diagonal(offset=0, dim1=-1, dim2=-2)  # (N//B, K_pivot)
    pivot_z_exp = torch.exp(pivot_z)  # (N//B, K_pivot, K)

    numerator_sum = torch.sum(pivot_z_diagonal.clone(), dim=-1)  # (N//B)
    row_exp_sum = torch.sum(pivot_z_exp.clone(), dim=2, keepdim=True)  # (N//B, K_pivot, 1) --eventually--> (N//B, B, 1)
    col_exp_sum = torch.sum(pivot_z_exp.clone(), dim=1, keepdim=True)  # (N//B, 1, K_pivot) --eventually--> (N//B, 1, B)

    # Initialize 
    X_temp = torch.zeros(n_batch, K_loop, D).to(device)
    Y_temp = torch.zeros(n_batch, K_loop, D).to(device)
    row_exp_sum_temp = torch.zeros(n_batch, K_loop, 1).to(device)
    col_exp_sum_temp = torch.zeros(n_batch, 1, K_loop).to(device)
    is_used_batch = torch.zeros(n_batch).to(device)
    
    count = 0
    temp_cnt = 0
    start = time.time()
    # print("Create Inverse Greedy Batches...")
    last_loop = False
    whole_loop = len(remain_idx) // K_loop
    # for temp_i in tqdm(range(whole_loop), total=B*(n_batch if n_batch is None else n_batch)):
    for temp_i in range(whole_loop):

        # start_time = time.time()
        # temp_cnt += time.time() - start_time
        # if count == 10:
        #     print(f"ig_avg_time : {temp_cnt / count}")
        #     exit()
        
        if len(remain_idx) == 0:
            break
        newcomer_idx = np.random.choice(remain_idx, K_loop)  # (K)
        x_new = x_features[newcomer_idx]  # (K, D)
        y_new = y_features[newcomer_idx]  # (K, D)

        ##############################################################
        # [1] numerator
        # Preparation for nuemrator
        xy_new = exp_logit_scale * torch.einsum('kd,ld->kl', x_new, y_new)  # (K, K)
        xy_new_diagonal = xy_new.diagonal(offset=0, dim1=-1, dim2=-2)  # (K)
        # Update numerator sum 
        numerator_sum_updated = numerator_sum + torch.sum(xy_new_diagonal, dim=0)  # (N//B)
        # Loss from numerators
        tmp = 2 * numerator_sum_updated  # (N//B)
        ##############################################################

        ###############################################################
        # [2] (Right, Bottom) wings
        # Preparation for wings
        X_prod_y_new = exp_logit_scale * torch.einsum("nhd,gd->nhg", X, y_new)  # (N//B, H, G)
        Y_prod_x_new = exp_logit_scale * torch.einsum("nhd,gd->ngh", Y, x_new)  # (N//B, G, H)
        exp_X_prod_y_new = torch.exp(X_prod_y_new)  # (N//B, H, G)
        exp_Y_prod_x_new = torch.exp(Y_prod_x_new)  # (N//B, G, H)
        # Right wing (column-vector) / vector shape : (H, 1)
        all_row_exp_sum = (row_exp_sum + torch.sum(exp_X_prod_y_new, dim=2, keepdim=True))  # (N//B, H, 1)
        # Bottom wing (row-vector) / vector shape : (1, H)
        all_col_exp_sum = (col_exp_sum + torch.sum(exp_Y_prod_x_new, dim=1, keepdim=True))  # (N//B, 1, H)
        # Loss from wings
        tmp += torch.sum(-torch.log(all_row_exp_sum.squeeze(dim=2)), dim=1)  #(N//B)
        tmp += torch.sum(-torch.log(all_col_exp_sum.squeeze(dim=1)), dim=1)  #(N//B)
        ###############################################################

        ################################################################
        # [3] newcomer's wings
        tiled_xy_new = torch.tile(torch.exp(xy_new).unsqueeze(0), (n_batch, 1, 1))  # (N//B, K_loop, K_loop)
        # 1) Sum of Bottom wing and 2) new_z_exp // [H+1, 1]
        new_row_exp = torch.concat([exp_Y_prod_x_new, tiled_xy_new], dim=2)  # (N//B, K_loop, K_pivot + c * K_loop)
        new_row_exp_sum = torch.sum(new_row_exp, dim=2, keepdim=True)        # (N//B, K_loop, 1)
        # 1) Sum of Right wing and 2) new_z_exp // [1, H+1]
        new_col_exp = torch.concat([exp_X_prod_y_new, tiled_xy_new], dim=1)  # (N//B, K_pivot + c * K_loop, K_loop)
        new_col_exp_sum = torch.sum(new_col_exp, dim=1, keepdim=True)        # (N//B, 1, K_loop)
        # Loss from newcomer's wing
        tmp += torch.sum(-torch.log(new_row_exp_sum.squeeze(dim=2)), dim=1)  # (N//B) / (N//B, 1, K_loop) -> (N//B)
        tmp += torch.sum(-torch.log(new_col_exp_sum.squeeze(dim=1)), dim=1)  # (N//B) / (N//B, K_loop, 1) -> (N//B)
        ################################################################

        # Negative Log Likelihood (Loss)
        tmp = -tmp

        # Ignore used batch (-inf)
        tmp += is_used_batch
        
        ################################################################
        # [4] Get maximum CLIP loss idx
        max_idx = torch.argmax(tmp)        
        ################################################################

        # Save temporal X, Y, row_exp_sum, col_exp_sum
        X_temp[max_idx] = x_features[newcomer_idx]
        Y_temp[max_idx] = y_features[newcomer_idx]
        row_exp_sum_temp[max_idx] = new_row_exp_sum[max_idx]
        col_exp_sum_temp[max_idx] = new_col_exp_sum[max_idx]

        # Update row, col, numerator sum
        row_exp_sum[max_idx] = all_row_exp_sum[max_idx]  # (N//B, H, 1)
        col_exp_sum[max_idx] = all_col_exp_sum[max_idx]  # (N//B, 1, H)
        numerator_sum[max_idx] = numerator_sum_updated[max_idx]

        # Update batch_idx_list
        batch_idx_list[max_idx] += newcomer_idx.tolist()
        # batch_idx_list[max_idx] += [newcomer_idx]
        is_used_batch[max_idx] = -float('inf')
        count += 1

        remain_idx = np.setdiff1d(remain_idx, newcomer_idx)

        # If all batch room is booked, Apepnd & Re-initialize 
        if count % (n_batch) == 0:

            # Append temporal variable to live variable
            row_exp_sum = torch.concat([row_exp_sum, row_exp_sum_temp], dim=1)  # (N//B, H, 1) -> (N//B, H+K, 1)
            col_exp_sum = torch.concat([col_exp_sum, col_exp_sum_temp], dim=2)  # (N//B, 1, H) -> (N//B, 1, H+K)
            X = torch.concat([X, X_temp], dim=1)                                # (N//B, H, D) -> (N//B, H+K, D)
            Y = torch.concat([Y, Y_temp], dim=1)                                # (N//B, H, D) -> (N//B, H+K, D)

            sanity_check = False
            if sanity_check:
                for b_idx in range(n_batch):
                    print(f"{b_idx} cur row  : {row_exp_sum[b_idx].squeeze()}")
                    row_exp_sum_prev = torch.sum(z_exp[batch_idx_list[b_idx], :][:, batch_idx_list[b_idx]], dim=1)
                    print(f"{b_idx} prev row : {row_exp_sum_prev}")
                    print(f"{b_idx} cur col  : {col_exp_sum[b_idx].squeeze()}")
                    col_exp_sum_prev = torch.sum(z_exp[batch_idx_list[b_idx], :][:, batch_idx_list[b_idx]], dim=0)
                    print(f"{b_idx} prev col : {col_exp_sum_prev}")
                    print()

            if last_loop:
                # print("last_loop")
                break

            # Check whether next loop is the last loop 
            current_batch_size = X.shape[1]
            if current_batch_size + K_loop >= B:
                # print(f"current_batch_size + K : {current_batch_size + K} / B : {B}")
                last_B = B - current_batch_size
                K_loop = last_B
                last_loop = True

            # Re-initialize
            X_temp = torch.zeros(n_batch, K_loop, D).to(device)
            Y_temp = torch.zeros(n_batch, K_loop, D).to(device)
            row_exp_sum_temp = torch.zeros(n_batch, K_loop, 1).to(device)
            col_exp_sum_temp = torch.zeros(n_batch, 1, K_loop).to(device)
            is_used_batch = torch.zeros(n_batch).to(device)
            
    end = time.time()
    et = end - start

    assert len(remain_idx) < N
    try:
        assert all([len(batch_idx) == B for batch_idx in batch_idx_list])
    except:
        import ipdb; ipdb.set_trace()

    if max_n_batch is not None:
        batch_idx_list = fill_the_rest_randomly(max_B, batch_idx_list, remain_idx)

    return batch_idx_list


# @logging_time
def create_balance_greedy_batches(N: int, B: int, x_features: torch.Tensor, y_features: torch.Tensor, exp_logit_scale: torch.Tensor, 
    num_rand: int = None,
    z: torch.Tensor = None, z_exp: torch.Tensor = None, 
    epoch: int = None, 
    device: str = 'cuda',
    K: int = 1,
    D: int = 1024,
    max_B: int = None,
    max_n_batch: int = None):
    """
    Time complexity : O ( N * (N // B) * (B * f(N)) ) = O (N^2 f(N))
    """


    # ### Psuedo Code ###
    # for B:
    #     for N // B:
    #         for N:      # will be optimized by matics trick
    #             for B   # will be optimized by matics trick
    # ### Psuedo Code ###

    x_features = x_features.to(device)
    y_features = y_features.to(device)

    n_batch = N // B
    if max_n_batch is not None:
        n_batch = max_n_batch
        assert max_B is not None

    if num_rand != None and num_rand != 0:
        K_pivot = num_rand # 초기에 채워진 개수
    else:
        K_pivot = K  # assign as K (=1)

    pivots = np.random.choice(N, n_batch * K_pivot, replace=False).reshape(n_batch, K_pivot)  # (N//B, K_pivot)
    batch_idx_list = [idx.tolist() for idx in pivots]                                         # (N//B, K_pivot)

    X = x_features[pivots.reshape(-1)].reshape(n_batch, K_pivot, D)  # (N//B, A, D) --eventually--> (N//B, B, D)
    Y = y_features[pivots.reshape(-1)].reshape(n_batch, K_pivot, D)  # (N//B, A, D) --eventually--> (N//B, B, D)

    pivot_z = exp_logit_scale * torch.einsum('nkd,nld->nkl', X, Y)   # (N//B, K_pivot, K_pivot)  
    pivot_z_diagonal = pivot_z.diagonal(offset=0, dim1=-1, dim2=-2)  # (N//B, K_pivot)
    pivot_z_exp = torch.exp(pivot_z)  # (N//B, K_pivot, K_pivot) 

    numerator_sum = torch.sum(pivot_z_diagonal.clone(), dim=-1)  # (N//B)
    row_exp_sum = torch.sum(pivot_z_exp.clone(), dim=2)  # (N//B, A) --eventually--> (N//B, B)
    col_exp_sum = torch.sum(pivot_z_exp.clone(), dim=1)  # (N//B, A) --eventually--> (N//B, B)

    # Initialize 
    X_temp = torch.zeros(n_batch, D).to(device)  # (N//B, D)
    Y_temp = torch.zeros(n_batch, D).to(device)  # (N//B, D)
    row_exp_sum_temp = torch.zeros(n_batch).to(device)  # (N//B)
    col_exp_sum_temp = torch.zeros(n_batch).to(device)  # (N//B)

    except_idx = np.array([batch_idx_list[j] for j in range(n_batch)]).reshape(-1)
    remain_idx = list(np.setdiff1d(np.arange(N), except_idx))                      # range(N) 

    # print("Create Balance Greedy Batches...")
    count = 0
    temp_cnt = 0
    temp_cnt2 = 0
    for batch_cum in range(B-K_pivot):  # N
        for b_idx in range(n_batch): # N // B
    # for batch_cum in tqdm(range(B-K_pivot), desc='batch'):  # N
    #     for b_idx in tqdm(range(n_batch), desc=f"batch_cum : {batch_cum}", total=n_batch if max_n_batch is None else max_n_batch): # N // B

            # start_time = time.time()
            # temp_cnt += time.time() - start_time
            # if count == 10:
            #     print(f"\nbg avg time : {temp_cnt / count}")
            #     exit()

            cur2ori_idx_map = {_iter: cur  for _iter, cur in enumerate(remain_idx)}  # range(REMAIN) "->" range(N) 
            compare_x = x_features[np.array(remain_idx)] # (REMAIN, D)
            compare_y = y_features[np.array(remain_idx)] # (REMAIN, D)
            num_remain = compare_x.shape[0]

            ##############################################################
            # [1] numerator
            z_new = exp_logit_scale * torch.einsum('nd,nd->n', compare_x, compare_y)  # (REMAIN, D), (REMAIN, D) -> (REMAIN)
            numerator_sum_updated = numerator_sum[b_idx] + z_new                      # (,), (REMAIN) -> (REMAIN)
            tmp = 2 * numerator_sum_updated                                           # (REMAIN)
            ##############################################################

            ##############################################################
            # [2] (Right, Bottom) wings
            # Right wings
            X_prod_y_new = exp_logit_scale * torch.einsum('rd,cd->rc', compare_y, X[b_idx])  # (REMAIN, C)
            # Bottom wings
            Y_prod_x_new = exp_logit_scale * torch.einsum('rd,cd->rc', compare_x, Y[b_idx])  # (REMAIN, C)
            exp_X_prod_y_new = torch.exp(X_prod_y_new)  # (REMAIN, C)
            exp_Y_prod_x_new = torch.exp(Y_prod_x_new)  # (REMAIN, C)
            # Right wing (column-vector) / vector shape : (C)
            all_row_exp_sum = (row_exp_sum[b_idx].unsqueeze(0) + exp_X_prod_y_new)  # (REMAIN, C)
            # assert list(all_row_exp_sum.shape) == [num_remain, batch_cum+1]
            # Bottom wing (row-vector) / vector shape : (C)
            all_col_exp_sum = (col_exp_sum[b_idx].unsqueeze(0) + exp_Y_prod_x_new)  # (REMAIN, C)
            # assert list(all_col_exp_sum.shape) == [num_remain, batch_cum+1]
            # Loss from wings
            # assert all_row_exp_sum.shape[1] == batch_cum+1
            # assert all_col_exp_sum.shape[1] == batch_cum+1
            tmp += torch.sum(-torch.log(all_row_exp_sum), dim=1)  #(REMAIN, C) -> (REMAIN)
            tmp += torch.sum(-torch.log(all_col_exp_sum), dim=1)  #(REMAIN, C) -> (REMAIN)

            ################################################################
            # [3] newcomer's wings
            # Sum of 1) Bottom wing and 2) new_z_exp
            # exp_Y_prod_x_new       (REMAIN, C)
            # z_new.unsqueeze(1)     (REMAIN, 1)
            new_row_exp = torch.concat([exp_Y_prod_x_new, torch.exp(z_new).unsqueeze(1)], dim=1)  # (RENAIN, C + 1)
            new_row_exp_sum = torch.sum(new_row_exp, dim=1)                   # (RENAIN)
            # Sum of 1) Right wing and 2) new_z_exp
            # exp_X_prod_y_new.           (REMAIN, C)
            # ...
            new_col_exp = torch.concat([exp_X_prod_y_new, torch.exp(z_new).unsqueeze(1)], dim=1)  # (RENAIN, C + 1)
            new_col_exp_sum = torch.sum(new_col_exp, dim=1)                   # (RENAIN)
            # Loss from newcomer's wing
            tmp += -torch.log(new_row_exp_sum)                                  # (RENAIN)
            tmp += -torch.log(new_col_exp_sum)                                  # (RENAIN)
            ################################################################
           
            # Negative Log Likelihood (Loss)
            tmp = -tmp

            ################################################################
            # [4] Get maximum CLIP loss idx
            max_idx = torch.argmax(tmp)
            ################################################################

            # Save temporal X, Y, row_exp_sum, col_exp_sum
            ori_max_idx = cur2ori_idx_map[max_idx.item()]
            X_temp[b_idx] = x_features[ori_max_idx]  # (D)
            Y_temp[b_idx] = y_features[ori_max_idx]  # (D)
            row_exp_sum_temp[b_idx] = new_row_exp_sum[max_idx]  # (1)
            col_exp_sum_temp[b_idx] = new_col_exp_sum[max_idx]  # (1)

            # Update row, col, numerator sum
            row_exp_sum[b_idx] = all_row_exp_sum[max_idx]  #
            col_exp_sum[b_idx] = all_col_exp_sum[max_idx]
            numerator_sum[b_idx] = numerator_sum_updated[max_idx]

            # Update batch_idx_list
            batch_idx_list[b_idx].append(ori_max_idx)  # max_idx: range(N-except) "->"" range(N)

            # Count update
            count += 1

            # Update remain_idx
            remain_idx.remove(ori_max_idx)
            
            # if max_n_batch is not None and max_n_batch-1 == b_idx:
            #     break
        # If all batch room is booked, Apepnd & Re-initialize 
        # Append temporal variable to live variable
        # import ipdb; ipdb.set_trace()
        row_exp_sum = torch.concat([row_exp_sum, row_exp_sum_temp.unsqueeze(1)], dim=1)  # (N//B, C) -> (N//B, C+1)
        col_exp_sum = torch.concat([col_exp_sum, col_exp_sum_temp.unsqueeze(1)], dim=1)  # (N//B, C) -> (N//B, C+1)
        X = torch.concat([X, X_temp.unsqueeze(1)], dim=1)                                # (N//B, C, D) -> (N//B, C+1, D)
        Y = torch.concat([Y, Y_temp.unsqueeze(1)], dim=1)                                # (N//B, C, D) -> (N//B, C+1, D)

        # assert row_exp_sum.shape[1] == batch_cum + 2
        # assert X.shape[1] == batch_cum + 2

        # Break (all batch has #batch_size (B))
        if X.shape[1] >= B:
            break

        sanity_check = False
        if sanity_check:
            for b_idx in range(N//B):
                print(f"{b_idx} cur row  : {row_exp_sum[b_idx].squeeze()}")
                row_exp_sum_prev = torch.sum(z_exp[batch_idx_list[b_idx], :][:, batch_idx_list[b_idx]], dim=1)
                print(f"{b_idx} prev row : {row_exp_sum_prev.squeeze()}")
                print(f"{b_idx} cur col  : {col_exp_sum[b_idx].squeeze()}")
                col_exp_sum_prev = torch.sum(z_exp[batch_idx_list[b_idx], :][:, batch_idx_list[b_idx]], dim=0)
                print(f"{b_idx} prev col : {col_exp_sum_prev.squeeze()}")
                print()

        # Re-initialize
        X_temp = torch.zeros(n_batch, D).to(device)
        Y_temp = torch.zeros(n_batch, D).to(device)
        row_exp_sum_temp = torch.zeros(n_batch).to(device)
        col_exp_sum_temp = torch.zeros(n_batch).to(device)

    if max_n_batch is not None:
        batch_idx_list = fill_the_rest_randomly(max_B, batch_idx_list, remain_idx)

    return batch_idx_list


# @logging_time
def create_greedy_batches(N: int, B: int, x_features: torch.Tensor, y_features: torch.Tensor, exp_logit_scale: torch.Tensor, 
    num_rand: int = None,
    z: torch.Tensor = None, z_exp: torch.Tensor = None, 
    epoch: int = None, 
    device: str = 'cuda',
    K: int = 1,
    D: int = 1024,
    max_B: int = None,
    max_n_batch: int = None):
    """
    Time complexity : O ( N * (N // B) * (B * f(N)) ) = O (N^2 f(N))
    """
    # ### Psuedo Code ###
    # for N // B:
    #     for B:
    #         for N:      # will be optimized by matics trick
    #             for B   # will be optimized by matics trick
    # ### Psuedo Code ###

    x_features = x_features.to(device)
    y_features = y_features.to(device)

    n_batch = N // B
    if max_n_batch is not None:
        n_batch = max_n_batch
        assert max_B is not None

    if num_rand != None and num_rand != 0:
        K_pivot = num_rand # 초기에 채워진 개수
    else:
        K_pivot = K  # assign as K (=1)

    pivots = np.random.choice(N, n_batch * K_pivot, replace=False).reshape(n_batch, K_pivot)  # (N//B, K_pivot)
    batch_idx_list = [idx.tolist() for idx in pivots]                                         # (N//B, K_pivot)

    X = x_features[pivots.reshape(-1)].reshape(n_batch, K_pivot, D)  # (N//B, A, D) --eventually--> (N//B, B, D)
    Y = y_features[pivots.reshape(-1)].reshape(n_batch, K_pivot, D)  # (N//B, A, D) --eventually--> (N//B, B, D)

    pivot_z = exp_logit_scale * torch.einsum('nkd,nld->nkl', X, Y)   # (N//B, K_pivot, K_pivot)  
    pivot_z_diagonal = pivot_z.diagonal(offset=0, dim1=-1, dim2=-2)  # (N//B, K_pivot)
    pivot_z_exp = torch.exp(pivot_z)  # (N//B, K_pivot, K_pviot) 

    # numerator_sum = torch.sum(pivot_z_diagonal.clone(), dim=-1)  # (N//B)
    # pivot_z_exp (N//B, 1, 1)
    # row_exp_sum = torch.sum(pivot_z_exp.clone(), dim=2)  # (N//B, A) --eventually--> (N//B, B)
    # col_exp_sum = torch.sum(pivot_z_exp.clone(), dim=1)  # (N//B, A) --eventually--> (N//B, B)

    # Initialize 
    # X_temp = torch.zeros(n_batch, D).to(device)  # (N//B, D)
    # Y_temp = torch.zeros(n_batch, D).to(device)  # (N//B, D)
    # row_exp_sum_temp = torch.zeros(n_batch).to(device)  # (N//B)
    # col_exp_sum_temp = torch.zeros(n_batch).to(device)  # (N//B)

    except_idx = np.array([batch_idx_list[j] for j in range(n_batch)]).reshape(-1)
    remain_idx = list(np.setdiff1d(np.arange(N), except_idx))                      # range(N) 

    # print("Create Greedy Batches...")
    count = 0
    temp_cnt = 0
    temp_cnt2 = 0
    for b_idx in range(n_batch): # N // B
    # for b_idx in tqdm(range(n_batch)): # N // B

        X_b = X[b_idx]                            # (1, D)
        Y_b = Y[b_idx]                            # (1, D)
        numerator_sum = torch.sum(pivot_z_diagonal[b_idx])   # (,)
        row_exp_sum = torch.sum(pivot_z_exp[b_idx], dim=1)  # (1) --> (B)
        col_exp_sum = torch.sum(pivot_z_exp[b_idx], dim=0)  # (1) --> (B)

        for batch_cum in range(B-K_pivot):  # B
        # for batch_cum in tqdm(range(B-K_pivot), desc=f'batch : {b_idx}'):  # B

            # start_time = time.time()
            # temp_cnt += time.time() - start_time
            # if count == 10:
            #     print(f"\nbg avg time : {temp_cnt / count}")
            #     exit()

            # sanity_check = True
            # if sanity_check:
            #     print(f"{b_idx} cur row  : {row_exp_sum[:5]}")
            #     row_exp_sum_prev = torch.sum(z_exp[batch_idx_list[b_idx], :][:, batch_idx_list[b_idx]], dim=1)
            #     print(f"{b_idx} prev row : {row_exp_sum_prev[:5]}")
            #     # print(f"{b_idx} cur col  : {col_exp_sum.squeeze()[:5]}")
            #     # col_exp_sum_prev = torch.sum(z_exp[batch_idx_list[b_idx], :][:, batch_idx_list[b_idx]], dim=0)
            #     # print(f"{b_idx} prev col : {col_exp_sum_prev.squeeze()[:5]}")
            #     print()

            cur2ori_idx_map = {_iter: cur for _iter, cur in enumerate(remain_idx)}  # range(REMAIN) "->" range(N) 
            compare_x = x_features[np.array(remain_idx)] # (REMAIN, D)
            compare_y = y_features[np.array(remain_idx)] # (REMAIN, D)
            num_remain = compare_x.shape[0]

            ##############################################################
            # [1] numerator
            z_new = exp_logit_scale * torch.einsum('nd,nd->n', compare_x, compare_y)  # (REMAIN, D), (REMAIN, D) -> (REMAIN)
            numerator_sum_updated = numerator_sum + z_new                             # (,), (REMAIN) -> (REMAIN)
            tmp = 2 * numerator_sum_updated                                           # (REMAIN)
            ##############################################################
            # print(tmp.shape)

            ##############################################################
            # [2] (Right, Bottom) wings
            # Right wings
            X_prod_y_new = exp_logit_scale * torch.einsum('rd,cd->rc', compare_y, X_b)  # (REMAIN, C)
            # Bottom wings
            Y_prod_x_new = exp_logit_scale * torch.einsum('rd,cd->rc', compare_x, Y_b)  # (REMAIN, C)
            exp_X_prod_y_new = torch.exp(X_prod_y_new)  # (REMAIN, C)
            exp_Y_prod_x_new = torch.exp(Y_prod_x_new)  # (REMAIN, C)
            # Right wing (column-vector) / vector shape : (C)
            # all_row_exp_sum = (row_exp_sum[b_idx].unsqueeze(0) + exp_X_prod_y_new)  # (REMAIN, C)
            all_row_exp_sum = row_exp_sum.unsqueeze(0) + exp_X_prod_y_new  # (REMAIN, C)
            # assert list(all_row_exp_sum.shape) == [num_remain, batch_cum+1]
            # Bottom wing (row-vector) / vector shape : (C)
            all_col_exp_sum = col_exp_sum.unsqueeze(0) + exp_Y_prod_x_new  # (REMAIN, C)
            # assert list(all_col_exp_sum.shape) == [num_remain, batch_cum+1]
            # Loss from wings
            # assert all_row_exp_sum.shape[1] == batch_cum+1
            # assert all_col_exp_sum.shape[1] == batch_cum+1
            tmp += torch.sum(-torch.log(all_row_exp_sum), dim=1)  #(REMAIN, C) -> (REMAIN)
            tmp += torch.sum(-torch.log(all_col_exp_sum), dim=1)  #(REMAIN, C) -> (REMAIN)

            ################################################################
            # [3] newcomer's wings
            # Sum of 1) Bottom wing and 2) new_z_exp
            # exp_Y_prod_x_new       (REMAIN, C)
            # z_new.unsqueeze(1)     (REMAIN, 1)
            new_row_exp = torch.concat([exp_Y_prod_x_new, torch.exp(z_new).unsqueeze(1)], dim=1)  # (RENAIN, C + 1)
            new_row_exp_sum = torch.sum(new_row_exp, dim=1)                   # (RENAIN)
            # Sum of 1) Right wing and 2) new_z_exp
            # exp_X_prod_y_new.           (REMAIN, C)
            # ...
            new_col_exp = torch.concat([exp_X_prod_y_new, torch.exp(z_new).unsqueeze(1)], dim=1)  # (RENAIN, C + 1)
            new_col_exp_sum = torch.sum(new_col_exp, dim=1)                   # (RENAIN)
            # Loss from newcomer's wing
            tmp += -torch.log(new_row_exp_sum)                                  # (RENAIN)
            tmp += -torch.log(new_col_exp_sum)                                  # (RENAIN)
            ################################################################
           
            # Negative Log Likelihood (Loss)
            tmp = -tmp

            ################################################################
            # [4] Get maximum CLIP loss idx
            max_idx = torch.argmax(tmp)
            ################################################################

            # 
            ori_max_idx = cur2ori_idx_map[max_idx.item()]

            # Update row, col, numerator sum
            # X_b -- (A, D), compare_x[max_idx] -- (REMAIN, D)
            X_b = torch.concat((X_b, compare_x[max_idx:max_idx+1]))
            Y_b = torch.concat((Y_b, compare_y[max_idx:max_idx+1]))
            row_exp_sum = torch.concat((all_row_exp_sum[max_idx], new_row_exp_sum[max_idx:max_idx+1]), dim=0)  # (C), (1) -> (C+1)
            col_exp_sum = torch.concat((all_col_exp_sum[max_idx], new_col_exp_sum[max_idx:max_idx+1]), dim=0)
            numerator_sum = numerator_sum_updated[max_idx]
            
            # Update batch_idx_list
            batch_idx_list[b_idx].append(ori_max_idx)  # max_idx: range(N-except) "->"" range(N)

            # Count update
            count += 1

            # Update remain_idx
            remain_idx.remove(ori_max_idx)
            # print(f"remain_idx shape : {remain_idx}")
            
            sanity_check = False
            if sanity_check:
                # print(batch_idx_list[b_idx])
                print(f"{b_idx} cur row  : {row_exp_sum.squeeze()[:5]}")
                row_exp_sum_prev = torch.sum(z_exp[batch_idx_list[b_idx], :][:, batch_idx_list[b_idx]], dim=1)
                print(f"{b_idx} prev row : {row_exp_sum_prev.squeeze()[:5]}")
                print(f"{b_idx} cur col  : {col_exp_sum.squeeze()[:5]}")
                col_exp_sum_prev = torch.sum(z_exp[batch_idx_list[b_idx], :][:, batch_idx_list[b_idx]], dim=0)
                print(f"{b_idx} prev col : {col_exp_sum_prev.squeeze()[:5]}")
                print()

            # Break (all batch has #batch_size (B))
            if X_b.shape[0] >= B:
                break

        # assert row_exp_sum.shape[1] == batch_cum + 2
        # assert X.shape[1] == batch_cum + 2


        # sanity_check = True
        # if sanity_check:
        #     print(f"{b_idx} cur row  : {row_exp_sum.squeeze()[:5]}")
        #     row_exp_sum_prev = torch.sum(z_exp[batch_idx_list[b_idx], :][:, batch_idx_list[b_idx]], dim=1)
        #     print(f"{b_idx} prev row : {row_exp_sum_prev.squeeze()[:5]}")
        #     print(f"{b_idx} cur col  : {col_exp_sum.squeeze()[:5]}")
        #     col_exp_sum_prev = torch.sum(z_exp[batch_idx_list[b_idx], :][:, batch_idx_list[b_idx]], dim=0)
        #     print(f"{b_idx} prev col : {col_exp_sum_prev.squeeze()[:5]}")
        #     print()

        # # Re-initialize
        # X_temp = torch.zeros(n_batch, D).to(device)
        # Y_temp = torch.zeros(n_batch, D).to(device)
        # row_exp_sum_temp = torch.zeros(n_batch).to(device)
        # col_exp_sum_temp = torch.zeros(n_batch).to(device)

        # if max_n_batch is not None and max_n_batch-1 == b_idx:
        #     break

    # if max_n_batch is not None:
    #     batch_idx_list = fill_the_rest_randomly(max_B, batch_idx_list, remain_idx)

    return batch_idx_list


def fill_the_rest_randomly(max_B, batch_idx_list, remain_idx):

    batch_idx_list = np.array(batch_idx_list)
    n_batch, selected_batch_size = batch_idx_list.shape
    random.shuffle(remain_idx)
    random_batch_idx_list = np.array(remain_idx[:n_batch * (max_B - selected_batch_size)]).reshape(n_batch, -1)
    batch_idx_list = np.concatenate((batch_idx_list, random_batch_idx_list), axis=-1)

    return batch_idx_list.tolist()


###################################
###     Spectral Clustering     ###
###################################
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment    
from sklearn.utils import check_random_state
from sklearn.manifold import spectral_embedding
from sklearn.cluster import k_means, KMeans, SpectralClustering


def spectral_clustering(
    affinity,
    *,
    n_clusters=8,
    n_components=None,
    eigen_solver=None,
    random_state=None,
    n_init=10,
    eigen_tol="auto",
    assign_labels="kmeans",
    verbose=False,
    batch_selection=None,
    batch_size=None,
    norm_laplacian=True,
):
    if assign_labels not in ("kmeans", "discretize", "cluster_qr"):
        raise ValueError(
            "The 'assign_labels' parameter should be "
            "'kmeans' or 'discretize', or 'cluster_qr', "
            f"but {assign_labels!r} was given"
        )
    if isinstance(affinity, np.matrix):
        raise TypeError(
            "spectral_clustering does not support passing in affinity as an "
            "np.matrix. Please convert to a numpy array with np.asarray. For "
            "more information see: "
            "https://numpy.org/doc/stable/reference/generated/numpy.matrix.html",  # noqa
        )
    random_state = check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components

    maps = spectral_embedding(
        affinity,
        n_components=n_components,
        eigen_solver=eigen_solver,
        random_state=random_state,
        eigen_tol=eigen_tol,
        norm_laplacian=True,
        drop_first=False,
    )
    if verbose:
        print(f"Computing label assignment using {assign_labels}")

    assert batch_selection and batch_size
    if assign_labels == "kmeans":
        if batch_selection == "sc":
            _, labels, _ = k_means(
                maps, n_clusters, random_state=random_state, n_init=n_init, verbose=verbose
            )
        elif batch_selection == "sc_even":
            X, cluster_size = maps, batch_size
            kmeans = KMeans(n_clusters, n_init=n_init)
            kmeans.fit(X)
            centers = kmeans.cluster_centers_
            centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
            distance_matrix = cdist(X, centers)
            labels = linear_sum_assignment(distance_matrix)[1]//cluster_size

    elif assign_labels == "cluster_qr":
        labels = cluster_qr(maps)
    else:
        labels = discretize(maps, random_state=random_state)

    return labels


class CustomSpectralClustering(SpectralClustering):

    def __init__(self, n_clusters, batch_selection, batch_size, affinity='precomputed', eigen_solver='arpack', n_components=None, norm_laplacian=True):
        super().__init__(n_clusters, affinity=affinity, eigen_solver=eigen_solver)
        assert batch_selection in ['sc', 'sc_even']
        self.batch_selection = batch_selection
        self.batch_size = batch_size
        self.n_components = n_components
        self.norm_laplacian = norm_laplacian

    def fit(self, X, y=None):
        # self._validate_params()

        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc", "coo"],
            dtype=np.float64,
            ensure_min_samples=2,
        )
        allow_squared = self.affinity in [
            "precomputed",
            "precomputed_nearest_neighbors",
        ]
        if X.shape[0] == X.shape[1] and not allow_squared:
            warnings.warn(
                "The spectral clustering API has changed. ``fit``"
                "now constructs an affinity matrix from data. To use"
                " a custom affinity matrix, "
                "set ``affinity=precomputed``."
            )

        if self.affinity == "nearest_neighbors":
            connectivity = kneighbors_graph(
                X, n_neighbors=self.n_neighbors, include_self=True, n_jobs=self.n_jobs
            )
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == "precomputed_nearest_neighbors":
            estimator = NearestNeighbors(
                n_neighbors=self.n_neighbors, n_jobs=self.n_jobs, metric="precomputed"
            ).fit(X)
            connectivity = estimator.kneighbors_graph(X=X, mode="connectivity")
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == "precomputed":
            self.affinity_matrix_ = X
        else:
            params = self.kernel_params
            if params is None:
                params = {}
            if not callable(self.affinity):
                params["gamma"] = self.gamma
                params["degree"] = self.degree
                params["coef0"] = self.coef0
            self.affinity_matrix_ = pairwise_kernels(
                X, metric=self.affinity, filter_params=True, **params
            )

        random_state = check_random_state(self.random_state)
        self.labels_ = spectral_clustering(
            self.affinity_matrix_,
            n_clusters=self.n_clusters,
            n_components=self.n_components,
            eigen_solver=self.eigen_solver,
            random_state=random_state,
            n_init=self.n_init,
            eigen_tol=self.eigen_tol,
            assign_labels=self.assign_labels,
            verbose=False,#self.verbose,
            batch_selection=self.batch_selection,
            batch_size=self.batch_size,
            norm_laplacian=self.norm_laplacian,
        )
        return self


def custom_affinity(U, V, B):
    # Compute the pairwise Euclidean distances between data points
    Z = U@np.transpose(V)  # UV
    Z_T = np.transpose(Z)  # VU

    d = np.diag(Z)  ## UU
    Z_sub = np.transpose(d*np.ones((len(d), len(d))))

    affinity = np.log(1+(B-1)*np.exp(Z-Z_sub))
    affinity += np.log(1+(B-1)*np.exp(Z_T-Z_sub))

    affinity += np.transpose(affinity)

    return affinity


def index_groups(a_idx, input_list):
    index_dict = defaultdict(list)

    for idx, value in enumerate(input_list):
        index_dict[value].append(idx)

    return [a_idx[index].tolist() for index in list(index_dict.values())]


def sc_naive(batch_size, x_features: torch.Tensor, y_features: torch.Tensor, batch_selection="sc_even", tqdm_desc=True):
    # normalize embeddings
    a_idx = torch.arange(0, x_features.shape[0], dtype=torch.int)
    a_z_i = x_features
    a_z_j = y_features

    assert len(a_idx) % batch_size == 0
    n_clusters = len(a_idx) // batch_size

    sc = CustomSpectralClustering(n_clusters, batch_selection, batch_size, affinity='precomputed', eigen_solver='arpack')
    affinity = custom_affinity(a_z_i.cpu().detach().numpy(), a_z_j.cpu().detach().numpy(), batch_size)
    psd_affinity = np.exp(affinity)
    y_pred = sc.fit_predict(psd_affinity)

    iterator = y_pred if not tqdm_desc else tqdm(y_pred, desc=f'sc naive')
    return index_groups(a_idx, iterator)


if __name__ == "__main__":
    epoch = 0
    max_dataset_size = 1187
    batch_size = 100
    d = 1024
    device = 'cuda'  #'cuda'
    x = torch.rand(max_dataset_size, d).to(device)
    y = torch.rand(max_dataset_size, d).to(device)

    x = torch.nn.functional.normalize(x)
    y = torch.nn.functional.normalize(y)
    exp_logit_scale = 100.

    z_matrix, z_exp = None, None
    # z_matrix = exp_logit_scale * x @ y.T
    # z_exp = torch.exp(z_matrix)

    selected_batch_size, max_n_batch = batch_size, None
    # rand_ratio = 0.3
    # selected_batch_size = int(batch_size * (1 - rand_ratio))
    # max_n_batch = max_dataset_size // batch_size

    batch_idx_list = create_greedy_batches(max_dataset_size, selected_batch_size, x, y, exp_logit_scale, max_B=batch_size, max_n_batch=max_n_batch,
        z=z_matrix, z_exp=z_exp, epoch=epoch, device=device)

    # batch_idx_list = create_balance_greedy_batches(max_dataset_size, selected_batch_size, x, y, exp_logit_scale, max_B=batch_size, max_n_batch=max_n_batch,
    #     z=z_matrix, z_exp=z_exp, epoch=epoch, device=device)

    # batch_idx_list = create_inverse_greedy_batches_with_K(max_dataset_size, selected_batch_size, x, y, exp_logit_scale, max_B=batch_size, max_n_batch=max_n_batch,
    #     z=z_matrix, z_exp=z_exp, epoch=epoch, device=device)

    import ipdb; ipdb.set_trace()

    # N = max_dataset_size
    # B = batch_size
    # ig_z = torch.rand(N//B, B, B).to('cuda')
    # bg_z = torch.rand(N, B, B).to('cuda')
    # ig_z_exp = torch.exp(ig_z)
    # bg_z_exp = torch.exp(bg_z)

    # tmp = 0
    # start_time = time.time()
    # ig_diag = ig_z.diagonal(offset=0, dim1=-1, dim2=-2)
    # ig_num = torch.sum(ig_diag, dim=-1)
    # tmp += 2*ig_num
    # row_wise_sum = -torch.sum(torch.log(torch.sum(ig_z_exp, dim=1, keepdim=True)), dim=2, keepdim=True).squeeze()
    # col_wise_sum = -torch.sum(torch.log(torch.sum(ig_z_exp, dim=2, keepdim=True)), dim=1, keepdim=True).squeeze()
    # tmp += row_wise_sum
    # tmp += col_wise_sum
    # tmp -= tmp

    # end_time = time.time()
    # print(f"ig : {end_time - start_time}")

    # start_time = time.time()
    # tmp = 0
    # bg_diag = bg_z.diagonal(offset=0, dim1=-1, dim2=-2)
    # bg_num = torch.sum(bg_diag, dim=-1)
    # tmp += 2*bg_num
    # row_wise_sum = -torch.sum(torch.log(torch.sum(bg_z_exp, dim=1, keepdim=True)), dim=2, keepdim=True).squeeze()
    # col_wise_sum = -torch.sum(torch.log(torch.sum(bg_z_exp, dim=2, keepdim=True)), dim=1, keepdim=True).squeeze()
    # tmp += row_wise_sum
    # tmp += col_wise_sum
    # tmp -= tmp
    # end_time = time.time()
    # print(f"bg : {end_time - start_time}")


    temp = 1

#@title CLIP loss (negative log likelihood)
def CLIP_loss_torch(z): # negative log likelihood
  z_exp = torch.exp(z)
  tmp = 0 # (positive) log likelihood
  for i in range(z.shape[0]):
    tmp += 2*z[i,i] # numerator
    tmp -= torch.log(torch.sum(z_exp[i,:])) # x to y classification
    tmp -= torch.log(torch.sum(z_exp[:,i])) # y to x classification
  tmp = tmp / z.shape[0] / 2  # average
  return -tmp # make it negative


def get_random_batch_idxs(N, B=2):
    batch_idxs = np.arange(N)
    np.random.shuffle(batch_idxs)
    if (N % B) == 0:
        batch_idxs = batch_idxs.reshape(-1, B).tolist()
    elif (N // B) == 1: # allow overlap between batches
        batch_idxs = [batch_idxs[:B].tolist(), batch_idxs[-B:].tolist()]
    else:
        raise NotImplementedError
    return batch_idxs


