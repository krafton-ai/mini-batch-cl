#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from random import randint
import math
import copy

from tqdm import tqdm
import torch
import torch.nn as nn
import random
import numpy as np
import itertools
import torch.nn.functional as F

from collections import defaultdict
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
device = 'cuda:0' # ['cpu', 'cuda:0']


# In[ ]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=20)
parser.add_argument("--d", type=int, default=32)
parser.add_argument("--lr_full", type=float, default=0.5)
parser.add_argument("--lr_scaling", default=False, action='store_true')
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--logging_step_ratio", type=float, default=0.1)
parser.add_argument("--gradient_accumulation", default=False, action='store_true')
args = parser.parse_args()

print("N", args.N)
print("d", args.d)
print("lr_full", args.lr_full)
print("lr_scaling", args.lr_scaling)
print("num_steps", args.num_steps)
print("logging_step_ratio", args.logging_step_ratio)
print("gradient_accumulation", args.gradient_accumulation)


# In[2]:


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # making sure GPU runs are deterministic even if they are slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print("Seeded everything: {}".format(seed))


def full_batch_loss(u, v):
    n = u.shape[0]
    logits = torch.exp(u @ v.T)
    loss = -torch.log(logits/torch.sum(logits, dim=1)).diagonal(dim1=0).sum()
    return loss/n

def clip_batch_loss(u, v):
    return full_batch_loss(u, v) + full_batch_loss(v, u)


# In[3]:


def mini_batch_loss(u, v, batch_idxs=None, B=2):
    loss = 0
    if batch_idxs == None:
        # find all possible batches of size B
        batch_idxs = list(itertools.combinations([i for i in range(u.shape[0])], B))
    n = len(batch_idxs)
    for batch_idx in batch_idxs:
        u_batch = u[list(batch_idx)]
        v_batch = v[list(batch_idx)]
        loss += clip_batch_loss(u_batch, v_batch)
    return loss/n


# In[4]:


N = args.N
d = args.d
Bs = [2, 5, 10, 20]
dict_template = { # in the order of algorithms & in the order of B
    "u": [],
    "v": [],
    "metrics": [],
    "true_loss": [],
    "loss": [],
    "norm": [],
}
res = defaultdict(lambda:copy.deepcopy(dict_template))

lr_full = args.lr_full
lr_scaling = args.lr_scaling
NUM_STEPS = args.num_steps
logging_step = int(NUM_STEPS * args.logging_step_ratio)
gradient_accumulation = args.gradient_accumulation

ga_tag = f"_accum" if gradient_accumulation else ""
lrs_tag = f"_lrs" if lr_scaling else ""
output_dir = f"output/sim_plots_N{N}_d{d}_lr{lr_full}{ga_tag}_s{NUM_STEPS}{lrs_tag}"
# import os
# print(os.getcwd(), output_dir)
os.makedirs(output_dir, exist_ok=True)

def get_embeddings(N, d):
    u = torch.randn((N, d), requires_grad=True, device=device)
    v = torch.randn((N, d), requires_grad=True, device=device)
    with torch.no_grad():
        u.data = F.normalize(u.data, p=2.0, dim=1)
        v.data = F.normalize(v.data, p=2.0, dim=1)
    return (u, v)

def plot_embeddings(u, v, d):
    # project down to 2d to visualize
    linear_projection = torch.randn(d, 2)
    proj_u = F.normalize(u.to('cpu')@linear_projection.detach(), p=2.0, dim=1)
    proj_v = F.normalize(v.to('cpu')@linear_projection.detach(), p=2.0, dim=1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(proj_u[:, 0].detach().numpy(), proj_u[:, 1].detach().numpy(), color='blue', label='u', marker="+", s=150)
    ax.scatter(proj_v[:, 0].detach().numpy(), proj_v[:, 1].detach().numpy(), color='red', label='v')
    ax.legend(loc='best')
    plt.grid()
    plt.show()
    return proj_u, proj_v


# In[5]:


def _get_mrr(indices):
    mrr = []
    for ii, inds in enumerate(indices):
        mrr.append(1 / (inds.tolist().index(ii)+1))
    return np.mean(mrr)

def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mrr"] = _get_mrr(ranking)
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


# ## First minimize full_batch loss

# In[6]:


set_seed(42)
u1, v1 = get_embeddings(N, d)
param_list = [u1, v1]

_, _ = plot_embeddings(u1[:30], v1[:30], d)

loss_full, norm_full, true_loss_full, metrics_full = {}, {}, {}, {}
optimizer = torch.optim.SGD(param_list, lr=lr_full)
for step in tqdm(range(NUM_STEPS), desc="[Full Batch]"):
    optimizer.zero_grad()
    loss = clip_batch_loss(u1, v1)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        u1.data = F.normalize(u1.data, p=2.0, dim=1)
        v1.data = F.normalize(v1.data, p=2.0, dim=1)
    if step %logging_step == 0 or step == NUM_STEPS-1:
        norm = torch.norm(u1.grad.data)
        # print("Step={} | Loss={} | Grad Norm={}".format(step, loss, norm))
        loss_full[step], norm_full[step], true_loss_full[step] = loss.detach().item(), norm.detach().item(), loss.detach().item()
        metrics_full[step] = get_metrics(u1, v1, 1.0)

res["full"] = {
    "u": [u1.detach().cpu()],
    "v": [v1.detach().cpu()],
    "metrics": [metrics_full],
    "loss": [loss_full],
    "norm": [norm_full],
    "true_loss": [true_loss_full],
}


# ## Check if it is ETF (It works!)

# In[7]:


u1_proj, v1_proj = plot_embeddings(u1[:30], v1[:30], d)
# first see if u = v
print("||u-v|| = {}".format(torch.norm(u1-v1)))
# now see if the inner products are equal
print("u^T v={}".format(u1 @ v1.T))


# ## Now minimize mini-batch loss over all specific batches

# In[ ]:


from batch_utils import create_inverse_greedy_batches_with_K, create_balance_greedy_batches, create_greedy_batches
def get_random_batch_idxs(N, B=2):
    batch_idxs = np.arange(N)
    np.random.shuffle(batch_idxs)
    return batch_idxs.reshape(-1, B).tolist()


# In[ ]:


batch_selections = ['full', 'f', 's', 'g', 'bg', 'ig'] #['full', 'f', 's', 'g', 'bg', 'ig']


# In[ ]:


for i, batch_selection in enumerate(batch_selections):
    if batch_selection == 'full':
        continue

    res[batch_selection] = copy.deepcopy(dict_template)

    for j, B in enumerate(Bs):
        #######
        set_seed(42)
        u2, v2 = get_embeddings(N, d)
        if i==1 and j==0: # plot once only
            _, _ = plot_embeddings(u2[:30], v2[:30], d)
        param_list = [u2, v2]
        #######
        optimizer = torch.optim.SGD(param_list, lr=lr_full*(B/N) if lr_scaling else lr_full)

        if B == N or batch_selection == 'f':
            batch_idxs = get_random_batch_idxs(N, B)

        loss_mini, norm_mini, true_loss_mini, metrics_mini = {}, {}, {}, {}
        for step in tqdm(range(NUM_STEPS), desc=f"[batch_selection:'{batch_selection}' | B:{B}] "):
            if B == N or batch_selection == 'f':
                pass
            elif batch_selection == 's':
                batch_idxs = get_random_batch_idxs(N, B)
            elif batch_selection == 'g':
                batch_idxs = create_greedy_batches(N, B, u2, v2, 1.0, device=device, D=d)
            elif batch_selection == 'bg':
                batch_idxs = create_balance_greedy_batches(N, B, u2, v2, 1.0, device=device, D=d)
            elif batch_selection == 'ig':
                batch_idxs = create_inverse_greedy_batches_with_K(N, B, u2, v2, 1.0, device=device, D=d)
            else:
                raise NotImplementedError(f'{batch_selection} is not available for batching')

            if gradient_accumulation:
                optimizer.zero_grad()
                loss = mini_batch_loss(u2, v2, batch_idxs=batch_idxs)
                loss.backward()
                optimizer.step()
            else:
                total_loss = 0
                for batch_idx in batch_idxs:
                    optimizer.zero_grad()
                    u_batch = u2[list(batch_idx)]
                    v_batch = v2[list(batch_idx)]
                    loss = clip_batch_loss(u_batch, v_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss
                loss = total_loss / len(batch_idxs)

            with torch.no_grad():
                u2.data = F.normalize(u2.data, p=2.0, dim = 1)
                v2.data = F.normalize(v2.data, p=2.0, dim = 1)
            if step %logging_step == 0 or step == NUM_STEPS-1:
                # print(step, batch_idxs[0], loss.detach().item(), clip_batch_loss(u2, v2).detach().item(), u2, v2)
                norm = torch.norm(u2.grad.data)
                # print("Step={} | Loss={} | Grad Norm={}".format(step, loss, norm))
                loss_mini[step], norm_mini[step], true_loss_mini[step] = loss.detach().item(), norm.detach().item(), clip_batch_loss(u2, v2).detach().item()
                metrics_mini[step] = get_metrics(u2, v2, 1.0)
                torch.save(u2, f"{output_dir}/u2_{batch_selection}_{step}.pt")
                torch.save(v2, f"{output_dir}/u2_{batch_selection}_{step}.pt")

        res[batch_selection]["u"] += [u2.detach().cpu()]
        res[batch_selection]["v"] += [v2.detach().cpu()]
        res[batch_selection]["metrics"] += [metrics_mini]
        res[batch_selection]["loss"] += [loss_mini]
        res[batch_selection]["norm"] += [norm_mini]
        res[batch_selection]["true_loss"] += [true_loss_mini]
    _, _ = plot_embeddings(u2[:30], v2[:30], d)


# ## Define variables for plots

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
from functools import partial

from matplotlib import rcParams


# In[ ]:


# define constants and other global variables
label_size = 25
tick_size = 25
title_size = 25
subtitle_size = 28
caption_size = 24
line_width = 5
legend_size = 20
markersize=18
#
colors = ['#073B4C', '#FFD166', '#EF476F', '#06D6A0',  '#118AB2', '#4D4C7D', '#FFDAC0', '#F54952']
colors2 = ['#dd3497', '#ae017e', '#7a0177', '#49006a'] # red-purple
warm_colors= ['#C2E4FF', '#2a2a72', '#009ffd', '#ffa400']
cold_colors = ['#DBDFFD', '#9BA3EB', '#646FD4', '#242F9B']
#colors2 = ['#dd3497', '#49006A', '#AE017E', '#7A0177'] # red-purple
# WT: '#073B4C'
# IMP: '#FFD166'
# EP: '#06D6A0'
# SR: '#118AB2'  
# GM: '#dd3497'
# GM (reinit): '#ae017e'
# GM (shuffle): '#7a0177'
# GM (invert): '#49006a'

fig_width = 8
fig_height = 4
x_label = [50, 20, 5, 2.5, 1.4, 0.5]
x_lim = [0.48, 55.0]
y_label_af = [i*10 for i in range(5,10)]
y_lim_af = [48, 95]
y_label_bf = [i*20-10 for i in range(1,6)]
y_lim_bf = [8, 95]

plot_eps = False #True #  set y-axis as (1-eps) instead of actual accuracy value
compare_iclr22 = False # check in visualize-results (fix ticks issue)-Copy1.ipynb
SPARSITY_FLIP=True # set sparsity = "density" for neurips camera ready
KEEP_Y_LABEL=False # keep this on for only the first column of plots


# In[ ]:


# Create sample data
res_data = {}
for batch_selection in batch_selections:
    mul_coef = len(Bs) if batch_selection == 'full' else 1
    res_data.update({
        f'{batch_selection}_batch': Bs,
        f'{batch_selection}_true_loss': [r[NUM_STEPS-1] for r in res[batch_selection]["true_loss"]] * mul_coef,
        f'{batch_selection}_loss': [r[NUM_STEPS-1] for r in res[batch_selection]["loss"]] * mul_coef,
        f'{batch_selection}_utov_R@1': [r[NUM_STEPS-1]["image_to_text_R@1"] for r in res[batch_selection]["metrics"]] * mul_coef,
        f'{batch_selection}_vtou_R@1': [r[NUM_STEPS-1]["text_to_image_R@1"] for r in res[batch_selection]["metrics"]] * mul_coef,
    })
res_data_df = pd.DataFrame(data=res_data)
res_data_df.to_csv(f'{output_dir}/res_data.csv', sep='|', encoding='utf-8-sig', index=False)
res_data_df


# ## Draw loss plot

# In[ ]:


### True loss Plot ###

labels = batch_selections#[:3] # whether to include greedy families

fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
line_width=2
markersize=10

for i, batch_selection in enumerate(labels):
    res_data_df.plot(x=f'{batch_selection}_batch', y=f'{batch_selection}_true_loss', marker='s', style="--", markersize=markersize, linewidth=line_width, c=colors[i], label=labels[i]+"_true_loss", ax=ax1)
    # res_data_df.plot(x=f'{batch_selection}_batch', y=f'{batch_selection}_loss', marker='^', markersize=markersize, linewidth=line_width, c=colors[i], label=labels[i]+"_loss", ax=ax1)
    # plt.axhline(100, linestyle='--')

## title, axis label
# plt.title("{} train T->I, RN50".format(measure), fontsize=title_size)
ax1.tick_params(axis='both', labelsize=tick_size)
ax1.set_xlabel("Batch Size", fontsize=label_size)



# ## tick, lim
# plt.tick_params(labelsize=tick_size)
# xticks_label = [1, 50, 100, 150, 200, 300]
# # zoom1
# # xticks_label = [0, 1, 5, 15]

# plt.xticks(xticks_label, xticks_label)

# yticks_label = [0.0, 2.0, 3.0, 4.0, 5.0]
yticks_label = [4.0, 4.1, 4.2]
plt.yticks(yticks_label, yticks_label)

# x_lim = [-1, 160]
# # zoom1
# # x_lim = [-1, 5]
# y_lim = [10, 100]

# ax1.set_xlim(x_lim)
# ax1.set_ylim(y_lim)


#plt.yticks(yticks_label, yticks_label)
#ax1.set_ylim(y_lim_bf)
ax1.set_ylabel(" ", fontsize=label_size)
# ax1.set_ylabel("Top-1 Recall (%)", fontsize=label_size)


    
## Legend
handles_ord, labels_ord = plt.gca().get_legend_handles_labels()
last_idx = len(labels)
#last_idx = len(labels) - 1
order = [i for i in range(last_idx)]
#order.insert(0,last_idx)

ax1.legend([handles_ord[idx] for idx in order],[labels_ord[idx] for idx in order], handlelength=2,
           fontsize=legend_size, loc='center left', bbox_to_anchor=(1.0, 0.5))
# ax1.get_legend().remove()
ax1.grid(True)

plt.savefig(f'{output_dir}/true_loss_all.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)


# In[ ]:


### Loss Plot ###

labels = batch_selections

fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
line_width=2
markersize=10

for i, batch_selection in enumerate(labels):
    # res_data_df.plot(x=f'{batch_selection}_batch', y=f'{batch_selection}_true_loss', marker='s', style="--", markersize=markersize, linewidth=line_width, c=colors[i], label=labels[i]+"_true_loss", ax=ax1)
    res_data_df.plot(x=f'{batch_selection}_batch', y=f'{batch_selection}_loss', marker='^', markersize=markersize, linewidth=line_width, c=colors[i], label=labels[i]+"_loss", ax=ax1)
    # plt.axhline(100, linestyle='--')

## title, axis label
# plt.title("{} train T->I, RN50".format(measure), fontsize=title_size)
ax1.tick_params(axis='both', labelsize=tick_size)
ax1.set_xlabel("Batch Size", fontsize=label_size)



# ## tick, lim
# plt.tick_params(labelsize=tick_size)
# xticks_label = [1, 50, 100, 150, 200, 300]
# # zoom1
# # xticks_label = [0, 1, 5, 15]

# plt.xticks(xticks_label, xticks_label)

yticks_label = [0.0, 2.0, 3.0, 4.0, 5.0]
# yticks_label = [4.0, 4.1]
plt.yticks(yticks_label, yticks_label)

# x_lim = [-1, 160]
# # zoom1
# # x_lim = [-1, 5]
# y_lim = [10, 100]

# ax1.set_xlim(x_lim)
# ax1.set_ylim(y_lim)


#plt.yticks(yticks_label, yticks_label)
#ax1.set_ylim(y_lim_bf)
ax1.set_ylabel(" ", fontsize=label_size)
# ax1.set_ylabel("Top-1 Recall (%)", fontsize=label_size)


    
## Legend
handles_ord, labels_ord = plt.gca().get_legend_handles_labels()
last_idx = len(labels)
#last_idx = len(labels) - 1
order = [i for i in range(last_idx)]
#order.insert(0,last_idx)

ax1.legend([handles_ord[idx] for idx in order],[labels_ord[idx] for idx in order], handlelength=2,
           fontsize=legend_size, loc='center left', bbox_to_anchor=(1.0, 0.5))
# ax1.get_legend().remove()
ax1.grid(True)

plt.savefig(f'{output_dir}/loss_all.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)


# ## Draw R@1 plot

# In[ ]:


### R@1 Plot ###

labels = batch_selections

fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
line_width=2
markersize=10

for i, batch_selection in enumerate(labels):
    res_data_df.plot(x=f'{batch_selection}_batch', y=f'{batch_selection}_utov_R@1', marker='s', style="--", markersize=markersize, linewidth=line_width, c=colors[i], label=labels[i]+"_utov_R@1", ax=ax1)
    res_data_df.plot(x=f'{batch_selection}_batch', y=f'{batch_selection}_vtou_R@1', marker='^', markersize=markersize, linewidth=line_width, c=colors[i], label=labels[i]+"_vtou_R@1", ax=ax1)
    # plt.axhline(100, linestyle='--')

## title, axis label
# plt.title("{} train T->I, RN50".format(measure), fontsize=title_size)
ax1.tick_params(axis='both', labelsize=tick_size)
ax1.set_xlabel("Batch Size", fontsize=label_size)



# ## tick, lim
# plt.tick_params(labelsize=tick_size)
# xticks_label = [1, 50, 100, 150, 200, 300]
# # zoom1
# # xticks_label = [0, 1, 5, 15]

# plt.xticks(xticks_label, xticks_label)

yticks_label = [0.0, 1.0]
plt.yticks(yticks_label, yticks_label)

# x_lim = [-1, 160]
# # zoom1
# # x_lim = [-1, 5]
# y_lim = [10, 100]

# ax1.set_xlim(x_lim)
# ax1.set_ylim(y_lim)


#plt.yticks(yticks_label, yticks_label)
#ax1.set_ylim(y_lim_bf)
ax1.set_ylabel(" ", fontsize=label_size)
# ax1.set_ylabel("Top-1 Recall (%)", fontsize=label_size)


    
## Legend
handles_ord, labels_ord = plt.gca().get_legend_handles_labels()
last_idx = len(labels) * 2
#last_idx = len(labels) - 1
order = [i for i in range(last_idx)]
#order.insert(0,last_idx)

ax1.legend([handles_ord[idx] for idx in order],[labels_ord[idx] for idx in order], handlelength=2,
           fontsize=legend_size, loc='center left', bbox_to_anchor=(1.0, 0.5))
# ax1.get_legend().remove()
ax1.grid(True)

plt.savefig(f'{output_dir}/R@1_all.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)


# In[ ]:


plt.close('all')


# ## Draw logging plots

# In[ ]:


# Create sample data
logging_data = {}
for batch_selection in batch_selections:
    logging_data.update({
        f'{batch_selection}_step': res[batch_selection]["true_loss"][0].keys(),
    })
    for B_i, B in enumerate(Bs):
        if batch_selection == "full":
            B_i = 0
        logging_data.update({
            f'{batch_selection}_{B}_true_loss': res[batch_selection]["true_loss"][B_i].values(),
            f'{batch_selection}_{B}_loss': res[batch_selection]["loss"][B_i].values(),
            f'{batch_selection}_{B}_utov_R@1': [l["image_to_text_R@1"] for l in res[batch_selection]["metrics"][B_i].values()],
            f'{batch_selection}_{B}_vtou_R@1': [l["text_to_image_R@1"] for l in res[batch_selection]["metrics"][B_i].values()],
        })
logging_data_df = pd.DataFrame(data=logging_data)
logging_data_df.to_csv(f'{output_dir}/logging_data.csv', sep='|', encoding='utf-8-sig', index=False)
logging_data_df


# In[ ]:


### Logging Plot ###

labels = batch_selections

for i, batch_selection in enumerate(labels):

    for measure in ["true_loss", "loss", "utov_R@1", "vtou_R@1"]:
        fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
        line_width=2
        markersize=10

        for B_i, B in enumerate(Bs):
            logging_data_df.plot(x=f'{batch_selection}_step', y=f'{batch_selection}_{B}_{measure}', marker='s', style="--", markersize=markersize, linewidth=line_width, c=colors[(i+B_i)%len(colors)], label=f"{batch_selection}_{measure}_{B}", ax=ax1)
        # logging_data_df.plot(x=f'{batch_selection}_batch', y=f'{batch_selection}_loss', marker='^', markersize=markersize, linewidth=line_width, c=colors[i], label=labels[i]+"_loss", ax=ax1)
        # plt.axhline(100, linestyle='--')

        ## title, axis label
        # plt.title("{} train T->I, RN50".format(measure), fontsize=title_size)
        ax1.tick_params(axis='both', labelsize=tick_size)
        ax1.set_xlabel("step", fontsize=label_size)



        # ## tick, lim
        # plt.tick_params(labelsize=tick_size)
        # xticks_label = [1, 50, 100, 150, 200, 300]
        # # zoom1
        # # xticks_label = [0, 1, 5, 15]

        # plt.xticks(xticks_label, xticks_label)

        if "R@1" in measure:
            yticks_label = [0.0, 1.0]
        else:
            yticks_label = [0.0, 2.0, 3.0, 4.0, 5.0]
        plt.yticks(yticks_label, yticks_label)

        # x_lim = [-1, 160]
        # # zoom1
        # # x_lim = [-1, 5]
        # y_lim = [10, 100]

        # ax1.set_xlim(x_lim)
        # ax1.set_ylim(y_lim)


        #plt.yticks(yticks_label, yticks_label)
        #ax1.set_ylim(y_lim_bf)
        ax1.set_ylabel(" ", fontsize=label_size)
        # ax1.set_ylabel("Top-1 Recall (%)", fontsize=label_size)


                
        ## Legend
        handles_ord, labels_ord = plt.gca().get_legend_handles_labels()
        last_idx = len(Bs)
        #last_idx = len(labels) - 1
        order = [i for i in range(last_idx)]
        #order.insert(0,last_idx)

        if batch_selection == "full":
            ax1.get_legend().remove()
        else:
            ax1.legend([handles_ord[idx] for idx in order],[labels_ord[idx] for idx in order], handlelength=2,
                    fontsize=legend_size, loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax1.grid(True)

        plt.savefig(f'{output_dir}/logging_{batch_selection}_{measure}.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)
        plt.close('all')


# In[ ]:


labels = batch_selections

for measure in ["true_loss", "loss", "utov_R@1", "vtou_R@1"]:
    for B_i, B in enumerate(Bs):

        fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
        line_width=2
        markersize=10

        for i, batch_selection in enumerate(labels):
            logging_data_df.plot(x=f'{batch_selection}_step', y=f'{batch_selection}_{B}_{measure}', marker='s', style="--", markersize=markersize, linewidth=line_width, c=colors[i], label=f"{batch_selection}_{measure}_{B}", ax=ax1)
        # logging_data_df.plot(x=f'{batch_selection}_batch', y=f'{batch_selection}_loss', marker='^', markersize=markersize, linewidth=line_width, c=colors[i], label=labels[i]+"_loss", ax=ax1)
        # plt.axhline(100, linestyle='--')

        ## title, axis label
        # plt.title("{} train T->I, RN50".format(measure), fontsize=title_size)
        ax1.tick_params(axis='both', labelsize=tick_size)
        ax1.set_xlabel("step", fontsize=label_size)



        # ## tick, lim
        # plt.tick_params(labelsize=tick_size)
        # xticks_label = [1, 50, 100, 150, 200, 300]
        # # zoom1
        # # xticks_label = [0, 1, 5, 15]

        # plt.xticks(xticks_label, xticks_label)

        if "R@1" in measure:
            yticks_label = [0.0, 1.0]
        else:
            yticks_label = [0.0, 2.0, 3.0, 4.0, 5.0]
        plt.yticks(yticks_label, yticks_label)

        # x_lim = [-1, 160]
        # # zoom1
        # # x_lim = [-1, 5]
        # y_lim = [10, 100]

        # ax1.set_xlim(x_lim)
        # ax1.set_ylim(y_lim)


        #plt.yticks(yticks_label, yticks_label)
        #ax1.set_ylim(y_lim_bf)
        ax1.set_ylabel(" ", fontsize=label_size)
        # ax1.set_ylabel("Top-1 Recall (%)", fontsize=label_size)


                
        ## Legend
        handles_ord, labels_ord = plt.gca().get_legend_handles_labels()
        last_idx = len(labels)
        #last_idx = len(labels) - 1
        order = [i for i in range(last_idx)]
        #order.insert(0,last_idx)

        if batch_selection == "full":
            ax1.get_legend().remove()
        else:
            ax1.legend([handles_ord[idx] for idx in order],[labels_ord[idx] for idx in order], handlelength=2,
                    fontsize=legend_size, loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax1.grid(True)

        plt.savefig(f'{output_dir}/logging_{B}_{measure}.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)
        plt.close('all')
