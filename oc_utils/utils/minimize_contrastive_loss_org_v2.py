# %%
import os
from random import randint
import math

import json
from tqdm import tqdm
import torch
import torch.nn as nn
import random
import numpy as np
import itertools
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime
# try:
#     import wandb
# except ImportError:
wandb = None

torch.backends.cudnn.benchmark = True
device = 'cuda:0'


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


def mini_batch_loss(u, v, batch_idxs=None, B=2):
    loss = 0
    if batch_idxs == None:
        # find all possible batches of size B
        batch_idxs = list(itertools.combinations([i for i in range(u.shape[0])], B))
    n = len(batch_idxs)
    # print("len(batch_idxs)", n, len(batch_idxs[0]))
    for batch_idx in batch_idxs:
        u_batch = u[list(batch_idx)]
        v_batch = v[list(batch_idx)]
        loss += clip_batch_loss(u_batch, v_batch)
    return loss/n



# %%

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--N", type=int, default=4)#20)
# parser.add_argument("--d", type=int, default=2)#32)
# parser.add_argument("--lr_full", type=float, default=0.5)
# # parser.add_argument("--lr_scaling", default=False, action='store_true')
# parser.add_argument("--num_steps", type=int, default=10000)
# parser.add_argument("--logging_step_ratio", type=float, default=0.1)
# # parser.add_argument("--gradient_accumulation", default=False, action='store_true')
# parser.add_argument("--batch_size_list", nargs='+', type=int, default=[2])
# parser.add_argument("--wandb_notes", default="", type=str, help="additional wandb logging note")
# args = parser.parse_args()

# print("N", args.N)
# print("d", args.d)
# print("lr_full", args.lr_full)
# # print("lr_scaling", args.lr_scaling)
# print("num_steps", args.num_steps)
# print("logging_step_ratio", args.logging_step_ratio)
# # print("gradient_accumulation", args.gradient_accumulation)
# print("batch_size_list", args.batch_size_list)


N = 4 #args.N
d = 2 #args.d
lr_full = 0.5 #args.lr_full
batch_size_list = [2] #args.batch_size_list
NUM_STEPS = 10000 #args.num_steps
num_steps_factor = 5 if N > 12 else 2
# logging_step = int(NUM_STEPS * args.logging_step_ratio)
logging_step = 100
time_tag = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
exp_tag = f"{time_tag}_hst_N{N}_d{d}_lr{lr_full}_s{NUM_STEPS}x{num_steps_factor}_all_Bs{batch_size_list}"
output_dir = f"output_uni/{exp_tag}"
os.makedirs(output_dir, exist_ok=True)
print("output_dir:", output_dir)

def get_unimodal_embeddings(N, d):
    u = torch.randn((N, d), requires_grad=True, device=device)
    v = u.clone().detach() + torch.randn((N, d), requires_grad=False, device=device)
    # v = torch.randn((N, d), requires_grad=True, device=device)
    with torch.no_grad():
        u.data = F.normalize(u.data, p=2.0, dim=1)
        v.data = F.normalize(v.data, p=2.0, dim=1)
    return (u, v)

def get_bimodal_embeddings(N, d):
    u = torch.randn((N, d), requires_grad=True, device=device)
    v = torch.randn((N, d), requires_grad=True, device=device)
    with torch.no_grad():
        u.data = F.normalize(u.data, p=2.0, dim=1)
        v.data = F.normalize(v.data, p=2.0, dim=1)
    return (u, v)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", plot_cbar=True, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = None
    if plot_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_heatmap(z, filename=None, plot_cbar=True):
    N = z.shape[0]
    fig, ax = plt.subplots()
    im, cbar = heatmap(z, np.arange(N), np.arange(N), ax=ax, plot_cbar=plot_cbar, cmap="YlGn", vmin=-1.0, vmax=1.0)
    texts = annotate_heatmap(im, valfmt="")#"{x:.4f}")
    fig.tight_layout()
    if filename is not None:
        plt.savefig(f'{filename}.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)
        plt.savefig(f'{filename}.pdf', format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close('all')


# %%


from batch_utils import create_inverse_greedy_batches_with_K, create_balance_greedy_batches, create_greedy_batches
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


# %%
B = 2
batch_idxs = get_random_batch_idxs(N, B)
set_seed(42)

# %%

def plot_embeddings(u, v, d, is_proj=True, filename=None):
    circle = plt.Circle((0, 0), 1, color='black', fill=False)
    if is_proj:
        # project down to 2d to visualize
        linear_projection = torch.randn(d, 2)
        proj_u = F.normalize(u.to('cpu')@linear_projection.detach(), p=2.0, dim=1)
        proj_v = F.normalize(v.to('cpu')@linear_projection.detach(), p=2.0, dim=1)
    else:
        proj_u = u.to('cpu')
        proj_v = v.to('cpu')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.add_patch(circle)
    ax.scatter(proj_u[:, 0].detach().numpy(), proj_u[:, 1].detach().numpy(), color='blue', label='u', marker="+", s=150)
    for i in range(proj_u.shape[0]):
        ax.text(proj_u[i, 0], proj_u[i, 1], f'u_{i}')
    ax.scatter(proj_v[:, 0].detach().numpy(), proj_v[:, 1].detach().numpy(), color='red', label='v')
    for i in range(proj_v.shape[0]):
        ax.text(proj_v[i, 0], proj_v[i, 1], f'v_{i}')
    ax.legend(loc='best')
    plt.grid()
    plt.show()
    if filename is not None:
        plt.savefig(f'{filename}.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    return proj_u, proj_v


# %%
#u3 = torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0]]).type(torch.FloatTensor)
#v3 = torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0]]).type(torch.FloatTensor)
u3, v3 = get_bimodal_embeddings(N, d)  # (N, d)
print(u3)
_, _ = plot_embeddings(u3, v3, d, is_proj=False)


# %%
param_list = [u3, v3]
uv = u3 @ v3.T

# %%
def batch_selection_utility(u3, v3):

    # batch sampling
    uv = u3 @ v3.T
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(3, affinity='precomputed')
    temp = sc.fit_predict(uv)


    combinations = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]

    max_util_value = -100000
    max_comb_idx = -1
    for comb_idx, (comb1, comb2) in enumerate(combinations):
        util_value = uv[comb1[0], comb1[1]] + uv[comb1[1], comb1[0]] + uv[comb2[0], comb2[1]] + uv[comb2[1], comb2[0]]
        #print(f"[{comb1, comb2}] utility value : {util_value}")
        if util_value > max_util_value:
            max_util_value = util_value
            max_comb_idx = comb_idx

    #print(f"max_combination : {combinations[max_comb_idx]} / max_utility_value : {max_util_value:.3f}")
    batch_idxs = combinations[max_comb_idx]

    return batch_idxs

# %%
output_dir

# %%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

"""
- radius 리스트를 입력받아서, 각 반지름의 원에 위치하는 n개의 샘플을 뽑아서 x, y 리스트들을 리턴해줍니다. 
"""
def return_circle_xy(rs, n):
    xs, ys = [], []
    for r in rs:
        for i in range(0, n):
            angle = np.pi * np.random.uniform(0, 2)
            xs.append( r*np.cos(angle) + np.random.random())
            ys.append( r*np.sin(angle) + np.random.random())
    return xs, ys

x, y = return_circle_xy([10, 5], 500)
df = pd.DataFrame({"x":x, "y":y})

from sklearn.cluster import SpectralClustering, AgglomerativeClustering

f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
f.set_size_inches((10, 4)) 

# spectral clustering and scattering
CluNums = SpectralClustering(n_clusters=2, n_init=10).fit_predict(df)
axes[0].scatter(x, y, c=CluNums, cmap=plt.cm.rainbow, alpha=0.3)
axes[0].set_title("Spectral Clustering")

# agglomerative clustering and scattering
CluNums = AgglomerativeClustering(n_clusters=2).fit_predict(df)
axes[1].scatter(x, y, c=CluNums, cmap=plt.cm.rainbow, alpha=0.3)
axes[1].set_title("Agglomerative Clustering")

# %%
u3, v3 = get_bimodal_embeddings(N, d)  # (N, d)

uv = u3 @ v3.T
from sklearn.cluster import SpectralClustering
sc = SpectralClustering(2, affinity='precomputed', n_init=100, assign_labels='discretize')

print(f"uv : {uv.cpu().detach().numpy()}")

#temp = sc.fit_predict([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])

temp = sc.fit_predict(uv.tolist())

#temp

# %%
temp


from sklearn.cluster import SpectralClustering
import numpy as np
X = np.array([[1, 1], [2, 1], [1, 0],
               [4, 7], [3, 5], [3, 6]])
clustering = SpectralClustering(n_clusters=2,
        assign_labels='discretize',
        random_state=0).fit_predict(X)

# %%
batch_idxs = get_random_batch_idxs(N, B)

u3, v3 = get_bimodal_embeddings(N, d)  # (N, d)
param_list = [u3, v3]

loss_dict, true_loss_dict = {}, {}
optimizer = torch.optim.SGD(param_list, lr=lr_full)
for step in tqdm(range(1000)): #tqdm(range(NUM_STEPS*num_steps_factor)):

    batch_selection = 'utility'

    strategy = 'precomputed'

    if strategy == 'precomputed':
        # batch sampling
        uv = u3 @ v3.T
        from sklearn.cluster import SpectralClustering
        sc = SpectralClustering(2, affinity='precomputed')

        uv_ii = torch.diag(uv)
        u_vDiff = uv - uv_ii.unsqueeze(1)  # (N, N) - (N, 1)
        uDiff_v = uv - uv_ii.unsqueeze(0)  # (N, N) - (1, N) 

        sum_ij = u_vDiff + uDiff_v

        uv_np = sum_ij.cpu().detach().numpy()
        print("uv_np: ", uv_np)

        def is_pos_def(x):
            return np.all(np.linalg.eigvals(x) > 0)

        def isPSD(A, tol=1e-8):
            E = np.linalg.eigvalsh(A)
            return np.all(E > -tol)

        sum_triu_tril = np.triu(uv_np) + np.tril(uv_np).T
        keep_diag = sum_triu_tril[np.arange(sum_triu_tril.shape[0]), np.arange(sum_triu_tril.shape[0])]
        sum_triu_tril[np.arange(sum_triu_tril.shape[0]), np.arange(sum_triu_tril.shape[0])] = 0

        new_triu_tril = sum_triu_tril + sum_triu_tril.T
        new_triu_tril[np.arange(sum_triu_tril.shape[0]), np.arange(sum_triu_tril.shape[0])] = keep_diag

        new_triu_tril -= np.min(new_triu_tril)

        new_triu_tril += np.eye(new_triu_tril.shape[0]) * 10

        print(f"new_triu_tril : {new_triu_tril}")

        print(f"is_pos_def(new_triu_tril) : {isPSD(new_triu_tril)}")

        temp = sc.fit_predict(new_triu_tril)

        batch_idxs = []
        for t in set(temp):
            batch_idx = np.where(np.array(temp) == t)[0].tolist()
            if len(batch_idx) > 1:
                batch_idxs.append(batch_idx)
        print(f"batch_idxs : {batch_idxs}")

    else:
        sc = SpectralClustering(2)
        temp = sc.fit_predict(u3.cpu().detach().numpy())

        batch_idxs = []
        for t in set(temp):
            batch_idx = np.where(np.array(temp) == t)[0].tolist()
            if len(batch_idx) > 1:
                batch_idxs.append(batch_idx)
        print(f"batch_idxs : {batch_idxs}")

    # combinations = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]

    # max_util_value = -100000
    # max_comb_idx = -1
    # for comb_idx, (comb1, comb2) in enumerate(combinations):
    #     util_value = uv[comb1[0], comb1[1]] + uv[comb1[1], comb1[0]] + uv[comb2[0], comb2[1]] + uv[comb2[1], comb2[0]]
    #     #print(f"[{comb1, comb2}] utility value : {util_value}")
    #     if util_value > max_util_value:
    #         max_util_value = util_value
    #         max_comb_idx = comb_idx

    # #print(f"max_combination : {combinations[max_comb_idx]} / max_utility_value : {max_util_value:.3f}")
    # batch_idxs = combinations[max_comb_idx]
    

    optimizer.zero_grad()
    loss = mini_batch_loss(u3, v3, batch_idxs=batch_idxs)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        u3.data = F.normalize(u3.data, p=2.0, dim = 1)
        v3.data = F.normalize(v3.data, p=2.0, dim = 1)
    if step %logging_step == 0 or step == NUM_STEPS-1:
        print("B={} | Step={} | Loss={} | Grad Norm={}".format(B, step, loss, torch.norm(u3.grad.data)))
        torch.save(u3, f"{output_dir}/u3_{batch_selection}_mini_batch_B{B}_{step}.pt")
        torch.save(v3, f"{output_dir}/v3_{batch_selection}_mini_batch_B{B}_{step}.pt")
        loss_dict[step], true_loss_dict[step] = loss.item(), clip_batch_loss(u3, v3).detach().item()
        if wandb:
            assert wandb is not None, 'Please install wandb.'
            wandb.log({
                'step': step,
                'loss': loss_dict[step],
                'true_loss': true_loss_dict[step],
            })
with open(f'{output_dir}/loss_{batch_selection}_mini_batch_B{B}.json', 'w') as f:
    f.write(json.dumps(loss_dict, indent=4))
with open(f'{output_dir}/true_loss_{batch_selection}_mini_batch_B{B}.json', 'w') as f:
    f.write(json.dumps(true_loss_dict, indent=4))

# first see if u = v
# print("||u-v|| = {}".format(torch.norm(u3-v3)))
# now see if the inner products are equal
z3 = (u3 @ v3.T).detach().cpu()
# print("u^T v={}".format(z3))
torch.save(z3, f"{output_dir}/z3_{batch_selection}_mini_batch_B{B}_{step}.pt")

u3_proj, v3_proj = plot_embeddings(u3, v3, d, is_proj=False, filename=f'{output_dir}/plot_embeddings_{batch_selection}_mini_batch_B{B}')
plot_heatmap(z3, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z3_fixed_mini_batch_B{B}_{batch_selection}_w_cbar"), plot_cbar=True)
plot_heatmap(z3, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z3_fixed_mini_batch_B{B}_{batch_selection}_wo_cbar"), plot_cbar=False)

if wandb:
    z3_w_cbar = wandb.Image(os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z3_fixed_mini_batch_B{B}_{batch_selection}_w_cbar" + ".png"))
    z3_wo_cbar = wandb.Image(os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z3_fixed_mini_batch_B{B}_{batch_selection}_wo_cbar" + ".png"))
    wandb.log({"heatmap_w_cbar": [z3_w_cbar]})
    wandb.log({"heatmap_wo_cbar": [z3_wo_cbar]})
    wandb.finish()


# %%
# batch
combinations = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]

max_util_value = -100000
max_comb_idx = -1
for comb_idx, (comb1, comb2) in enumerate(combinations):
    util_value = uv[comb1[0], comb1[1]] + uv[comb1[1], comb1[0]] + uv[comb2[0], comb2[1]] + uv[comb2[1], comb2[0]]
    print(f"[{comb1, comb2}] utility value : {util_value}")
    if util_value > max_util_value:
        max_util_value = util_value
        max_comb_idx = comb_idx

print(f"max_combination : {combinations[max_comb_idx]} / max_utility_value : {max_util_value}")

loss_dict, true_loss_dict = {}, {}
optimizer = torch.optim.SGD(param_list, lr=args.lr_full)
for step in tqdm(range(NUM_STEPS*num_steps_factor), desc=f"[B:{B}] "):

    z = u3 @ v3.T
    z_ii = torch.diag(z)
    u_vDiff = z - z_ii.unsqueeze(1)  # (N, N) - (N, 1)
    uDiff_v = z - z_ii.unsqueeze(0)  # (N, N) - (1, N) 

    sum_ij = u_vDiff + uDiff_v

    if batch_selection == 'osgd':
        total_loss = 0
        for _ in range(len(batch_idxs)): # different meaning of epochs in here
            with torch.no_grad():
                batch_idx = create_greedy_batches(N, B, u3.detach(), v3.detach(), 1.0, device=device, D=d, max_B=B, max_n_batch=1)[0]
            optimizer.zero_grad()
            u_batch = u3[list(batch_idx)]
            v_batch = v3[list(batch_idx)]
            loss = clip_batch_loss(u_batch, v_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss
        loss = total_loss / len(batch_idxs)
    else:
        with torch.no_grad():
            if batch_selection == 'f':
                pass
            elif batch_selection == 's':
                batch_idxs = get_random_batch_idxs(N, B)
            elif batch_selection == 'g':
                batch_idxs = create_greedy_batches(N, B, u3.detach(), v3.detach(), 1.0, device=device, D=d)
            elif batch_selection == 'bg':
                batch_idxs = create_balance_greedy_batches(N, B, u3.detach(), v3.detach(), 1.0, device=device, D=d)
            elif batch_selection == 'ig':
                batch_idxs = create_inverse_greedy_batches_with_K(N, B, u3.detach(), v3.detach(), 1.0, device=device, D=d)
            else:
                raise NotImplementedError(f'{batch_selection} is not available for batching')
        optimizer.zero_grad()
        loss = mini_batch_loss(u3, v3, batch_idxs=batch_idxs)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        u3.data = F.normalize(u3.data, p=2.0, dim = 1)
        v3.data = F.normalize(v3.data, p=2.0, dim = 1)
    if step %logging_step == 0 or step == NUM_STEPS-1:
        # print("B={} | Step={} | Loss={} | Grad Norm={}".format(B, step, loss, torch.norm(u3.grad.data)))
        torch.save(u3, f"{output_dir}/u3_{batch_selection}_mini_batch_B{B}_{step}.pt")
        torch.save(v3, f"{output_dir}/v3_{batch_selection}_mini_batch_B{B}_{step}.pt")
        loss_dict[step], true_loss_dict[step] = loss.item(), clip_batch_loss(u3, v3).detach().item()
        if wandb:
            assert wandb is not None, 'Please install wandb.'
            wandb.log({
                'step': step,
                'loss': loss_dict[step],
                'true_loss': true_loss_dict[step],
            })
with open(f'{output_dir}/loss_{batch_selection}_mini_batch_B{B}.json', 'w') as f:
    f.write(json.dumps(loss_dict, indent=4))
with open(f'{output_dir}/true_loss_{batch_selection}_mini_batch_B{B}.json', 'w') as f:
    f.write(json.dumps(true_loss_dict, indent=4))

# first see if u = v
# print("||u-v|| = {}".format(torch.norm(u3-v3)))
# now see if the inner products are equal
z3 = (u3 @ v3.T).detach().cpu()
# print("u^T v={}".format(z3))
torch.save(z3, f"{output_dir}/z3_{batch_selection}_mini_batch_B{B}_{step}.pt")

u3_proj, v3_proj = plot_embeddings(u3, v3, d, filename=f'{output_dir}/plot_embeddings_{batch_selection}_mini_batch_B{B}')
plot_heatmap(z3, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z3_fixed_mini_batch_B{B}_{batch_selection}_w_cbar"), plot_cbar=True)
plot_heatmap(z3, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z3_fixed_mini_batch_B{B}_{batch_selection}_wo_cbar"), plot_cbar=False)

if wandb:
    z3_w_cbar = wandb.Image(os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z3_fixed_mini_batch_B{B}_{batch_selection}_w_cbar" + ".png"))
    z3_wo_cbar = wandb.Image(os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z3_fixed_mini_batch_B{B}_{batch_selection}_wo_cbar" + ".png"))
    wandb.log({"heatmap_w_cbar": [z3_w_cbar]})
    wandb.log({"heatmap_wo_cbar": [z3_wo_cbar]})
    wandb.finish()




# %%
param_list = [u3, v3]

z = u3 @ v3.T
z_ii = torch.diag(z)
u_vDiff = z - z_ii.unsqueeze(1)  # (N, N) - (N, 1)
uDiff_v = z - z_ii.unsqueeze(0)  # (N, N) - (1, N) 

sum_ij = u_vDiff + uDiff_v

# %%
u_vDiff

# %%



#batch_selections = ['f', 's', 'g', 'bg', 'ig', 'osgd']
batch_selections = ['g', 'bg', 'ig', 'osgd']
## Now minimize mini-batch loss over all specific batches
for B in batch_size_list:

    batch_idxs = get_random_batch_idxs(N, B)

    for batch_selection in batch_selections:
        set_seed(42)

        # if wandb:
        #     print(f"init wandb logging for {batch_selection} B{B}...")
        #     exp_name = '-'.join([
        #         f"{batch_selection}_B{B}",
        #     ])

        #     wandb.init(
        #         entity="krafton_clap",
        #         project="simulations_spectral_clustering",
        #         group=f"{exp_tag}",
        #         name=exp_name,
        #         notes=args.wandb_notes,
        #         config=vars(args)
        #     )

        u3, v3 = get_unimodal_embeddings(N, d)  # (N, d)
        param_list = [u3, v3]

        loss_dict, true_loss_dict = {}, {}
        optimizer = torch.optim.SGD(param_list, lr=args.lr_full)
        for step in tqdm(range(NUM_STEPS*num_steps_factor), desc=f"[batch_selection:'{batch_selection}' | B:{B}] "):

            z = u3 @ v3.T
            z_ii = torch.diag(z)
            u_vDiff = z - z_ii.unsqueeze(1)  # (N, N) - (N, 1)
            uDiff_v = z - z_ii.unsqueeze(0)  # (N, N) - (1, N) 

            sum_ij = u_vDiff + uDiff_v

            if batch_selection == 'osgd':
                total_loss = 0
                for _ in range(len(batch_idxs)): # different meaning of epochs in here
                    with torch.no_grad():
                        batch_idx = create_greedy_batches(N, B, u3.detach(), v3.detach(), 1.0, device=device, D=d, max_B=B, max_n_batch=1)[0]
                    optimizer.zero_grad()
                    u_batch = u3[list(batch_idx)]
                    v_batch = v3[list(batch_idx)]
                    loss = clip_batch_loss(u_batch, v_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss
                loss = total_loss / len(batch_idxs)
            else:
                with torch.no_grad():
                    if batch_selection == 'f':
                        pass
                    elif batch_selection == 's':
                        batch_idxs = get_random_batch_idxs(N, B)
                    elif batch_selection == 'g':
                        batch_idxs = create_greedy_batches(N, B, u3.detach(), v3.detach(), 1.0, device=device, D=d)
                    elif batch_selection == 'bg':
                        batch_idxs = create_balance_greedy_batches(N, B, u3.detach(), v3.detach(), 1.0, device=device, D=d)
                    elif batch_selection == 'ig':
                        batch_idxs = create_inverse_greedy_batches_with_K(N, B, u3.detach(), v3.detach(), 1.0, device=device, D=d)
                    else:
                        raise NotImplementedError(f'{batch_selection} is not available for batching')
                optimizer.zero_grad()
                loss = mini_batch_loss(u3, v3, batch_idxs=batch_idxs)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                u3.data = F.normalize(u3.data, p=2.0, dim = 1)
                v3.data = F.normalize(v3.data, p=2.0, dim = 1)
            if step %logging_step == 0 or step == NUM_STEPS-1:
                # print("B={} | Step={} | Loss={} | Grad Norm={}".format(B, step, loss, torch.norm(u3.grad.data)))
                torch.save(u3, f"{output_dir}/u3_{batch_selection}_mini_batch_B{B}_{step}.pt")
                torch.save(v3, f"{output_dir}/v3_{batch_selection}_mini_batch_B{B}_{step}.pt")
                loss_dict[step], true_loss_dict[step] = loss.item(), clip_batch_loss(u3, v3).detach().item()
                if wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({
                        'step': step,
                        'loss': loss_dict[step],
                        'true_loss': true_loss_dict[step],
                    })
        with open(f'{output_dir}/loss_{batch_selection}_mini_batch_B{B}.json', 'w') as f:
            f.write(json.dumps(loss_dict, indent=4))
        with open(f'{output_dir}/true_loss_{batch_selection}_mini_batch_B{B}.json', 'w') as f:
            f.write(json.dumps(true_loss_dict, indent=4))

        # first see if u = v
        # print("||u-v|| = {}".format(torch.norm(u3-v3)))
        # now see if the inner products are equal
        z3 = (u3 @ v3.T).detach().cpu()
        # print("u^T v={}".format(z3))
        torch.save(z3, f"{output_dir}/z3_{batch_selection}_mini_batch_B{B}_{step}.pt")

        u3_proj, v3_proj = plot_embeddings(u3, v3, d, filename=f'{output_dir}/plot_embeddings_{batch_selection}_mini_batch_B{B}')
        plot_heatmap(z3, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z3_fixed_mini_batch_B{B}_{batch_selection}_w_cbar"), plot_cbar=True)
        plot_heatmap(z3, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z3_fixed_mini_batch_B{B}_{batch_selection}_wo_cbar"), plot_cbar=False)

        if wandb:
            z3_w_cbar = wandb.Image(os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z3_fixed_mini_batch_B{B}_{batch_selection}_w_cbar" + ".png"))
            z3_wo_cbar = wandb.Image(os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z3_fixed_mini_batch_B{B}_{batch_selection}_wo_cbar" + ".png"))
            wandb.log({"heatmap_w_cbar": [z3_w_cbar]})
            wandb.log({"heatmap_wo_cbar": [z3_wo_cbar]})
            wandb.finish()
