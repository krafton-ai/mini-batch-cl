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


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=20)
parser.add_argument("--d", type=int, default=32)
parser.add_argument("--lr_full", type=float, default=0.5)
# parser.add_argument("--lr_scaling", default=False, action='store_true')
parser.add_argument("--num_steps", type=int, default=10000)
parser.add_argument("--logging_step_ratio", type=float, default=0.1)
# parser.add_argument("--gradient_accumulation", default=False, action='store_true')
parser.add_argument("--batch_size_list", nargs='+', type=int, default=[2])
args = parser.parse_args()

print("N", args.N)
print("d", args.d)
print("lr_full", args.lr_full)
# print("lr_scaling", args.lr_scaling)
print("num_steps", args.num_steps)
print("logging_step_ratio", args.logging_step_ratio)
# print("gradient_accumulation", args.gradient_accumulation)
print("batch_size_list", args.batch_size_list)


N = args.N
d = args.d
lr_full = args.lr_full
batch_size_list = args.batch_size_list
NUM_STEPS = args.num_steps
num_steps_factor = 5 if N > 12 else 2
logging_step = int(NUM_STEPS * args.logging_step_ratio)
output_dir = f"output/hst_N{N}_d{d}_lr{lr_full}_s{NUM_STEPS}x{num_steps_factor}_all_Bs{batch_size_list}"
os.makedirs(output_dir, exist_ok=True)
step = NUM_STEPS*num_steps_factor - 1

def get_embeddings(N, d):
    u = torch.randn((N, d), requires_grad=True, device=device)
    v = torch.randn((N, d), requires_grad=True, device=device)
    with torch.no_grad():
        u.data = F.normalize(u.data, p=2.0, dim=1)
        v.data = F.normalize(v.data, p=2.0, dim=1)
    return (u, v)

def plot_embeddings(u, v, d, filename=None):
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
    if filename is not None:
        plt.savefig(f'{filename}.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    return proj_u, proj_v


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


## First minimize full_batch loss
# set_seed(42)
# u1, v1 = get_embeddings(N, d)
# param_list = [u1, v1]

# loss_dict = {}
# optimizer = torch.optim.SGD(param_list, lr=args.lr_full)
# for step in range(NUM_STEPS*num_steps_factor):
#     optimizer.zero_grad()
#     loss = clip_batch_loss(u1, v1)
#     loss.backward()
#     optimizer.step()
#     with torch.no_grad():
#         u1.data = F.normalize(u1.data, p=2.0, dim = 1)
#         v1.data = F.normalize(v1.data, p=2.0, dim = 1)
#     if step %logging_step == 0 or step == NUM_STEPS-1:
#         # print("Step={} | Loss={} | Grad Norm={}".format(step, loss, torch.norm(u1.grad.data)))
#         torch.save(u1, f"{output_dir}/u1_full_batch_{step}.pt")
#         torch.save(v1, f"{output_dir}/v1_full_batch_{step}.pt")
#         loss_dict[step] = loss.item()
# with open(f'{output_dir}/loss_full_batch.json', 'w') as f:
#     f.write(json.dumps(loss_dict, indent=4))

uv_norms = {}
# ## Check if it is ETF (It works!)
# # first see if u = v
# # print("||u-v|| = {}".format(torch.norm(u1-v1)))
# # now see if the inner products are equal
# z1 = (u1 @ v1.T).detach().cpu()
# # print("u^T v={}".format(z1))
# torch.save(z1, f"{output_dir}/z1_full_batch_{step}.pt")
# z1 = torch.load(f"{output_dir}/z1_full_batch_{step}.pt")
step = 19000
u1 = torch.load(f"{output_dir}/u1_full_batch_{step}.pt")
v1 = torch.load(f"{output_dir}/v1_full_batch_{step}.pt")
uv_norms[f'full_{step}'] = torch.norm(u1-v1).item()
uu1 = (u1 @ u1.T).detach().cpu()
vv1 = (v1 @ v1.T).detach().cpu()
plot_heatmap(uu1, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_uu1_full_batch_w_cbar"), plot_cbar=True)
plot_heatmap(uu1, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_uu1_full_batch_wo_cbar"), plot_cbar=False)
plot_heatmap(vv1, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_vv1_full_batch_w_cbar"), plot_cbar=True)
plot_heatmap(vv1, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_vv1_full_batch_wo_cbar"), plot_cbar=False)
# u1_proj, v1_proj = plot_embeddings(u1, v1, d, filename=f'{output_dir}/plot_embeddings_full_batch')
# plot_heatmap(z1, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z1_full_batch_w_cbar"), plot_cbar=True)
# plot_heatmap(z1, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z1_full_batch_wo_cbar"), plot_cbar=False)


## Now minimize mini-batch loss over all (NcB) batches
for B in batch_size_list:
    # set_seed(42)
    # u2, v2 = get_embeddings(N, d)
    # param_list = [u2, v2]

    # loss_dict, true_loss_dict = {}, {}
    # optimizer = torch.optim.SGD(param_list, lr=args.lr_full)
    # for step in tqdm(range(NUM_STEPS*num_steps_factor), desc=f"[batch_selection:'All NcB' | B:{B}] "):
    #     optimizer.zero_grad()
    #     loss = mini_batch_loss(u2, v2, B=B)
    #     loss.backward()
    #     optimizer.step()
    #     with torch.no_grad():
    #         u2.data = F.normalize(u2.data, p=2.0, dim = 1)
    #         v2.data = F.normalize(v2.data, p=2.0, dim = 1)
    #     if step %logging_step == 0 or step == NUM_STEPS-1:
    #         # print("B={} | Step={} | Loss={} | Grad Norm={}".format(B, step, loss, torch.norm(u2.grad.data)))
    #         torch.save(u2, f"{output_dir}/u2_NcB_mini_batch_B{B}_{step}.pt")
    #         torch.save(v2, f"{output_dir}/v2_NcB_mini_batch_B{B}_{step}.pt")
    #         loss_dict[step], true_loss_dict[step] = loss.item(), clip_batch_loss(u2, v2).detach().item()
    # with open(f'{output_dir}/loss_NcB_mini_batch_B{B}.json', 'w') as f:
    #     f.write(json.dumps(loss_dict, indent=4))
    # with open(f'{output_dir}/true_loss_NcB_mini_batch_B{B}.json', 'w') as f:
    #     f.write(json.dumps(true_loss_dict, indent=4))

    # ## Check if it is ETF (It works!)
    # # first see if u = v
    # # print("||u-v|| = {}".format(torch.norm(u2-v2)))
    # # now see if the inner products are equal
    # z2 = (u2 @ v2.T).detach().cpu()
    # # print("u^T v={}".format(z2))
    # torch.save(z2, f"{output_dir}/z2_NcB_mini_batch_B{B}_{step}.pt")
    # z2 = torch.load(f"{output_dir}/z2_NcB_mini_batch_B{B}_{step}.pt")
    u2 = torch.load(f"{output_dir}/u2_NcB_mini_batch_B{B}_{step}.pt")
    v2 = torch.load(f"{output_dir}/v2_NcB_mini_batch_B{B}_{step}.pt")
    uv_norms[f'NcB_B{B}_{step}'] = torch.norm(u2-v2).item()
    uu2 = (u2 @ u2.T).detach().cpu()
    vv2 = (v2 @ v2.T).detach().cpu()
    plot_heatmap(uu2, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_uu2_NcB_mini_batch_B{B}_w_cbar"), plot_cbar=True)
    plot_heatmap(uu2, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_uu2_NcB_mini_batch_B{B}_wo_cbar"), plot_cbar=False)
    plot_heatmap(vv2, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_vv2_NcB_mini_batch_B{B}_w_cbar"), plot_cbar=True)
    plot_heatmap(vv2, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_vv2_NcB_mini_batch_B{B}_wo_cbar"), plot_cbar=False)
    # u2_proj, v2_proj = plot_embeddings(u2, v2, d, filename=f'{output_dir}/plot_embeddings_NcB_mini_batch_B{B}')
    # plot_heatmap(z2, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z2_NcB_mini_batch_B{B}_w_cbar"), plot_cbar=True)
    # plot_heatmap(z2, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z2_NcB_mini_batch_B{B}_wo_cbar"), plot_cbar=False)


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


batch_selections = ['f', 's', 'g', 'bg', 'ig']
## Now minimize mini-batch loss over all specific batches
for B in batch_size_list:

    # batch_idxs = get_random_batch_idxs(N, B)

    for batch_selection in batch_selections:
        # set_seed(42)
        # u3, v3 = get_embeddings(N, d)
        # param_list = [u3, v3]

        # loss_dict, true_loss_dict = {}, {}
        # optimizer = torch.optim.SGD(param_list, lr=args.lr_full)
        # for step in tqdm(range(NUM_STEPS*num_steps_factor), desc=f"[batch_selection:'{batch_selection}' | B:{B}] "):
        #     if batch_selection == 'f':
        #         pass
        #     elif batch_selection == 's':
        #         batch_idxs = get_random_batch_idxs(N, B)
        #     elif batch_selection == 'g':
        #         batch_idxs = create_greedy_batches(N, B, u3, v3, 1.0, device=device, D=d)
        #     elif batch_selection == 'bg':
        #         batch_idxs = create_balance_greedy_batches(N, B, u3, v3, 1.0, device=device, D=d)
        #     elif batch_selection == 'ig':
        #         batch_idxs = create_inverse_greedy_batches_with_K(N, B, u3, v3, 1.0, device=device, D=d)
        #     else:
        #         raise NotImplementedError(f'{batch_selection} is not available for batching')
        #     optimizer.zero_grad()
        #     loss = mini_batch_loss(u3, v3, batch_idxs=batch_idxs)
        #     loss.backward()
        #     optimizer.step()
        #     with torch.no_grad():
        #         u3.data = F.normalize(u3.data, p=2.0, dim = 1)
        #         v3.data = F.normalize(v3.data, p=2.0, dim = 1)
        #     if step %logging_step == 0 or step == NUM_STEPS-1:
        #         # print("B={} | Step={} | Loss={} | Grad Norm={}".format(B, step, loss, torch.norm(u3.grad.data)))
        #         torch.save(u3, f"{output_dir}/u3_{batch_selection}_mini_batch_B{B}_{step}.pt")
        #         torch.save(v3, f"{output_dir}/v3_{batch_selection}_mini_batch_B{B}_{step}.pt")
        #         loss_dict[step], true_loss_dict[step] = loss.item(), clip_batch_loss(u3, v3).detach().item()
        # with open(f'{output_dir}/loss_{batch_selection}_mini_batch_B{B}.json', 'w') as f:
        #     f.write(json.dumps(loss_dict, indent=4))
        # with open(f'{output_dir}/true_loss_{batch_selection}_mini_batch_B{B}.json', 'w') as f:
        #     f.write(json.dumps(true_loss_dict, indent=4))

        # # first see if u = v
        # # print("||u-v|| = {}".format(torch.norm(u3-v3)))
        # # now see if the inner products are equal
        # z3 = (u3 @ v3.T).detach().cpu()
        # # print("u^T v={}".format(z3))
        # torch.save(z3, f"{output_dir}/z3_{batch_selection}_mini_batch_B{B}_{step}.pt")
        # z3 = torch.load(f"{output_dir}/z3_{batch_selection}_mini_batch_B{B}_{step}.pt")
        u3 = torch.load(f"{output_dir}/u3_{batch_selection}_mini_batch_B{B}_{step}.pt")
        v3 = torch.load(f"{output_dir}/v3_{batch_selection}_mini_batch_B{B}_{step}.pt")
        uv_norms[f'{batch_selection}_B{B}_{step}'] = torch.norm(u3-v3).item()
        uu3 = (u3 @ u3.T).detach().cpu()
        vv3 = (v3 @ v3.T).detach().cpu()
        plot_heatmap(uu3, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_uu3_fixed_mini_batch_B{B}_{batch_selection}_w_cbar"), plot_cbar=True)
        plot_heatmap(uu3, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_uu3_fixed_mini_batch_B{B}_{batch_selection}_wo_cbar"), plot_cbar=False)
        plot_heatmap(vv3, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_vv3_fixed_mini_batch_B{B}_{batch_selection}_w_cbar"), plot_cbar=True)
        plot_heatmap(vv3, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_vv3_fixed_mini_batch_B{B}_{batch_selection}_wo_cbar"), plot_cbar=False)
        # u3_proj, v3_proj = plot_embeddings(u3, v3, d, filename=f'{output_dir}/plot_embeddings_{batch_selection}_mini_batch_B{B}')
        # plot_heatmap(z3, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z3_fixed_mini_batch_B{B}_{batch_selection}_w_cbar"), plot_cbar=True)
        # plot_heatmap(z3, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z3_fixed_mini_batch_B{B}_{batch_selection}_wo_cbar"), plot_cbar=False)
with open(f'{output_dir}/uv_norms.json', 'w') as f:
    f.write(json.dumps(uv_norms, indent=4))
