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

def osgd_NcB_batches(u, v, B=2):
    max_loss = -1
    max_loss_batch_idx = None
    batch_idxs = list(itertools.combinations([i for i in range(u.shape[0])], B))
    n = len(batch_idxs)
    with torch.no_grad():
        for batch_idx in batch_idxs:
            u_batch = u[list(batch_idx)]
            v_batch = v[list(batch_idx)]
            loss = clip_batch_loss(u_batch, v_batch)
            if loss > max_loss:
                max_loss = loss
                max_loss_batch_idx = list(batch_idx)
    return [max_loss_batch_idx]

def get_embeddings(N, d):
    # cos(10 degree), sin(10 degree) = (0.9848134698792882, 0.17361575258114187)
    u = torch.tensor([[0.9848134698792882, 0.17361575258114187],
                      [0.9848134698792882, -0.17361575258114187],
                      [-0.9848134698792882, 0.17361575258114187],
                      [-0.9848134698792882, -0.17361575258114187]],requires_grad=True, device=device)
    v = torch.tensor([[0.9848134698792882, 0.17361575258114187],
                      [0.9848134698792882, -0.17361575258114187],
                      [-0.9848134698792882, 0.17361575258114187],
                      [-0.9848134698792882, -0.17361575258114187]],requires_grad=True, device=device)
                      
    # u = torch.randn((N, d), requires_grad=True, device=device)
    # v = torch.randn((N, d), requires_grad=True, device=device)
    with torch.no_grad():
        u.data = F.normalize(u.data, p=2.0, dim=1)
        v.data = F.normalize(v.data, p=2.0, dim=1)
    return (u, v)

def save_embeddings(u, v, d, filename=None):
    proj_u = F.normalize(u.to('cpu'), p=2.0, dim=1)
    proj_v = F.normalize(v.to('cpu'), p=2.0, dim=1)
    proj_u = proj_u.detach().numpy()
    proj_v = proj_v.detach().numpy()
    if filename is not None:
        np.savez(filename, u=proj_u, v=proj_v)

def plot_embeddings(u, v, d, filename=None):
    # project down to 2d to visualize
    # linear_projection = torch.randn(d, 2)
    # proj_u = F.normalize(u.to('cpu')@linear_projection.detach(), p=2.0, dim=1)
    # proj_v = F.normalize(v.to('cpu')@linear_projection.detach(), p=2.0, dim=1)
    proj_u = F.normalize(u.to('cpu'), p=2.0, dim=1)
    proj_v = F.normalize(v.to('cpu'), p=2.0, dim=1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(proj_u[:, 0].detach().numpy(), proj_u[:, 1].detach().numpy(), color='blue', label='u', marker="+", s=150)
    ax.scatter(proj_v[:, 0].detach().numpy(), proj_v[:, 1].detach().numpy(), color='red', label='v')
    ax.legend(loc='best')
    plt.grid()
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.show()
    if filename is not None:
        plt.savefig(f'{filename}.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    return proj_u, proj_v


# def heatmap(data, row_labels, col_labels, ax=None,
#             cbar_kw=None, cbarlabel="", plot_cbar=True, **kwargs):
#     """
#     Create a heatmap from a numpy array and two lists of labels.

#     Parameters
#     ----------
#     data
#         A 2D numpy array of shape (M, N).
#     row_labels
#         A list or array of length M with the labels for the rows.
#     col_labels
#         A list or array of length N with the labels for the columns.
#     ax
#         A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
#         not provided, use current axes or create a new one.  Optional.
#     cbar_kw
#         A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
#     cbarlabel
#         The label for the colorbar.  Optional.
#     **kwargs
#         All other arguments are forwarded to `imshow`.
#     """

#     if ax is None:
#         ax = plt.gca()

#     if cbar_kw is None:
#         cbar_kw = {}

#     # Plot the heatmap
#     im = ax.imshow(data, **kwargs)

#     # Create colorbar
#     cbar = None
#     if plot_cbar:
#         cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
#         cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

#     # Show all ticks and label them with the respective list entries.
#     ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
#     ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

#     # Let the horizontal axes labeling appear on top.
#     ax.tick_params(top=True, bottom=False,
#                    labeltop=True, labelbottom=False)

#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
#              rotation_mode="anchor")

#     # Turn spines off and create white grid.
#     ax.spines[:].set_visible(False)

#     ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
#     ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
#     ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
#     ax.tick_params(which="minor", bottom=False, left=False)

#     return im, cbar

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", plot_cbar=True, **kwargs):

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
        cbar.set_ticks([-0.9, 0.0, 0.9])  # Set specific colorbar tick labels with dummy ticks
        cbar.ax.set_yticklabels(['-1.0', '0.0', '1.0'])  # Set tick labels with specified decimal places
        cbar.ax.tick_params(labelsize=28)  # Increase font size
        cbar.ax.tick_params(length=0)  # Remove tick lines

    # Remove all ticks and label them with the respective list entries.
    ax.set_xticks([])
    ax.set_yticks([])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    #import pdb; pdb.set_trace()
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

def rearranging(z, solution):
    for i in range(z.shape[0]):
        diffs = torch.abs(solution[i].unsqueeze(-1) - z[i])
        diffs_np = diffs.numpy()
        selected = set()
        result = []
        for row in diffs_np:
            for idx in np.argsort(row):
                if idx not in selected:
                    selected.add(idx)
                    result.append(z[i][idx].item())
                    break
        z[i] = torch.tensor(result)
    return z
