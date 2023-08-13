import os
import time
import copy
import torch
import random
import builtins
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from torch import distributed as dist
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from typing import Optional
import torch.nn.functional as F


class CustomBatchSampler(Sampler):
    def __init__(self, batch_index_list=None):
        super().__init__(data_source=None)
        self.indices = batch_index_list.squeeze(0).tolist()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DistributedBatchSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, batch_index_list: torch.Tensor, num_replicas: Optional[int] = None, 
                rank: Optional[int] = None, shuffle: bool = True,
                seed: int = 0, drop_last: bool = False, batch_size = 10) -> None:
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.batch_size = batch_size
        self.batch_index_list = batch_index_list.tolist()
        self.rank = rank

    def __iter__(self):
        return iter(self.batch_index_list[self.rank])

    def __len__(self) -> int:
        return self.num_samples


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def all_gather(features, world_size=1):
    output = [torch.zeros_like(features) for _ in range(world_size)]
    dist.all_gather(output, features)
    return torch.cat(tuple(output))


def get_learned_features(feature_loader, model, global_step, args, full_extraction=False):
    a_idx = a_z_i = a_z_j = None
    assert args.world_size > 0
    kb_size = len(feature_loader) * args.feature_batch_size
    # print(f"[get_learned_features] kb_size: {kb_size}, len(feature_loader): {len(feature_loader)}, args.feature_batch_size: {args.feature_batch_size}")
    if args.distributed:
        # should call the set_epoch() method at the beginning of each global_step (for OSGD family)
        feature_loader.sampler.set_epoch(global_step)
    with torch.no_grad():
        for step, ((x_i, x_j), _, idx) in enumerate(tqdm(feature_loader, desc=f'rank[{args.rank}] | feature extraction')):
            # print("x_i: ", x_i[:3, :5])
            # print("x_j: ", x_j[:3, :5])
            x_i = x_i.cuda(args.gpu, non_blocking=True)
            x_j = x_j.cuda(args.gpu, non_blocking=True)

            with torch.cuda.amp.autocast(True):
                z_i, z_j = model(x_i, x_j, None, None, args, wo_loss=True)
            if full_extraction:
                x_i, x_j, z_i, z_j = x_i.cpu(), x_j.cpu(), z_i.cpu(), z_j.cpu()

            if step == 0:
                a_idx = idx
                a_z_i = z_i
                a_z_j = z_j
            else:
                a_idx = torch.cat([a_idx, idx], dim=0)
                a_z_i = torch.cat([a_z_i, z_i], dim=0)
                a_z_j = torch.cat([a_z_j, z_j], dim=0)
            if not full_extraction and args.batch_sampling in ["osgd_kb", "osgd"] and args.k and args.k * args.batch_size < (step+1) * a_idx.shape[0]:
                kb_size = args.k * args.batch_size
                break
        assert a_idx is not None
        if args.distributed:
            a_idx = all_gather(a_idx.cuda(args.gpu, non_blocking=True), args.world_size).cpu()
            a_z_i = all_gather(a_z_i, args.world_size)
            a_z_j = all_gather(a_z_j, args.world_size)
            kb_size *= args.world_size
    return a_idx[:kb_size], a_z_i[:kb_size], a_z_j[:kb_size]


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
            kmeans = KMeans(n_clusters)
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


def sc_even_kb_loose(features, batch_selection, k, args, tqdm_desc=True):
    idxs, u, v = features
    d = u.shape[-1]
    B = args.global_batch_size
    N = (idxs.shape[0]//(k*B))*(k*B)
    # print(f"[sc_even_kb_loose] idxs.shape {idxs.shape[0]}  B: {B}, N: {N}, k*B: {k*B}")

    # Reshape stacked features to (-1, batch_size) features
    idxs, u_, v_ = idxs[:N].reshape((N//(k*B), k, B)), u[:N].reshape((N//(k*B), k, B, d)), v[:N].reshape((N//(k*B), k, B, d)) 
    #print("u_: ", u_[:3, :5])
    #print("v_: ", v_[:3, :5])
    # Get k batches and their loss
    batch_idxs = []
    # end = time.time()

    iterator = zip(idxs, u_, v_) if not tqdm_desc else tqdm(zip(idxs, u_, v_), total=N//(k*B), desc=f'rank[{args.rank}] | sc even kb loose')
    for _, (k_idxs, k_u, k_v) in enumerate(iterator):  # [N//(k*B)], [k, B, dim]
        batch_idxs += sc_naive((k_idxs.reshape(-1), k_u.reshape(-1, d), k_v.reshape(-1, d)), "sc_even", B, args, tqdm_desc=False)
        # end = time.time()
    # assert len(set(element for sublist in batch_idxs for element in sublist)) == N, f"N={N}, len={len(set(element for sublist in batch_idxs for element in sublist))}"
    return batch_idxs


def index_groups(a_idx, input_list):
    index_dict = defaultdict(list)

    for idx, value in enumerate(input_list):
        index_dict[value].append(idx)

    return [a_idx[index].tolist() for index in list(index_dict.values())]


def sc_naive(features, batch_selection, batch_size, args, tqdm_desc=True):
    # normalize embeddings
    a_idx = features[0]
    a_z_i = F.normalize(features[1], p=2, dim=1) / np.sqrt(args.t)
    a_z_j = F.normalize(features[2], p=2, dim=1) / np.sqrt(args.t)
    #print("a_z_i : ", a_z_i[:3, :5])
    #print("a_z_j : ", a_z_j[:3, :5])

    assert len(a_idx) % batch_size == 0
    n_clusters = len(a_idx) // batch_size

    sc = CustomSpectralClustering(n_clusters, batch_selection, batch_size, affinity='precomputed', eigen_solver='arpack')
    affinity = custom_affinity(a_z_i.cpu().detach().numpy(), a_z_j.cpu().detach().numpy(), batch_size)
    #print("psd_affinity not exp : ", affinity[:5, :5])
    psd_affinity = np.exp(affinity)
    y_pred = sc.fit_predict(psd_affinity)

    iterator = y_pred if not tqdm_desc else tqdm(y_pred, desc=f'rank[{args.rank}] | sc naive')
    return index_groups(a_idx, iterator)


def osgd_kb_loose(criterion, features, k, q, args):
    idxs, u, v = features
    d = u.shape[-1]
    B = args.global_batch_size
    N = (idxs.shape[0]//(k*B))*(k*B)

    # Reshape stacked features to (-1, batch_size) features
    idxs, u_, v_ = idxs[:N].reshape((N//(k*B), k, B)), u[:N].reshape((N//(k*B), k, B, d)), v[:N].reshape((N//(k*B), k, B, d)) 
    # Get k batches and their loss
    batch_idxs = []
    # end = time.time()
    for _, (k_idxs, k_u, k_v) in enumerate(tqdm(zip(idxs, u_, v_), desc=f'rank[{args.rank}] | osgd (kb-loose)', total=N//(k*B))):  # [N//(k*B)]
        batch_idxs_temp,losses = [], []
        for b_idx, (idx, u_temp, v_temp) in enumerate(zip(k_idxs, k_u, k_v)):  # [k, B, dim]
            batch_idxs_temp.append(idx)
            loss = criterion(torch.cat([u_temp, v_temp], dim=0), distributed=False)
            losses.append(loss.item())
        # Get top-q batches from losses
        if args.best_criteria == "min":
            topk_idxs = np.argsort(np.array(losses))[:q]
        else:
            topk_idxs = np.argsort(np.array(losses))[-q:]
        batch_idxs_temp = torch.stack(batch_idxs_temp, dim=0)
        batch_idxs_temp = batch_idxs_temp[topk_idxs].tolist()
        batch_idxs += batch_idxs_temp
        # end = time.time()
    return batch_idxs


def osgd_kb_loose_bimodal(criterion, features, k, q, args):
    idxs, u, v = features
    d = u.shape[-1]
    B = args.global_batch_size
    N = (idxs.shape[0]//(k*B))*(k*B)

    # Reshape stacked features to (-1, batch_size) features
    idxs, u_, v_ = idxs[:N].reshape((N//(k*B), k, B)), u[:N].reshape((N//(k*B), k, B, d)), v[:N].reshape((N//(k*B), k, B, d)) 
    # Get k batches and their loss
    batch_idxs = []
    # end = time.time()
    for _, (k_idxs, k_u, k_v) in enumerate(tqdm(zip(idxs, u_, v_), desc=f'rank[{args.rank}] | osgd (kb-loose)', total=N//(k*B))):  # [N//(k*B)]
        batch_idxs_temp,losses = [], []
        for b_idx, (idx, u_temp, v_temp) in enumerate(zip(k_idxs, k_u, k_v)):  # [k, B, dim]
            batch_idxs_temp.append(idx)
            loss = criterion(u_temp, v_temp, logit_scale=1, target_values=None)
            losses.append(loss.item())
        # Get top-q batches from losses
        if args.best_criteria == "min":
            topk_idxs = np.argsort(np.array(losses))[:q]
        else:
            topk_idxs = np.argsort(np.array(losses))[-q:]
        batch_idxs_temp = torch.stack(batch_idxs_temp, dim=0)
        batch_idxs_temp = batch_idxs_temp[topk_idxs].tolist()
        batch_idxs += batch_idxs_temp
        # end = time.time()
    return batch_idxs

def random_naive(total_dataset_size, batch_size, target_batch_num, args):
    assert total_dataset_size % batch_size == 0, "Check drop last option."
    batch_index_list = np.arange(total_dataset_size)
    np.random.shuffle(batch_index_list)
    batch_index_list = batch_index_list.reshape(total_dataset_size // batch_size, batch_size)
    return batch_index_list[:target_batch_num].tolist()


###################################
###         Main Utils         ###
###################################
def customize_train_loader(model, preemptive_loader, feature_loader, target_batch_num, epoch, step, args, features=None):
    batch_sampling = args.batch_sampling

    # get features
    features = get_learned_features(feature_loader, model, step, args) if features is None else features
    start = time.time()
    
    with torch.no_grad():
        if args.rank == 0:
            if batch_sampling in ["osgd_kb_loose"]:
                if args.bimodal:
                    batch_index_list = osgd_kb_loose_bimodal(unwrap_model(model).infonce_loss, features, args.k, args.q, args)
                else:
                    batch_index_list = osgd_kb_loose(unwrap_model(model).simclr_criteria, features, args.k, args.q, args)
            elif batch_sampling in ["sc_even_kb_loose"]:
                batch_index_list = sc_even_kb_loose(features, batch_sampling, args.k, args)
            else:
                raise NotImplementedError
            batch_index_list_tensor = torch.tensor(batch_index_list, dtype=torch.long)               # [target_batch_num, batch_size_per_gpu * world_size]
            batch_index_list_tensor = torch.split(batch_index_list_tensor, args.batch_size, dim=-1)  # (world_size, (target_batch_num, batch_size_per_gpu))
            batch_index_list_tensor = torch.stack(batch_index_list_tensor, dim=0).to(args.rank)      # [world_size, target_batch_num, batch_size_per_gpu]
        else:
            batch_index_list_tensor = torch.zeros((args.world_size, target_batch_num, args.batch_size), dtype=torch.long).to(args.rank)

    # create custom train loader using batch_index_list
    if args.distributed:
        # copy batch_index_list_tensor from rank:0 to all
        dist.broadcast(batch_index_list_tensor, src=0)
        sampler = DistributedBatchSampler(feature_loader.dataset, rank=args.rank, batch_index_list=batch_index_list_tensor, shuffle=False, batch_size=args.batch_size)
    else:
        sampler = CustomBatchSampler(batch_index_list=batch_index_list_tensor)
    
    train_loader = torch.utils.data.DataLoader(feature_loader.dataset, batch_sampler=sampler, num_workers=args.workers)

    batch_sample_time = time.time() - start
    return train_loader, batch_sample_time


def sample_loader(preemptive_loader, feature_loader, model, epoch, step, args, features=None):
    """
    return: train_loader
    """
    batch_sampling = args.batch_sampling

    if batch_sampling in ["s"]:
        # "s": handled at outer loop
        return preemptive_loader, None

    # calculate iters_per_sampling
    iters_per_sampling = len(feature_loader) * (args.feature_batch_size // args.batch_size)
    if batch_sampling in ["osgd_kb_loose", "sc_even_kb_loose"]:
        assert iters_per_sampling >= args.k, f"iters_per_sampling : {iters_per_sampling}, args.k : {args.k} / len(feature_loader) : {len(feature_loader)} / " + \
            f"args.feature_batch_size : {args.feature_batch_size} / args.batch_size : {args.batch_size}"
        iters_per_sampling = len(preemptive_loader.dataset) // (args.k * args.global_batch_size) * args.q

    train_loader, batch_sample_time = customize_train_loader(
        model, preemptive_loader, feature_loader, iters_per_sampling, epoch, step, args, features=features)

    return train_loader, batch_sample_time


###################################
###       Synthetic Utils       ###
###################################
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
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

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


def sc_naive_synthetic(batch_size, x_features: torch.Tensor, y_features: torch.Tensor, batch_selection="sc_even", tqdm_desc=True):
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
