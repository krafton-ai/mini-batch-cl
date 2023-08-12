import os
from random import randint
import math
import argparse

import json
from tqdm import tqdm
from minimize_contrastive_loss_text_utils import *
from batch_utils import get_random_batch_idxs, create_inverse_greedy_batches_with_K, create_balance_greedy_batches, create_greedy_batches, sc_naive

import torch
import torch.nn as nn
import random
import numpy as np
import itertools
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

import open_clip
from open_clip import tokenizer

from datetime import datetime
try:
    import wandb
except ImportError:
    wandb = None

torch.backends.cudnn.benchmark = True
device = 'cuda:0'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--lr_full", type=float, default=0.5)
    parser.add_argument("--num_steps", type=int, default=1000)
    # parser.add_argument("--logging_step_ratio", type=float, default=0.1)
    parser.add_argument("--logging_step", type=float, default=100)
    parser.add_argument("--batch_size_list", nargs='+', type=int, default=[2])
    parser.add_argument("--batch_selections", nargs='+', type=str, default=[])
    parser.add_argument("--wandb_notes", default="", type=str, help="additional wandb logging note")
    parser.add_argument("--dataset", type=str, default="synthetic_emb")
    parser.add_argument("--wandb_use", type=bool, default=True)
    parser.add_argument("--grad_accum", default=False, action="store_true")
    parser.add_argument("--equal_init", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--saving_step", type=int, default=2000)
    args = parser.parse_args()

    print("N", args.N)
    print("d", args.d)
    print("lr_full", args.lr_full)
    print("num_steps", args.num_steps)
    # print("logging_step_ratio", args.logging_step_ratio)
    print("logging_step", args.logging_step)
    print("batch_size_list", args.batch_size_list)
    print("dataset", args.dataset)

    return args


def get_dataset(args):
    if args.dataset == "synthetic_emb":
        # u = torch.tensor([[0.9848134698792882, 0.17361575258114187],
        #                   [0.9848134698792882, -0.17361575258114187],
        #                   [-0.9848134698792882, 0.17361575258114187],
        #                   [-0.9848134698792882, -0.17361575258114187]], requires_grad=True, device=device)
        # v = torch.tensor([[0.9848134698792882, 0.17361575258114187],
        #                   [0.9848134698792882, -0.17361575258114187],
        #                   [-0.9848134698792882, 0.17361575258114187],
        #                   [-0.9848134698792882, -0.17361575258114187]],requires_grad=True, device=device)
        # dataset = (u,v)
        # args.N = 4
        # args.d = 2

        u = torch.randn((args.N, args.d), requires_grad=True, device=device)
        v = u.clone().detach().requires_grad_(True) \
            if args.equal_init else torch.randn((args.N, args.d), requires_grad=True, device=device)
        dataset = (u,v)

    elif dataset_name == "synthetic_text":
        text_pairs = {"She always keeps the room tidy.": "She never leaves the room messy.",
                      "She keeps the room dirty.": "She never leaves the room clean.", 
                      "She always drives fast on highways.": "She never takes highways slowly." ,
                      "She drives slowly on highways.": "She never takes highways fast.",}
        
        text_u = text_pairs.keys()
        text_v = text_pairs.values()

        x_u = tokenizer.tokenize(text_u).to(device)
        x_v = tokenizer.tokenize(text_v).to(device)

        dataset = (x_u, x_v)
        args.N = 4
        args.d = 512
        
    return dataset

def get_batch_idxs(args, batch_selection, embeddings):

    u,v = embeddings

    if batch_selection == 'f':
        return 
    elif batch_selection == 's':
        batch_idxs = get_random_batch_idxs(args.N, args.B)
    elif batch_selection == 'g':
        batch_idxs = create_greedy_batches(args.N, args.B, u.detach(), v.detach(), 1.0, device=device, D=args.d)
    elif batch_selection == 'bg':
        batch_idxs = create_balance_greedy_batches(args.N, args.B, u.detach(), v.detach(), 1.0, device=device, D=args.d)
    elif batch_selection == 'ig':
        batch_idxs = create_inverse_greedy_batches_with_K(args.N, args.B, u.detach(), v.detach(), 1.0, device=device, D=args.d)
    elif batch_selection == 'osgd':
        batch_idx = create_greedy_batches(args.N, args.B, u.detach(), v.detach(), 1.0, device=device, D=args.d, max_B=args.B, max_n_batch=1)[0]
        batch_idxs = [batch_idx]
    elif batch_selection == 'osgd_NcB':
        batch_idxs = osgd_NcB_batches(u.detach(), v.detach())
    elif batch_selection == 'sc_even':
        batch_idxs = sc_naive(args.B, u.detach(), v.detach(), batch_selection='sc_even', tqdm_desc=False)
    elif batch_selection in ['full', 'NcB']:
        batch_idxs = [list(range(u.shape[0]))]
    else:
        raise NotImplementedError(f'{batch_selection} is not available for batching')

    return batch_idxs

def get_embeddings(args, model, dataset):
    x_u, x_v = dataset
    if args.dataset == "synthetic_emb":
        u, v = x_u, x_v
    elif args.dataset == "synthetic_text":
        u = model.encode_text(x_u)
        v = model.encode_text(x_v)

    u = u / u.norm(dim=-1, keepdim=True)
    v = v / v.norm(dim=-1, keepdim=True)
    return (u, v)


def get_solution(N, d, B, output_dir):
    if d == 2*N:
        # ETF
        solution = torch.full((N, N), -1 / (N - 1))
        for i in range(N):
            solution[i, i] = 1.0
        plot_heatmap(solution, filename=os.path.join(output_dir, f"N{N}_d{2*N}_B{B}_ETF_w_cbar"), plot_cbar=True)
        plot_heatmap(solution, filename=os.path.join(output_dir, f"N{N}_d{2*N}_B{B}_ETF_wo_cbar"), plot_cbar=False)
    elif d == N//2:
        # Cross-polytope
        # solution = torch.zeros((N, N))
        # for i in range(N):
        #     solution[i, i] = 1.0
        #     random_index = random.choice([x for x in range(N) if x != i])
        #     solution[i, random_index] = -1.0
        # Cross-polytope
        crp_solution = torch.zeros((N, N))
        for i in range(N):
            crp_solution[i, i] = 1.0
            if i % 2 == 0:
                j = i+1
            else:
                j = i-1
            crp_solution[i, j] = -1.0
        solution = crp_solution

        plot_heatmap(solution, filename=os.path.join(output_dir, f"N{N}_d{N//2}_B{B}_CP_w_cbar"), plot_cbar=True)
        plot_heatmap(solution, filename=os.path.join(output_dir, f"N{N}_d{N//2}_B{B}_CP_wo_cbar"), plot_cbar=False)
    else:
        return None
    return solution


def train(args, dataset, batch_selections, output_dir):

    for B in batch_size_list:
        batch_idxs = get_random_batch_idxs(N, B)
        solution = get_solution(args.N, args.d, B, output_dir)
        # solution = torch.load('output/2023_05_17-06_27_40_N8_d3_lr0.5_accumFalse_s50000x1_u=vFalse_Bs[2]_seed44/z_full_mini_batch_B2_49999.pt')
        # plot_heatmap(solution, filename=os.path.join(output_dir, f"N{N}_d{N//2}_B{B}_CP_w_cbar"), plot_cbar=True)
        # plot_heatmap(solution, filename=os.path.join(output_dir, f"N{N}_d{N//2}_B{B}_CP_wo_cbar"), plot_cbar=False)

        for batch_selection in batch_selections:
            if args.wandb_use:
                print(f"init wandb logging for {batch_selection} B{B}...")
                exp_name = '-'.join([f"{batch_selection}_B{B}",])
                wandb.init(
                    entity="krafton_clap",
                    project="unimodal_synthetic",
                    group=f"{exp_tag}",
                    name=exp_name,
                    notes=args.wandb_notes,
                    config=vars(args)
                )

            if args.dataset == "synthetic_emb":
                dataset = get_dataset(args)
                x_u, x_v = dataset
                u, v = x_u, x_v
                model = None
                param_list = [u, v]
                optimizer = torch.optim.SGD(param_list, lr=args.lr_full)

            elif args.dataset == "synthetic_text":
                x_u, x_v = dataset
                model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
                model = model.to(device)
                optimizer = torch.optim.SGD(model.transformer.parameters(), lr=args.lr_full)

            loss_dict, true_loss_dict, solution_norm_dict = {}, {}, {}
            NUM_STEPS = args.num_steps
            NUM_BATCHES = (x_u.shape[0]//B)

            if (batch_selection in ['osgd', 'osgd_NcB']) and (not args.grad_accum):
                NUM_STEPS = NUM_BATCHES * NUM_STEPS
            
            for step in tqdm(range(NUM_STEPS*num_steps_factor), desc=f"[batch_selection:'{batch_selection}' | B:{B}] "):
                if batch_selection not in ['f']:
                    with torch.no_grad():
                        embeddings = get_embeddings(args, model, dataset)
                        batch_idxs = get_batch_idxs(args, batch_selection, embeddings)
                # print("batch_idxs:", batch_idxs)
                if args.grad_accum:
                    loss = 0     
                    for batch_idx in batch_idxs:
                        # print("step:", step)
                        optimizer.zero_grad()
                        x_u_batch, x_v_batch = x_u[list(batch_idx)], x_v[list(batch_idx)]
                        u_batch, v_batch = get_embeddings(args, model, (x_u_batch, x_v_batch))
                        if batch_selection == 'NcB':
                            loss = mini_batch_loss(u_batch, v_batch)
                        else:
                            loss += clip_batch_loss(u_batch, v_batch)
                    # loss = loss/len(batch_idxs)
                    loss.backward()
                    optimizer.step()
                else:
                    for batch_idx in batch_idxs:
                        # print("batch_idx:", batch_idx)
                        # print("step:", step)
                        optimizer.zero_grad()
                        x_u_batch, x_v_batch = x_u[list(batch_idx)], x_v[list(batch_idx)]
                        u_batch, v_batch = get_embeddings(args, model, (x_u_batch, x_v_batch))
                        if batch_selection == 'NcB':
                            loss = mini_batch_loss(u_batch, v_batch)
                        else:
                            loss = clip_batch_loss(u_batch, v_batch)
                        loss.backward()
                        optimizer.step()
                if (batch_selection in ['osgd', 'osgd_NcB']) and (not args.grad_accum):
                    if step % NUM_BATCHES == 0:
                        step = step // NUM_BATCHES
                        if (step % args.logging_step == 0 or step == NUM_STEPS-1):
                            with torch.no_grad(), torch.cuda.amp.autocast():
                                u,v = get_embeddings(args, model, dataset)
                            loss_dict[step], true_loss_dict[step] = loss.item(), clip_batch_loss(u, v).detach().item()
                            if solution != None:
                                solution_norm_dict[step] = torch.linalg.norm(rearranging((u @ v.T).detach().cpu(), solution)-solution, ord='fro').item()
                            # save_embeddings(u, v, args.d, filename=f'{output_dir}/embeddings_{batch_selection}_B{B}_step_{step}.npz')
                            # u_proj, v_proj = plot_embeddings(u, v, args.d, filename=f'{output_dir}/plot_embeddings_{batch_selection}_mini_batch_B{B}_step_{step}')
                            if args.wandb_use:
                                assert wandb is not None, 'Please install wandb.'
                                if solution != None:
                                    dic = {'step': step, 'batch_loss': loss_dict[step], 'true_loss': true_loss_dict[step], 'solution_norm': solution_norm_dict[step],}
                                else:
                                    dic = {'step': step, 'batch_loss': loss_dict[step], 'true_loss': true_loss_dict[step],}
                                wandb.log(dic)                        
                else: 
                    if step % args.logging_step == 0 or step == NUM_STEPS-1:
                        with torch.no_grad(), torch.cuda.amp.autocast():
                            u,v = get_embeddings(args, model, dataset)
                        loss_dict[step], true_loss_dict[step] = loss.item(), clip_batch_loss(u, v).detach().item()
                        if solution != None:
                            solution_norm_dict[step] = torch.linalg.norm(rearranging((u @ v.T).detach().cpu(), solution)-solution, ord='fro').item()
                        # save_embeddings(u, v, args.d, filename=f'{output_dir}/embeddings_{batch_selection}_B{B}_step_{step}.npz')
                        # u_proj, v_proj = plot_embeddings(u, v, args.d, filename=f'{output_dir}/plot_embeddings_{batch_selection}_mini_batch_B{B}_step_{step}')
                        if args.wandb_use:
                            assert wandb is not None, 'Please install wandb.'
                            if solution != None:
                                dic = {'step': step, 'batch_loss': loss_dict[step], 'true_loss': true_loss_dict[step], 'solution_norm': solution_norm_dict[step],}
                            else:
                                dic = {'step': step, 'batch_loss': loss_dict[step], 'true_loss': true_loss_dict[step],}
                            wandb.log(dic)

                if step % args.saving_step == 0 and step != 0: 
                    z = (u @ v.T).detach().cpu()
                    if batch_selection not in ['f']:
                        if solution != None:
                            z = rearranging(z, solution)
                    print("u^T v={}".format(torch.round(z, decimals=3)))
                    torch.save(z, f"{output_dir}/z_{batch_selection}_mini_batch_B{B}_{step}.pt")
                    
                    plot_heatmap(z, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z_fixed_mini_batch_B{B}_{batch_selection}_step{step}_w_cbar"), plot_cbar=True)
                    plot_heatmap(z, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z_fixed_mini_batch_B{B}_{batch_selection}_step{step}_wo_cbar"), plot_cbar=False)

                    if args.wandb_use:
                        # embedding = wandb.Image(os.path.join(output_dir, f"plot_embeddings_{batch_selection}_mini_batch_B{B}.png"))
                        z_w_cbar = wandb.Image(os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z_fixed_mini_batch_B{B}_{batch_selection}_step{step}_w_cbar" + ".png"))
                        z_wo_cbar = wandb.Image(os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z_fixed_mini_batch_B{B}_{batch_selection}_step{step}_wo_cbar" + ".png"))
                        wandb.log({f"heatmap_w_cbar_step{step}": [z_w_cbar]})
                        wandb.log({f"heatmap_wo_cbar_step{step}": [z_wo_cbar]})


            with open(f'{output_dir}/loss_{batch_selection}_mini_batch_B{B}.json', 'w') as f:
                f.write(json.dumps(loss_dict, indent=4))
            with open(f'{output_dir}/true_loss_{batch_selection}_mini_batch_B{B}.json', 'w') as f:
                f.write(json.dumps(true_loss_dict, indent=4))
            with open(f'{output_dir}/solution_norm_{batch_selection}_mini_batch_B{B}.json', 'w') as f:
                f.write(json.dumps(solution_norm_dict, indent=4))

            z = (u @ v.T).detach().cpu()
            if batch_selection not in ['f']:
                if solution != None:
                    z = rearranging(z, solution)
            print("u^T v={}".format(torch.round(z, decimals=3)))
            torch.save(z, f"{output_dir}/z_{batch_selection}_mini_batch_B{B}_{step}.pt")

            plot_heatmap(z, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z_fixed_mini_batch_B{B}_{batch_selection}_w_cbar"), plot_cbar=True)
            plot_heatmap(z, filename=os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z_fixed_mini_batch_B{B}_{batch_selection}_wo_cbar"), plot_cbar=False)

            if args.wandb_use:
                # embedding = wandb.Image(os.path.join(output_dir, f"plot_embeddings_{batch_selection}_mini_batch_B{B}.png"))
                z_w_cbar = wandb.Image(os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z_fixed_mini_batch_B{B}_{batch_selection}_w_cbar" + ".png"))
                z_wo_cbar = wandb.Image(os.path.join(output_dir, f"N{N}_d{d}_lr{lr_full}_s{NUM_STEPS*num_steps_factor}_z_fixed_mini_batch_B{B}_{batch_selection}_wo_cbar" + ".png"))
                wandb.log({"heatmap_w_cbar": [z_w_cbar]})
                wandb.log({"heatmap_wo_cbar": [z_wo_cbar]})
                wandb.finish()


if __name__ == '__main__':
    
    args = get_args()

#    seed = 44
    set_seed(args.seed)

    N = args.N
    d = args.d
    lr_full = args.lr_full
    batch_size_list = args.batch_size_list
    NUM_STEPS = args.num_steps
    num_steps_factor = 5 if N > 12 else 2

    num_steps_factor = 1
    logging_step = args.logging_step
    time_tag = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    exp_tag = f"{time_tag}_N{N}_d{d}_lr{lr_full}_accum{args.grad_accum}_s{NUM_STEPS}x{num_steps_factor}_u=v{args.equal_init}_Bs{batch_size_list}_seed{args.seed}"
    # output_dir = f"/home/jovyan/research_projects/open_clip/output/{exp_tag}"
    output_dir = f"/data/clap/projects/jglee/gitlab_open_clip/open_clip/src/utils/output/{exp_tag}"
    os.makedirs(output_dir, exist_ok=True)
    print("output_dir:", output_dir)

    ## First minimize full_batch loss
    dataset = get_dataset(args)

    batch_selections = args.batch_selections
    # batch_selections = ['f']
    # batch_selections = ['full']
    # batch_selections = ['NcB']
    # batch_selections = ['full', 'NcB']
    # batch_selections = ['full', 'NcB', 'f']
    # batch_selections = ['s']
    # batch_selections = ['s', 'osgd_NcB', 'sc_even']

    # batch_selections = ['full']
    # batch_selections = ['osgd_NcB']
    # batch_selections = ['full', 'NcB', 'f']
    # batch_selections = ['s', 'g', 'osgd', 'osgd_NcB', 'sc_even']
    # batch_selections = ['full', 'NcB', 's', 'g', 'bg', 'ig', 'osgd']
    # batch_selections = ['full', 'NcB', 'f', 's', 'g', 'bg', 'ig', 'osgd']
    assert len(batch_selections) > 0
    
    train(args,
          dataset=dataset,
          batch_selections=batch_selections,
          output_dir=output_dir,)
    