# Mini-Batch Optimization of Contrastive Loss
<p align="left">
    <a href="https://github.com/krafton-ai/mini-batch-cl/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg">
    </a>
</p>

[Jaewoong Cho](https://sites.google.com/view/jaewoongcho), [Kartik Sreenivasan](https://ksreenivasan.github.io/), [Keon Lee](https://keonlee.notion.site/), Kyunghoo Mun, Soheun Yi, [Jeong-Gwan Lee](https://www.linkedin.com/in/jeonggwan-lee-511806102), Anna Lee, [Jy-yong Sohn](https://itml.yonsei.ac.kr/), [Dimitris Papailiopoulos](https://papail.io/), [Kangwook Lee](https://kangwooklee.com/).

Links: [Paper](https://arxiv.org/abs/2307.05906)

![Figure](figure1.png)

## Abstract

>Contrastive learning has gained significant attention as a method for self-supervised learning. The contrastive loss function ensures that embeddings of positive sample pairs (e.g., different samples from the same class or different views of the same object) are similar, while embeddings of negative pairs are dissimilar. Practical constraints such as large memory requirements make it challenging to consider all possible positive and negative pairs, leading to the use of mini-batch optimization.
In this paper, we investigate the theoretical aspects of mini-batch optimization in contrastive learning. We show that mini-batch optimization is equivalent to full-batch optimization if and only if all $\binom{N}{B}$ mini-batches are selected, while sub-optimality may arise when examining only a subset. We then demonstrate that utilizing high-loss mini-batches can speed up SGD convergence and propose a spectral clustering-based approach for identifying these high-loss mini-batches.
Our experimental results validate our theoretical findings and demonstrate that our proposed algorithm outperforms vanilla SGD in practically relevant settings, providing a better understanding of mini-batch optimization in contrastive learning.

## Conda Environment Setting

```bash
conda env create -f environment.yml
conda activate mini_batch_CL
# run the steps below only when the above procedures are not sufficient
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install wandb
pip install tensorboard
```

## Synthetic Dataset

```bash
CUDA_VISIBLE_DEVICES=0 python synthetic.py --N {N} --d {D} --num_steps {NUM_STEPS} --batch_selections {BATCH_SELECTION};
```
You can specify the total number of data points `--N`, the dimension of each data point `--d`, (configurable) multiple batch sizes `--batch_size_list`, the total number of training steps `--num_steps`, and (configurable) multiple batch selection methods `--batch_selections` to apply during training.


### Types of batch selection algorithms

* `f`: 'fixed' batch selection. It determines mini-batches fixed at epoch 0 and then trains with the same mini-batches.
* `s`: 'shuffle' batch selection. It determines shuffled mini-batches for each epoch.
* `full`: 'full-batch' batch selection. It uses full-batch to train for each epoch. 
* `NcB` : all 'N choose B' batch selection. It considers all combinations (NcB) of mini-batches and uses them during one epoch.
* `osgd_NcB`: 'Ordered SGD' batch selection. It configures one batch that returns maximum loss for one-step training, and repeats for N//B times (corresponding to one epoch)
* `sc_even`: 'Even-sized spectral clustering' batch selection. 

### Example Commands

1. Case: `d=2N` 

```bash
CUDA_VISIBLE_DEVICES=0 python synthetic.py --N 8 --batch_size_list 2 --d 16 --num_steps 2000 --batch_selections f s full NcB osgd_NcB sc_even 
```

2. Case: `d=N/2`

```bash
 CUDA_VISIBLE_DEVICES=0 python synthetic.py --N 8 --batch_size_list 2 --d 4 --num_steps 2000 --batch_selections f s full NcB osgd_NcB sc_even 
```

## Real Datasets

### Pre-training

Below is an example of self-supervised pre-training for a ResNet-18 model on CIFAR-100 and Tiny ImageNet using a single GPU server. By default, we use sqrt learning rate scaling, i.e., $\text{LearningRate}=0.075\times\sqrt{\text{BatchSize}}$, [LARS](https://arxiv.org/abs/1708.03888) optimizer, and a weight decay of 1e-6. For the temperature parameter $\tau$, we use a fixed value of $0.1$ from [SimCLR](https://arxiv.org/pdf/2002.05709.pdf). For GCL, gamma (γ in the paper) is an additional parameter for maintaining the moving average estimator, with a default value is $0.9$. 

We use a batch size of 32 and pretrain the ResNet-18 for 100 epochs. You can also increase the number of workers to accelerate the training speed.

```bash
# SGD
CUDA_VISIBLE_DEVICES=0 python train.py \
  --lr=.075 --epochs=100 --batch-size=32 --feature-batch-size=2496 \
  --arch resnet18 \
  --learning-rate-scaling=sqrt \
  --gamma 0.9 \
  --multiprocessing-distributed --world-size 1 --rank 0 --workers 16 \
  --crop-min=.08 \
  --wd=1e-6 \
  --dist-url 'tcp://localhost:10000' \
  --data_name {DATASET_NAME} \
  --data {DATASET_DIR} \
  --save_dir ./logs/ \
  --objective_type sim \
  --batch_sampling s \
  --print-freq 1 \
  --save-freq 10 \

# OSGD
CUDA_VISIBLE_DEVICES=1 python train.py \
  --lr=.075 --epochs=100 --batch-size=32 --feature-batch-size=2496 \
  --arch resnet18 \
  --learning-rate-scaling=sqrt \
  --gamma 0.9 \
  --multiprocessing-distributed --world-size 1 --rank 0 --workers 16 \
  --crop-min=.08 \
  --wd=1e-6 \
  --dist-url 'tcp://localhost:10001' \
  --data_name {DATASET_NAME} \
  --data {DATASET_DIR} \
  --save_dir ./logs/ \
  --objective_type sim \
  --batch_sampling osgd_kb_loose --best_criteria max --k 1500 --q 150 \
  --print-freq 1 \
  --save-freq 10 \

# SC (ours)
CUDA_VISIBLE_DEVICES=2 python train.py \
  --lr=.075 --epochs=100 --batch-size=32 --feature-batch-size=2496 \
  --arch resnet18 \
  --learning-rate-scaling=sqrt \
  --gamma 0.9 \
  --multiprocessing-distributed --world-size 1 --rank 0 --workers 16 \
  --crop-min=.08 \
  --wd=1e-6 \
  --dist-url 'tcp://localhost:10002' \
  --data_name {DATASET_NAME} \
  --data {DATASET_DIR} \
  --save_dir ./logs/ \
  --objective_type sim \
  --batch_sampling sc_even_kb_loose --best_criteria max --k 40 --q 40 \
  --print-freq 1 \
  --save-freq 10 \

```
Please note that the `--world-size` here refers to the number of nodes, not GPUs (so '1' should be the default value). Also, you can set `--data_name` to either `cifar100` or `tiny_imagenet` for CIFAR-100 and Tiny ImageNet, respectively, and `--data` to the corresponding data directory. Set `--objective_type` to `sog` if you want to change objective to SogCLR (`sim` stands for SimCLR objective).

### Retrieval Tasks
We provide evaluation code to run two retrieval tasks: on the corrupted dataset, and the original dataset.

**Corrupted dataset**
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py --ckpt_key cifar100_bz32_sim_all --evaluation corrupted_top1_accuracy --epoch_list 100 --eval_data_name cifar100_c --data_pathname "brightness/1";
```
Please note that `--eval_data_name` should be either `cifar100_c` or `tiny_imagenet_c` for the corruption retrieval task. The corruption name in `--data_pathname` does not need to be changed, as the runner will retrieve all corruptions based on the given directory path. For example, "brightness/1" and "contrast/1" will yield the same results. The only thing you may want to change is the severity number, for example, to five as in "brightness/5".

**Original dataset**
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py --ckpt_key cifar100_bz32_sim_all --evaluation top1_accuracy --epoch_list 100 --eval_data_name cifar100;
```
