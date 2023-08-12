CUDA_VISIBLE_DEVICES=0 python train.py \
  --lr=.075 --epochs=100 --batch-size=32 --feature-batch-size=2496 \
  --arch resnet18 \
  --learning-rate-scaling=sqrt \
  --gamma 0.9 \
  --multiprocessing-distributed --world-size 1 --rank 0 --workers 16 \
  --crop-min=.08 \
  --wd=1e-6 \
  --dist-url 'tcp://localhost:10026' \
  --data_name cifar100 \
  --data /data/clap/cooked_data/cifar-100-raw/images/ \
  --save_dir ./logs/ \
  --objective_type sim \
  --batch_sampling sc_even_kb_loose --best_criteria max --k 10 --q 10 \
  --resume logs/cifar100_resnet18_bz_32_accum1_E100_lr_0.075_sqrt_lars_obj_sim_sc_even_kb_loose_max_k10_q10/checkpoint_0060.pth.tar \
  --print-freq 1 \
  --save-freq 10 \
  --max-dataset-size 4992

  # --multiprocessing-distributed --world-size 1 --rank 0 --workers 16 \
