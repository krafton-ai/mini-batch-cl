# SC (ours)
PYTHONPATH=../../ CUDA_VISIBLE_DEVICES=0 python train.py \
  --lr=3e-5 --epochs=200 --batch-size=100 --feature-batch-size=2400 \
  --arch resnet18 \
  --learning-rate-scaling=sqrt \
  --gamma 0.9 \
  --multiprocessing-distributed --world-size 1 --rank 0 --workers 16 \
  --crop-min=.08 \
  --wd=0.1 \
  --dist-url 'tcp://localhost:10000' \
  --data_name mscoco \
  --data '/data/clap/cooked_data/COCO/Sep14/train_one_caption.csv' \
  --save_dir ./logs/ \
  --objective_type sim \
  --batch_sampling sc_even_kb_loose --best_criteria max --k 10 --q 10 \
  --print-freq 1 \
  --save-freq 10 \
  --bimodal