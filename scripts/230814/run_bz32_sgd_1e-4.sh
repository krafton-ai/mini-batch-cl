# SGD
PYTHONPATH=../../ CUDA_VISIBLE_DEVICES=3 python train.py \
  --lr=1e-4 --epochs=100 --batch-size=32 --feature-batch-size=2400 \
  --arch resnet18 \
  --learning-rate-scaling=sqrt \
  --gamma 0.9 \
  --multiprocessing-distributed --world-size 1 --rank 0 --workers 16 \
  --crop-min=.08 \
  --wd=0.1 \
  --dist-url 'tcp://localhost:10003' \
  --data_name mscoco \
  --data '/data/clap/cooked_data/COCO/Sep14/train_one_caption.csv' \
  --save_dir ./logs/ \
  --objective_type sim \
  --batch_sampling s \
  --print-freq 1 \
  --save-freq 10 \
  --bimodal