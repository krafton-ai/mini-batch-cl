PYTHONPATH=../../ CUDA_VISIBLE_DEVICES=1 python train.py     --lr=1e-7 --epochs=30 --batch-size=32 --feature-batch-size=2496     --warmup-epochs 3     --arch resnet18     --learning-rate-scaling=sqrt     --gamma 0.9     --multiprocessing-distributed --world-size 1 --rank 0 --workers 16     --crop-min=.08     --wd=0.1     --dist-url 'tcp://localhost:10001'     --data_name mscoco     --train_data '/data/clap/cooked_data/COCO/Sep14/train_one_caption.csv'     --val_data '/data/clap/cooked_data/COCO/Sep14/val_one_caption.csv'     --val_on_train_data '/data/clap/cooked_data/COCO/Sep14/train_one_caption_5000.csv'     --save_dir ./logs/     --objective_type sim     --batch_sampling s     --print-freq 1     --save-freq 3     --bimodal
  