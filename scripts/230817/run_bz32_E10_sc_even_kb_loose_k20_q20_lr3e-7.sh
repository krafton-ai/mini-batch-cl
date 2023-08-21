PYTHONPATH=../../ CUDA_VISIBLE_DEVICES=0 python train.py     --lr=3e-7 --epochs=10 --batch-size=32 --feature-batch-size=2496     --warmup-epochs 1     --arch resnet18     --learning-rate-scaling=sqrt     --gamma 0.9     --multiprocessing-distributed --world-size 1 --rank 0 --workers 16     --crop-min=.08     --wd=0.1     --dist-url 'tcp://localhost:10000'     --data_name mscoco     --train_data '/data/clap/cooked_data/COCO/Sep14/train_one_caption.csv'     --val_data '/data/clap/cooked_data/COCO/Sep14/val_one_caption.csv'     --val_on_train_data '/data/clap/cooked_data/COCO/Sep14/train_one_caption_5000.csv'     --save_dir ./logs/     --objective_type sim     --batch_sampling sc_even_kb_loose --best_criteria max --k 20 --q 20     --print-freq 200     --save-freq 1     --bimodal
  