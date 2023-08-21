CUDA_VISIBLE_DEVICES=0 python evaluation.py \
    --ckpt_key mscoco_bz32_sim_all \
    --evaluation top1_accuracy \
    --epoch_list 10 20 30 40 50 \
    --eval_data_name mscoco \
    --data_pathname eval \
    --bimodal
