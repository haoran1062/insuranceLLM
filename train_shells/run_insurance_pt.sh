CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 8765 pretraining.py \
    --model_type baichuan \
    --model_name_or_path /data/weights/baichuan13b_pretrain_weights \
    --train_file_dir /dev/shm/datasets/llm_train_datasets/lite_pretrain_datasets \
    --validation_file_dir /dev/shm/datasets/llm_train_datasets/lite_pretrain_datasets \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --fp16 \
    --max_train_samples 1000000 \
    --max_eval_samples 1000 \
    --num_train_epochs 1.5 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 32 \
    --block_size 1024 \
    --output_dir /data/weights/trained/pretrain/baichuan_insurance_pretrain_1.5e_lite_0.2_data \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True

# /data/datasets/llm_train_datasets/pretrain_datasets
# --tokenizer_name_or_path ./merged_tokenizer_hf \
# /data/datasets/insurance_textbook/all_txt
# /data/weights/Baichuan-13B-Chat
# /data/weights/baichuan13b_pretrain_weights