uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-32B-Instruct"
lr=1e-5
min_lr=0
epochs=5
micro_batch_size=2 # -> batch_size will be 16 if 8 gpus
push_to_hub=true
gradient_accumulation_steps=2
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
train/sft.py \
--per_device_train_batch_size=${micro_batch_size} \
--per_device_eval_batch_size=${micro_batch_size} \
--gradient_accumulation_steps=${gradient_accumulation_steps} \
--num_train_epochs=${epochs} \
--max_steps=${max_steps} \
--train_file_path="simplescaling/s1K_tokenized" \
--model_name=${base_model} \
--warmup_ratio=0.05 \
--fsdp="full_shard auto_wrap" \
--fsdp_config="train/fsdp_config_qwen.json" \
--bf16=True \
--eval_strategy="steps" \
--eval_steps=50 \
--logging_steps=1 \
--save_steps=100 \
--lr_scheduler_type="cosine" \
--learning_rate=${lr} \
--weight_decay=1e-4 \
--adam_beta1=0.9 \
--adam_beta2=0.95 \
--output_dir="ckpts/s1_${uid}" \
--hub_model_id="simplescaling/s1-${uid}" \
--push_to_hub=${push_to_hub} \
--save_only_model=True