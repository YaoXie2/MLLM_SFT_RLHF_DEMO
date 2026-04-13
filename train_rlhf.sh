export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1

CUDA_VISIBLE_DEVICES=1 \
MAX_PIXELS=1605632 \
NPROC_PER_NODE=1 \
swift rlhf \
    --rlhf_type grpo \
    --model /root/autodl-tmp/zz/outputs/SFT/v9-20260322-115029/checkpoint-149-merged \
    --external_plugins /root/autodl-tmp/zz/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --dataset /root/autodl-tmp/zz/datasets/AI-ModelScope/LaTeX_OCR/human_handwrite/validation-00000-of-00001_rlhf.jsonl \
    --max_completion_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --output_dir /root/autodl-tmp/zz/outputs/RLHF \
    --system /root/autodl-tmp/zz/system_prompt_rlhf.txt \
    --warmup_ratio 0.01 \
    --num_generations 4 \
    --generation_batch_size 4 \
    --temperature 1.0 \
    --log_completions true \
    --async_generate true \
    --beta 0.001