
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VLLM_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift rollout \
    --model /root/autodl-tmp/zz/outputs/SFT/v9-20260322-115029/checkpoint-149-merged \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 4096 \
    --max_new_tokens 2048 \
    --vllm_limit_mm_per_prompt '{"image": 2, "video": 1}' \
    --served_model_name Qwen3-VL-2B-Instruct-SFT 