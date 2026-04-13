# MLLM_SFT_RLHF_DEMO
详细结束内容介绍可参考我这篇博客：[MS-Swift框架下Qwen3-VL的SFT和RLHF微调实战](https://blog.csdn.net/messyking/article/details/159967379?sharetype=blogdetail&sharerId=159967379&sharerefer=PC&sharesource=messyking&spm=1011.2480.3001.8118)


仓库介绍：
- dataset：数据集目录
- model_weights: 模型权重目录
- output: 训练模型输出目录
- client.py: 客户端调用vllm部署的服务脚本
- deploy.sh: vllm部署脚本
- download_dataset.py: 数据集下载脚本
- download_model.py: 模型下载脚本
- infer.py: 模型推理-python版本
- infer.sh: 模型推理-shell版本(在线)
- merge_lora.sh: adapter权重和base model权重融合脚本
- plugin.py: 自定义奖励函数脚本
- rlhf_rollout.sh: rlhf训练的roll_out启动脚本
- system_prompt_rlhf.txt: system prompt脚本(rlhf)
- train_rlhf.sh: rlhf训练脚本 (GRPO)
- train_sft.sh: sft训练脚本
- trans_parquet2jsonl_RLHF.py: RLHF的parquet数据集转化成jsonl数据集脚本
- trans_parquet2jsonl_SFT.py: SFT的parquet数据集转化成jsonl数据集脚本
