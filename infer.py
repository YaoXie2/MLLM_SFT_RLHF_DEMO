import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.infer_engine import TransformersEngine, RequestConfig, InferRequest
from swift import get_model_processor, get_template
from swift.utils import safe_snapshot_download
from peft import PeftModel
# 请调整下面几行
model = '/root/autodl-tmp/zz/model_weights/Qwen/Qwen3-VL-2B-Instruct'
lora_checkpoint = safe_snapshot_download('/root/autodl-tmp/zz/outputs/RLHF/v1-20260413-002524/checkpoint-34')  # 修改成checkpoint_dir
# lora_checkpoint = None  # 如果不使用lora微调的权重，可以设置为None
template_type = None  # None: 使用对应模型默认的template_type
default_system = None  # None: 使用对应模型默认的default_system

# 加载模型和对话模板
model, tokenizer = get_model_processor(model)
if lora_checkpoint is not None:
    model = PeftModel.from_pretrained(model, lora_checkpoint)
template_type = template_type or model.model_meta.template
template = get_template(tokenizer, template_type=template_type, default_system=default_system)
engine = TransformersEngine(model, template=template, max_batch_size=2)
request_config = RequestConfig(max_tokens=512, temperature=0)

# 这里使用了2个infer_request来展示batch推理
infer_requests = [
    InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}]),
    InferRequest(messages=[{'role': 'user', 'content': '<image>\nConvert this image to LaTeX.'}],
                 images=['/root/autodl-tmp/zz/datasets/test/latex_ocr_1.jpg']),
]
resp_list = engine.infer(infer_requests, request_config)
query0 = infer_requests[0].messages[0]['content']
print(f'response0: {resp_list[0].choices[0].message.content}')
print(f'response1: {resp_list[1].choices[0].message.content}')