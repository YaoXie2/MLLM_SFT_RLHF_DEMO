from datasets import load_dataset
import json
import os
from PIL import Image

# ===== 原始 parquet 路径 =====
parquet_path = "/root/autodl-tmp/zz/datasets/AI-ModelScope/LaTeX_OCR/human_handwrite/train-00000-of-00001.parquet"
parquet_file_name = parquet_path.split('/')[-1].split('.')[0]

# ===== 自动解析父目录 =====
parent_dir = os.path.dirname(parquet_path)

# ===== 自动生成输出路径 =====
output_jsonl = os.path.join(parent_dir, f"{parquet_file_name}.jsonl")
image_save_dir = os.path.join(parent_dir, f"{parquet_file_name}_images")

# 创建图片目录
os.makedirs(image_save_dir, exist_ok=True)

# ===== 加载数据 =====
dataset = load_dataset(
    "parquet",
    data_files=parquet_path
)["train"]

# ===== 转换 =====
datas = []
for idx, sample in enumerate(dataset):
    img = sample.get("image", None)
    text = sample.get("text", "")

    # ---------- 处理图片 ----------
    if isinstance(img, Image.Image):
        image_path = os.path.join(image_save_dir, f"{idx}.png")
        img.convert("RGB").save(image_path)
    else:
        image_path = ""

    # ---------- 构造训练样本 ----------
    # SFT训练样本
    new_sample = {
        "messages": [
            {
                "role": "user",
                "content": "<image>\nConvert this image to LaTeX."
            },
            {
                "role": "assistant",
                "content": text
            }
        ],
        "images": [image_path]
    }
    datas.append(new_sample)
    print(f"处理第{idx}条数据")

with open(output_jsonl, "w", encoding="utf-8") as f:
    for item in datas:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Done: {output_jsonl}")
print(f"Images saved to: {image_save_dir}")
