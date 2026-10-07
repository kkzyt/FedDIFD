# run_bbox.py

from .captureKey import run_bbox_generator,run_bbox_generator_origin
from PIL import Image

import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

img = Image.open("./example-my/ILSVRC2012_val_00037383.JPEG")

class_names  = {
            "n01440764": "fish",
            "n02102040": "English springer",
            "n02979186": "cassette player",
            "n03000684": "chain saw",
            "n03028079": "church",
            "n03394916": "French horn",
            "n03417042": "garbage truck",
            "n03425413": "gas pump",
            "n03445777": "golf ball",
            "n03888257": "parachute"
        }

# 构造自然语言 prompt
prompts = [f"A {class_names[label]} is held in the hand of someone" for label in class_names]

clip_model, preprocess = clip.load("ViT-B/32", device=device)

text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

image_inputs = torch.cat([preprocess(img).unsqueeze(0)], dim=0).to(device)  # [B,3,224,224]

# 批量编码图像特征
image_features = clip_model.encode_image(image_inputs)  # [B, 512]
image_features /= image_features.norm(dim=-1, keepdim=True)

# 计算相似度 [B, num_prompts]
similarity = image_features @ text_features.T

# 对每张图像选出最匹配的prompt idx [B]
best_idxs = similarity.argmax(dim=1).tolist()
best_idx = best_idxs[0]
text = prompts[best_idx]

print(f"Best matching prompt: {text} (index {best_idx})")

text_prompt = text

run_bbox_generator_origin(
    model_id="stabilityai/stable-diffusion-2-1-base",
    source_image=img,
    source_prompt=text_prompt,
    iters=3,
    guidance_scale=3,
    word_idx=5,
)

