import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPImagenetteClassifier:
    def __init__(self, prompts_json_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(prompts_json_path, "r") as f:
            self.text_prompts = json.load(f)

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self._predict_image(image)

    def predict_topk(self, image_path, k=5):
        image = Image.open(image_path).convert("RGB")
        return self._predict_topk_image(image, k)

    def predict_pil(self, pil_image):
        """
        直接传入PIL.Image对象，返回最匹配提示词和相似度
        """
        image = pil_image.convert("RGB")
        return self._predict_image(image)

    def predict_topk_pil(self, pil_image, k=5):
        """
        直接传入PIL.Image对象，返回top-k提示词和相似度列表
        """
        image = pil_image.convert("RGB")
        return self._predict_topk_image(image, k)

    def _predict_image(self, image):
        inputs = self.processor(text=self.text_prompts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        similarity = (image_embeds @ text_embeds.T).squeeze(0)
        best_idx = similarity.argmax().item()
        return self.text_prompts[best_idx], similarity[best_idx].item()

    def _predict_topk_image(self, image, k):
        inputs = self.processor(text=self.text_prompts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        similarity = (image_embeds @ text_embeds.T).squeeze(0)
        topk = torch.topk(similarity, k)
        return [(self.text_prompts[idx], score.item()) for idx, score in zip(topk.indices, topk.values)]
