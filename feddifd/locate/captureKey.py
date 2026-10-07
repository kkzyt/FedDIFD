# bbox_generator.py

from typing import Tuple, Union, Optional, List
import os
from scipy import ndimage
import numpy as np
from PIL import Image
import torch
import cv2
from torch import Tensor as T
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .utils.auto_bbox import MyAttnProcessor  # 保留你的自定义模块

TN = Optional[T]
TS = Union[Tuple[T, ...], List[T]]
device = torch.device("cuda:0")


def seed_everything(seed):
    import random
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_512(image_path: str, left=0, right=0, top=0, bottom=0):
    image = np.array(Image.open(image_path))[:, :, :3]
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top : h - bottom, left : w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset : offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset : offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def load_512_from_pil(image: Image.Image, left=0, right=0, top=0, bottom=0) -> np.ndarray:
    image_np = np.array(image)[:, :, :3]  # 确保 RGB 三通道
    h, w, c = image_np.shape

    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - top - 1)
    bottom = min(bottom, h - top - 1)

    image_np = image_np[top : h - bottom, left : w - right]
    h, w, c = image_np.shape

    # 中心裁剪为方形
    if h < w:
        offset = (w - h) // 2
        image_np = image_np[:, offset : offset + h]
    elif w < h:
        offset = (h - w) // 2
        image_np = image_np[offset : offset + w]

    # 调整到 512x512
    image_resized = Image.fromarray(image_np).resize((512, 512))
    return np.array(image_resized)

def load_512_from_pil_gpu(image, left=0, right=0, top=0, bottom=0):
    # 判断是否为 Tensor
    if isinstance(image, torch.Tensor):
        # 如果是Tensor，假设是 [C,H,W] 或 [B,C,H,W]
        if image.device.type != 'cuda':
            image = image.to('cuda')
        if image.dim() == 4:
            image = image.squeeze(0)
        # 保证是 uint8
        if image.dtype != torch.uint8:
            image = (image * 255).byte() if image.max() <= 1 else image.byte()
    else:
        # 假设是 PIL Image 或 numpy array
        if isinstance(image, Image.Image):
            np_img = np.array(image)[:, :, :3]
        else:
            np_img = np.array(image)[:, :, :3]
        image = torch.tensor(np_img, dtype=torch.uint8).permute(2, 0, 1).to('cuda')

    _, h, w = image.shape

    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - top - 1)
    bottom = min(bottom, h - top - 1)

    image = image[:, top : h - bottom, left : w - right]
    _, h, w = image.shape

    if h < w:
        offset = (w - h) // 2
        image = image[:, :, offset : offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[:, offset : offset + w, :]

    image = image.unsqueeze(0).float()
    image_resized = F.interpolate(image, size=(512, 512), mode='bilinear', align_corners=False)
    image_resized = image_resized.squeeze(0).byte()

    return image_resized

@torch.no_grad()
def get_text_embeddings(pipe: StableDiffusionPipeline, text: str) -> T:
    tokens = pipe.tokenizer(
        [text],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    ).input_ids.to(device)
    return pipe.text_encoder(tokens).last_hidden_state.detach()

@torch.no_grad()
def denormalize(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image[0]

@torch.no_grad()
def decode(latent: T, pipe: StableDiffusionPipeline, im_cat: TN = None):
    image = pipe.vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
    image = denormalize(image)
    if im_cat is not None:
        image = np.concatenate((im_cat, image), axis=1)
    return Image.fromarray(image)

def init_pipe(device, dtype, unet, scheduler) -> Tuple[UNet2DConditionModel, T, T]:

    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    for p in unet.parameters():
        p.requires_grad = False
    return unet, alphas, sigmas

def sample_and_sort(t_min, t_max, num_iters, z_taregt):
    samples = torch.linspace(
        t_min,
        t_max,
        num_iters,
        device=z_taregt.device,
        dtype=torch.long,
    )
    sorted_samples, _ = torch.sort(samples, descending=True)
    return [tensor.view(1) for tensor in sorted_samples]


def masks_to_boxes(mask: np.array) -> np.array:
    x, y = np.where(mask != 0)
    return np.min(x), np.min(y), np.max(x), np.max(y)


class BGMLoss:

    def noise_input(self, z, eps=None, timestep: Optional[int] = None):
        if timestep is None:
            b = z.shape[0]

            timestep = torch.randint(
                low=100,
                high=900,  # Avoid the highest timestep.
                size=(b,),
                device=z.device,
                dtype=torch.long,
            )
        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t

    def get_eps_prediction(
        self,
        z_t: T,
        timestep: T,
        text_embeddings: T,
        alpha_t: T,
        sigma_t: T,
        get_raw=False,
        guidance_scale=7.5,
    ):

        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(
            -1, *text_embeddings.shape[2:]
        )

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            e_t = self.unet(
                latent_input,
                timestep,
                embedd,
            ).sample
            if self.prediction_type == "v_prediction":
                e_t = (
                    torch.cat([alpha_t] * 2) * e_t
                    + torch.cat([sigma_t] * 2) * latent_input
                )
            e_t_uncond, e_t = e_t.chunk(2)
            if get_raw:
                return e_t_uncond, e_t

            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)  # cfg
            assert torch.isfinite(e_t).all()
        if get_raw:
            return e_t
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, pred_z0

    def clear_list(self):
        self.cross_attn_map_store.clear()
        self.self_attn_map_store.clear()

    def get_bgm_loss(
        self,
        z_source: T,
        text_emb_source: T,
        eps=None,
        timestep: Optional[int] = None,
        guidance_scale=7.5,
        height=None,
        width=None,
        iter=None,
        output=None,
    ) -> TS:
        with torch.inference_mode():
            z_t_source, eps, timestep, alpha_t, sigma_t = self.noise_input(
                z_source, eps, timestep
            )
            eps_pred_source, _ = self.get_eps_prediction(
                z_t_source,
                timestep,
                text_emb_source,
                alpha_t,
                sigma_t,
                guidance_scale=guidance_scale,
            )
        return self.cross_attn_map_store, self.self_attn_map_store

    def get_grad(self):
        return self.former_grad

    def get_attn_str(self, attn_name):
        index = attn_name.rfind(".")
        substring = attn_name[:index]
        attn1_str = substring.split(".")[-1]
        return attn1_str

    def __init__(self, device, pipe: StableDiffusionPipeline, dtype=torch.float32):
        self.t_min = 50
        self.t_max = 950
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.dtype = dtype
        self.unet, self.alphas, self.sigmas = init_pipe(
            device, dtype, pipe.unet, pipe.scheduler
        )
        self.prediction_type = pipe.scheduler.prediction_type
        self.former_grad = None
        self.cross_attn_map_store = []
        self.self_attn_map_store = []
        attn_processor_dict = {}
        for k in pipe.unet.attn_processors.keys():
            if self.get_attn_str(k) == "attn2":
                attn_processor_dict[k] = MyAttnProcessor(
                    self.cross_attn_map_store, self.self_attn_map_store
                )
            else:
                attn_processor_dict[k] = MyAttnProcessor(
                    self.cross_attn_map_store, self.self_attn_map_store
                )

        pipe.unet.set_attn_processor(attn_processor_dict)


def image_optimization(
    pipe: StableDiffusionPipeline,
    image: np.ndarray,
    text_source: str,
    num_iters=200,
    height=None,
    width=None,
    output=None,
    guidance_scale=7.5,
    word_idx=None,
) -> None:
    rds_loss = BGMLoss(device, pipe)
    image_source = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1
    image_source = image_source.unsqueeze(0).to(device)

    with torch.no_grad():
        z_source = pipe.vae.encode(image_source)["latent_dist"].mean * 0.18215
        embedding_null = get_text_embeddings(pipe, "")
        embedding_text = get_text_embeddings(pipe, text_source)
        embedding_source = torch.stack([embedding_null, embedding_text], dim=1)

    z_taregt = z_source.clone().requires_grad_(True)
    timestep_list = sample_and_sort(100, 500, num_iters, z_taregt)

    file_pairs = []
    for i in tqdm(range(num_iters)):
        cross_data, self_data = rds_loss.get_bgm_loss(
            z_source,
            embedding_source,
            timestep=timestep_list[i],
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            iter=i,
            output=output,
        )
    for _ in range(num_iters):
        cross_attn = [rds_loss.cross_attn_map_store.pop(0) for _ in range(16)]
        self_attn = [rds_loss.self_attn_map_store.pop(0) for _ in range(16)]
        file_pairs.append((cross_attn, self_attn))

    masks = []
    for cross_attn, self_attn in file_pairs:
        cross_256 = [t.reshape(-1, 16 * 16, 77) for t in cross_attn if t.shape[1] == 256]
        self_256 = [t.reshape(-1, 16 * 16, 16 * 16) for t in self_attn if t.shape[1] == 256]
        cross_256 = torch.cat(cross_256, dim=0)
        attention_maps = cross_256.sum(dim=0) / cross_256.shape[0]
        attention_maps = torch.pow(attention_maps, 2)
        self_256 = torch.cat(self_256, dim=0)
        self_maps = self_256.sum(dim=0) / self_256.shape[0]
        attention_maps = (self_maps @ attention_maps).reshape(16, 16, 77).cpu()
        image = attention_maps[:, :, word_idx]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = np.array(Image.fromarray(image.numpy().astype(np.uint8)).resize((512, 512)))
        mask = (image / 255.0).mean(axis=2)
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        labels, num_features = ndimage.label(mask)
        sizes = ndimage.sum(mask, labels, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        cleaned = np.copy(mask)
        cleaned[labels != largest_label] = 0
        masks.append(cleaned)

    final_mask = np.logical_or.reduce(masks)
    st_h, st_w, ed_h, ed_w = masks_to_boxes(final_mask)
    bbox_out = np.zeros([512, 512])
    bbox_out[st_h:ed_h, st_w:ed_w] = 1
    bbox_out = (bbox_out * 255).astype(np.uint8)
    
    return Image.fromarray((bbox_out * 255).astype(np.uint8))

def image_optimization_torch(
    pipe,
    image,
    text_source: str,
    num_iters=200,
    height=None,
    width=None,
    guidance_scale=7.5,
    word_idx: Optional[int] = None,
    device='cuda'
):
    rds_loss = BGMLoss(device, pipe)

    # 转为 Tensor 格式 [1, 3, H, W], 范围 [-1, 1]
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.unsqueeze(0)
        image_source = (image.float() / 127.5 - 1).to(device)
    else:
        raise TypeError(f"Unsupported image type {type(image)}")

    with torch.no_grad():
        z_source = pipe.vae.encode(image_source)["latent_dist"].mean * 0.18215
        embedding_null = get_text_embeddings(pipe, "")
        embedding_text = get_text_embeddings(pipe, text_source)
        embedding_source = torch.stack([embedding_null, embedding_text], dim=1)

    z_target = z_source.clone().requires_grad_(True)
    timestep_list = sample_and_sort(100, 500, num_iters, z_target)

    file_pairs = []
    for i in tqdm(range(num_iters)):
        cross_data, self_data = rds_loss.get_bgm_loss(
            z_source,
            embedding_source,
            timestep=timestep_list[i],
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            iter=i,
        )

    for _ in range(num_iters):
        cross_attn = [rds_loss.cross_attn_map_store.pop(0) for _ in range(16)]
        self_attn = [rds_loss.self_attn_map_store.pop(0) for _ in range(16)]
        file_pairs.append((cross_attn, self_attn))

    masks = []
    for cross_attn, self_attn in file_pairs:
        cross_256 = [t.reshape(-1, 16 * 16, 77) for t in cross_attn if t.shape[1] == 256]
        self_256 = [t.reshape(-1, 16 * 16, 16 * 16) for t in self_attn if t.shape[1] == 256]
        if len(cross_256) == 0 or len(self_256) == 0:
            continue

        cross_256 = torch.cat(cross_256, dim=0)  # [B, 256, 77]
        self_256 = torch.cat(self_256, dim=0)    # [B, 256, 256]

        attn_map = (cross_256.mean(0).pow(2))    # [256, 77]
        self_map = self_256.mean(0)              # [256, 256]
        attn_map = (self_map @ attn_map).reshape(16, 16, 77)  # [16, 16, 77]

        image_map = attn_map[..., word_idx]      # [16, 16]
        image_map = image_map - image_map.min()
        image_map = 255 * image_map / (image_map.max() + 1e-5)
        image_map = F.interpolate(image_map.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)
        mask = image_map.squeeze().round().clamp(0, 255) / 255.0  # [512, 512]

        binary_mask = (mask >= 0.5).to(torch.uint8)  # [512, 512]
        masks.append(binary_mask)

    # 叠加掩膜
    final_mask = torch.stack(masks, dim=0).any(dim=0).to(torch.uint8)

    # 提取 bounding box 区域
    nonzero = torch.nonzero(final_mask)
    if nonzero.numel() == 0:
        return torch.zeros((1, 512, 512), dtype=torch.uint8)

    y_min, x_min = nonzero.min(0)[0]
    y_max, x_max = nonzero.max(0)[0]

    bbox_out = torch.zeros((512, 512), dtype=torch.uint8)
    bbox_out[y_min:y_max + 1, x_min:x_max + 1] = 1

    return bbox_out  # 返回 torch.Tensor, uint8 格式，1 表示 bbox 区域

def extract_image(bbox_img, target_img):
    # 读取两张图片（确保两张图片分辨率相同）
    #bbox_img = cv2.imread('example-my/bbox.jpg')
    #target_img = cv2.imread('example-my/ILSVRC2012_val_00009379.JPEG')

    bbox_img = np.array(bbox_img)
    bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
    target_img =  cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)

    # 确认尺寸一致
    assert bbox_img.shape == target_img.shape, "两张图片尺寸不一致！"

    # 转灰度得到掩膜
    bbox_gray = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(bbox_gray, 240, 255, cv2.THRESH_BINARY)

    # 找到掩膜中白色区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("没有检测到白色区域！")

    # 找到最大轮廓对应的边界框（最小矩形）
    x, y, w, h = cv2.boundingRect(contours[0])

    # 从目标图片中裁剪对应区域
    cropped_img = target_img[y:y+h, x:x+w]

    # 同时裁剪掩膜区域以确认正确（可选）
    cropped_mask = mask[y:y+h, x:x+w]

    # 把掩膜用作透明度或者把非掩膜区域变为白色或其他颜色
    # 这里示例是用掩膜把非掩膜区域变为白色
    cropped_img_mask_bool = cropped_mask.astype(bool)
    final_img = np.full_like(cropped_img, 255)  # 创建白色背景
    final_img[cropped_img_mask_bool] = cropped_img[cropped_img_mask_bool]

    return final_img

def extract_bbox_region_torch(bbox_tensor, target_tensor, threshold=0.5):
    """
    bbox_tensor: torch.Tensor, [H, W], 0/1掩码
    target_tensor: torch.Tensor, [C, H, W]

    返回：
        裁剪后的target_tensor，shape [C, h, w]
    """
    if(((bbox_tensor.dim() == 2 and target_tensor.dim() == 3) or target_tensor.shape[1:] == bbox_tensor.shape)==False):
        return target_tensor

    mask = bbox_tensor.bool()
    if mask.sum() == 0:
        raise ValueError("bbox_tensor中没有检测到1！")

    ys, xs = torch.where(mask)  # 找所有1的位置
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    cropped = target_tensor[:, y_min:y_max+1, x_min:x_max+1]

    return cropped

def save_tensor_as_image(tensor, filename):
    """
    保存三维tensor为图像文件
    tensor: torch.Tensor，shape [C, H, W]
            dtype: uint8 (0~255) 或 float (0~1)
    filename: 保存路径，如 "out.png"
    """
    # 处理数据类型和范围
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
        # 假设浮点数在0~1，转换到0~255 uint8
        tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)
    elif tensor.dtype != torch.uint8:
        raise TypeError("tensor dtype must be uint8 or float")

    # 转成 numpy，PIL 需要 HWC 格式
    np_img = tensor.permute(1, 2, 0).cpu().numpy()

    # 如果通道数是3，保存为RGB
    if np_img.shape[2] == 3:
        img = Image.fromarray(np_img, mode="RGB")
    elif np_img.shape[2] == 1:
        img = Image.fromarray(np_img.squeeze(2), mode="L")
    else:
        raise ValueError(f"不支持的通道数: {np_img.shape[2]}")

    img.save(filename)

def run_bbox_generator_origin(
    model_id: str,
    source_image,
    source_prompt: str,
    iters: int = 3,
    guidance_scale: float = 3,
    word_idx: int = 5,
):
    seed_everything(42)
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    image = load_512_from_pil(source_image)
    
    bbox_img = image_optimization(
        pipe,
        image,
        text_source=source_prompt,
        num_iters=iters,
        height=512,
        width=512,
        guidance_scale=guidance_scale,
        word_idx=word_idx,
    )

    print(f"type: {type(image)}, shape: {image.shape}")

    from datetime import datetime
    save_dir = "z_test_image/test2"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"output_{timestamp}_bbox.JPEG")
    bbox_img.save(save_path)
    print(f"Image saved to: {save_path}")

    final_img = extract_image(bbox_img, image)

    return final_img

def run_bbox_generator(
    model_id: str,
    source_image,
    source_prompt: str,
    iters: int = 3,
    guidance_scale: float = 3,
    word_idx: int = 5
):
    seed_everything(42)
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    if torch.cuda.is_available():
        # 用GPU版本
        image = load_512_from_pil_gpu(source_image)
    else:
        # 用你原先的CPU版本
        image = load_512_from_pil(source_image)
    
    bbox_img = image_optimization_torch(
        pipe,
        image,
        text_source=source_prompt,
        num_iters=iters,
        height=512,
        width=512,
        guidance_scale=guidance_scale,
        word_idx=word_idx,
    )

    final_img = extract_bbox_region_torch(bbox_img, image)

    # 保存代码
    # if final_img is not None:
    #     # 转PIL保存
    #     from datetime import datetime
    #     save_dir = "z_test_image/test2"
    #     os.makedirs(save_dir, exist_ok=True)
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     save_path = os.path.join(save_dir, f"output_{timestamp}_.JPEG")
    #     save_tensor_as_image(final_img, save_path)
    #     print(f"Image saved to: {save_path}")

    return final_img

    