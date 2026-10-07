import collections
import math
import random
from copy import deepcopy

from tqdm import tqdm
import os
import wandb
import torchvision.transforms as transforms
import torch.nn as nn

from utils.models import MultiRandomCrop, RandomMixup, RandomCutmix
from torch.utils.data import DataLoader, default_collate, TensorDataset
from fedsd2c.util import *
from utils.logger import Logger
from collections import defaultdict
from utils import AverageMeter
from utils.fed_utils import assign_dataset, init_model
from utils.models_gan import LargeGenerator
from utils.models import ConvNet
from torchvision.models import ResNet
from diffusers import AutoencoderKL


class FedSD2CClient(object):

    def __init__(self, args, client_id, dataset_id='MNIST'):
        """
        Client in the federated learning for FedD3
        :param client_id: Id of the client
        :param dataset_id: Dataset name for the application scenario
        """
        # Metadata
        self._id = client_id
        self._dataset_id = dataset_id
        self.args = args

        # Following private parameters are defined by dataset.
        self._image_length = -1
        self._image_width = -1
        self._image_channel = -1
        self._n_class, self._image_length, self._image_channel = assign_dataset(dataset_id)
        self._image_width = self._image_length

        # Initialize the parameters in the local client
        self._epoch = args.client_instance_n_epoch
        self._batch_size = args.client_instance_bs
        self._lr = args.client_instance_lr
        self._momentum = 0.9
        self.num_workers = 2
        self.loss_rec = []
        self.n_data = 0
        self.mixup_alpha = args.client_instance_mixup_alpha
        self.cutmix_alpha = args.client_instance_cutmix_alpha

        # Local dataset
        self._train_data = None
        self._test_data = None
        self._sd_data = None

        # Local distilled dataset
        self._distill_data = {'x': [], 'y': []}
        self._rest_data = {'x': [], 'y': [], 'dist': [], 'pred': []}
        self.coreset_select_indxs = []

        # FastDD parameters
        self.input_size = self._image_width
        self.num_crop = self.args.fedsd2c_num_crop
        self.factor = 1
        self.mipc = self.args.fedsd2c_mipc
        self.ipc = self.args.fedsd2c_ipc
        self.iter_mode = self.args.fedsd2c_iter_mode

        self.iterations_per_layer = self.args.fedsd2c_iteration
        self.jitter = self.args.fedsd2c_jitter
        self.sre2l_lr = self.args.fedsd2c_lr
        self.l2_scale = self.args.fedsd2c_l2_scale
        self.tv_l2 = self.args.fedsd2c_tv_l2
        self.r_bn = self.args.fedsd2c_r_bn
        self.r_c = self.args.fedsd2c_r_c
        self.r_adv = 0
        self.first_bn_multiplier = 10.
        self.inputs_init = self.args.fedsd2c_inputs_init

        self.noise_type = self.args.fedsd2c_noise_type
        self.noise_s = self.args.fedsd2c_noise_s
        self.noise_p = self.args.fedsd2c_noise_p

        self.normalizer = transforms.Normalize(means[self._dataset_id], stds[self._dataset_id])

        self._cls_record = None

        # Training on GPU
        gpu = args.gpu_id
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    def load_train(self, data):
        """
        Client loads the decentralized dataset, it can be Non-IID across clients.
        :param data: Local dataset for training.
        """
        self._train_data = {}
        # self._train_data = deepcopy(data)
        self._train_data = data
        self.n_data = len(data)

    def load_test(self, data):
        """
        Client loads the test dataset.
        :param data: Dataset for testing.
        """
        self._test_data = {}
        self._test_data = deepcopy(data)

    def load_cls_record(self, cls_record):
        """
        Client loads the statistic of local label.
        :param cls_record: class number record
        """
        self._cls_record = {}
        self._cls_record = {int(k): v for k, v in cls_record.items()}

    def train(self, model: nn.Module):
        """
        Client trains the model on local dataset
        :param model: model waited to be trained
        :return: Local updated model
        """
        model.train()
        model.to(self._device)
        mixup_transforms = []
        collate_fn = None
        if self.mixup_alpha > 0.0:
            mixup_transforms.append(RandomMixup(self._n_class, p=1.0, alpha=self.mixup_alpha))
        if self.cutmix_alpha > 0.0:
            mixup_transforms.append(RandomCutmix(self._n_class, p=1.0, alpha=self.cutmix_alpha))
        if mixup_transforms:
            mixupcutmix = transforms.RandomChoice(mixup_transforms)

            def collate_fn(batch):
                return mixupcutmix(*default_collate(batch))
        train_loader = DataLoader(self._train_data, batch_size=self._batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_fn)

        optimizer = torch.optim.SGD(model.parameters(), lr=self._lr, momentum=self._momentum, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        lr_scheduler = lr_cosine_policy(self._lr, 0, self._epoch)
        loss_func = nn.CrossEntropyLoss()

        # Training process
        loss_accumulator = AverageMeter()
        pbar = tqdm(range(self._epoch))
        local_step = 0
        for epoch in pbar:
            epoch_loss = AverageMeter()
            lr_scheduler(optimizer, epoch, epoch)
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    output = model(b_x)
                    loss = loss_func(output, b_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_accumulator.update(loss.data.cpu().item())
                epoch_loss.update(loss.data.cpu().item())
                if self.args.using_wandb:
                    wandb.log({
                        f"{self._id}C local_loss": loss.item(),
                        "iteration": local_step,
                    })
                    local_step += 1
            pbar.set_description('Epoch: %d' % epoch +
                                 '| Train loss: %.4f ' % epoch_loss.avg +
                                 '| lr: %.4f ' % optimizer.state_dict()['param_groups'][0]['lr'])

        return model, loss_accumulator.avg

    def test(self, model):
        """
        Server tests the model on test dataset.
        """
        test_loader = DataLoader(self._test_data, batch_size=self._batch_size, shuffle=False)
        model.to(self._device)
        accuracy_collector = 0
        for step, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                b_x = x.to(self._device)  # Tensor on GPU
                b_y = y.to(self._device)  # Tensor on GPU

                test_output = model(b_x)
                pred_y = torch.max(test_output, 1)[1].to(self._device).data.squeeze()
                accuracy_collector = accuracy_collector + sum(pred_y == b_y)
        accuracy = accuracy_collector / len(self._test_data)

        return accuracy.cpu().numpy()

    def get_ipc(self, label):
        return self.ipc

    def coreset_stage(self, model):
        model = deepcopy(model)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        _dataset = deepcopy(self._train_data)
        _dataset.dataset = deepcopy(_dataset.dataset)
        _dataset.dataset.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([self._image_length, self._image_width]),
            transforms.Normalize(mean=means[self._dataset_id], std=stds[self._dataset_id]),
        ])
        _dataset = CategoryDataset(_dataset, mipc=self.mipc, ipc=self.ipc * self.factor, shuffle=True, seed=self.args.sys_i_seed)

        ret_x = []
        ret_y = []

        mrc = MultiRandomCrop(self.num_crop, self.input_size, 1, 1)
        model.to(self._device)

        for c, (images, labels) in enumerate(_dataset):
            with torch.no_grad():
                images = mrc(images)
                ipc = self.get_ipc(labels[0].item())
                images, dists, rest_images, rest_dists, rest_preds, selected_indices = selector_coreset(
                    ipc * self.factor,
                    model,
                    images,
                    labels,
                    self.input_size,
                    device=self._device,
                    m=self.num_crop,
                    descending=False,
                    ret_all=True
                )
                self._rest_data['x'].extend([data.squeeze() for data in torch.split(rest_images.cpu(), 1)])
                self._rest_data['y'].extend([labels[0].cpu().item() for _ in range(rest_images.shape[0])])
                self._rest_data['dist'].extend([data.squeeze() for data in torch.split(rest_dists.cpu(), 1)])
                self._rest_data['pred'].extend([data.squeeze() for data in torch.split(rest_preds.cpu(), 1)])
                selected_indice_in_dset = []
                for indice in selected_indices.cpu().numpy().tolist():
                    selected_indice_in_dset.append(_dataset.class_indices[c][indice])
                self.coreset_select_indxs.extend(selected_indice_in_dset)
                images = mix_images(images, self.input_size, 1, images.shape[0]).cpu()

            # (ipc, 3, H, W)
            ret_x.extend([data.squeeze() for data in torch.split(images.cpu(), 1)])
            ret_y.extend([labels[0].cpu().clone() for _ in range(images.shape[0])])
        # ret_y = [0] * len(ret_x)
        self._distill_data['x'] = ret_x
        self._distill_data['y'] = ret_y

        return ret_x, ret_y

    def random_stage(self, model):
        _dataset = deepcopy(self._train_data)
        _dataset.dataset = deepcopy(_dataset.dataset)
        _dataset.dataset.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([self._image_length, self._image_width]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _dataset = CategoryDataset(_dataset, mipc=self.mipc, ipc=self.ipc, shuffle=True,
                                   seed=self.args.sys_i_seed)

        ret_x = []
        ret_y = []
        ret_score = {}

        for c, (images, labels) in enumerate(_dataset):
            with torch.no_grad():
                ipc = self.get_ipc(labels[0].item())
                indices = torch.randperm(len(images))[:ipc]
                images = images[indices]
                images = images.cpu()

            ret_x.extend([data.squeeze() for data in torch.split(images.cpu(), 1)])
            ret_y.extend([labels[0].cpu().clone() for _ in range(self.ipc)])
            # ret_y.extend([0] * int(images.shape[1]))
            ret_score[labels[0].item()] = 0
            self._rest_data['x'].extend([data.squeeze() for data in torch.split(images.cpu(), 1)])
            self._rest_data['y'].extend([labels[0].cpu().item() for _ in range(images.shape[0])])
            self._rest_data['dist'].extend([labels[0].cpu() for _ in range(images.shape[0])])
            self._rest_data['pred'].extend([labels[0].cpu() for _ in range(images.shape[0])])
        # ret_y = [0] * len(ret_x)
        self._distill_data['x'] = ret_x
        self._distill_data['y'] = ret_y

        return ret_x, ret_y, ret_score

    def synthesis_stage(self, model):
        logger = Logger()
        logger = logger.get_logger()

        ret_x = []
        ori_x = []
        ret_y = []
        ret_z = []
        loss_list = []
        loss_dict_list = {}

        # model init
        model = deepcopy(model)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        for p in vae.parameters():
            p.requires_grad = False

        # hook for loss computation init
        loss_r_feature_layers = []
        if isinstance(model, ResNet):
            loss_r_feature_layers.append(OutputHook(model.maxpool))
            for name, module in model.named_modules():
                if name in [f"layer{j}" for j in range(1, 5)]:
                    # print(f"Adding hook to {name}")
                    loss_r_feature_layers.append(OutputHook(module))
            loss_r_feature_layers.append(OutputHook(model.avgpool))
        elif isinstance(model, ConvNet):
            for j in range(4):
                # print(f"Adding hook to {j} pool")
                loss_r_feature_layers.append(OutputHook(model.layers["pool"][j]))
        else:
            raise NotImplementedError()
        loss_r_bn_layers = []
        if self.r_bn > 0:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    loss_r_bn_layers.append(BNFeatureHook(module))

        synset, batch_size = self._build_synset()
        synloader = torch.utils.data.DataLoader(synset, batch_size=batch_size, shuffle=False)
        for i, batch in enumerate(synloader):
            original_img, perturbed_img, y = batch

            # distillate initialization with fourier transformation
            with torch.no_grad():
                vae.to(self._device)
                if "fourier" in self.inputs_init:
                    z = vae.encode(denormalize(perturbed_img).to(self._device)).latent_dist.mode().clone().detach()
                else:
                    z = vae.encode(denormalize(original_img).to(self._device)).latent_dist.mode().clone().detach()
            targets = y.to(self._device)
            entropy_criterion = nn.CrossEntropyLoss()
            z.requires_grad = True
            optimizer = torch.optim.AdamW([z], lr=self.sre2l_lr, betas=(0.5, 0.9), eps=1e-8)
            lr_scheduler = lr_cosine_policy(self.sre2l_lr, 0, self.iterations_per_layer)

            best_inputs = None
            best_z = None
            best_cost = 1e4
            losses = []
            loss_dicts = {}
            for iteration in range(self.iterations_per_layer):
                lr_scheduler(optimizer, iteration, iteration)

                # inputs gen
                inputs = vae.decode(z).sample
                inputs = self.normalizer(inputs)

                im = original_img.clone().to(self._device)
                _inputs = torch.cat([inputs, im], dim=0)

                # 合成图像回传前，加入轻度数据增强
                aug_function = transforms.Compose([
                    transforms.RandomResizedCrop(self.input_size),
                    transforms.RandomHorizontalFlip(),
                ])
                #_inputs = aug_function(_inputs)

                im = _inputs[inputs.shape[0]:]
                with torch.no_grad():
                    model(im)
                    target_feat_lists = [mod.r_feature.clone().detach() for mod in loss_r_feature_layers]

                # _inputs = aug_function(inputs)
                _inputs = _inputs[:inputs.shape[0]]

                outputs = model(_inputs)
                input_feat_lists = [mod.r_feature for mod in loss_r_feature_layers]
                key_words = self.args.fedsd2c_loss.split("_")
                loss = 0
                loss_dict = {}
                for key_word in key_words:
                    cf = key_word.split("-")
                    if "gram" in cf:
                        loss_fn = gram_mse_loss
                    elif "factorization" in cf:
                        loss_fn = factorization_loss
                    else:
                        loss_fn = mse_loss

                    loss_feat = loss_fn(input_feat_lists[-1], target_feat_lists[-1], reduction="mean")
                    loss += loss_feat
                    loss_dict["feat"] = loss_feat.item()

                if self.r_bn > 0:
                    rescale = [self.first_bn_multiplier] + [1. for _ in range(len(loss_r_bn_layers) - 1)]
                    loss_r_bn = sum(
                        [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_bn_layers)])
                    loss += self.r_bn * loss_r_bn
                    loss_dict["r_bn"] = loss_r_bn.item()
                if self.r_c > 0:
                    loss_r_c = entropy_criterion(outputs, targets)
                    loss += self.r_c * loss_r_c
                    loss_dict["r_ce"] = loss_r_c.item()
                if self.r_adv > 0:
                    loss_r_adv = -mse_loss(inputs, original_img.clone().to(self._device), reduction="mean")
                    loss += self.r_adv * loss_r_adv
                    loss_dict["r_adv"] = loss_r_adv.item()
                assert loss != 0

                if best_cost > loss.item() or iteration >= 0:
                    best_inputs = inputs.data.cpu().clone()
                    if z is not None:
                        best_z = z.data.detach().cpu().clone()
                    best_cost = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                inputs.data = clip(inputs.data, dataset=self._dataset_id)
                losses.append(loss.item())
                for k, v in loss_dict.items():
                    if k not in loss_dicts:
                        loss_dicts[k] = [v]
                    else:
                        loss_dicts[k].append(v)
            if len(losses) == 0:
                losses = [0]

            # To simplify the implementation, we pass the decoded best_inputs directly back to the server,
            # skipping the process of decoding and decoding on the server
            ret_x.extend([data.squeeze() for data in torch.split(best_inputs, 1)])
            ori_x.extend([data.squeeze() for data in torch.split(original_img.data.cpu().clone(), 1)])
            ret_y.extend([data.squeeze() for data in torch.split(y.clone(), 1)])
            if "vae" in self.inputs_init:
                ret_z.extend([data.squeeze() for data in torch.split(best_z, 1)])

            logger.info("------------idx {} / {}----------".format(i * batch_size, len(self._distill_data['x'])))
            logger.info("loss avg: {}, final: {}, ".format(np.mean(losses), losses[-1]) + ", ".join(
                [f"{k}: {v[-1]}" for k, v in loss_dicts.items()]))
            
            for k, v in loss_dicts.items():
                if k not in loss_dict_list:
                    loss_dict_list[k] = [v]
                else:
                    loss_dict_list[k].append(v)

        if self.args.using_wandb:
            loss_mean = np.array(loss_list).mean(axis=0).tolist()
            loss_std = np.array(loss_list).std(axis=0).tolist()
            for i, loss in enumerate(loss_mean):
                wandb.log({
                    f"C{self._id} comp loss avg": loss,
                    f"C{self._id} comp loss std": loss_std[i],
                    "iteration": i,
                })
            for k, v in loss_dict_list.items():
                lm = np.array(v).mean(axis=0).tolist()
                ls = np.array(v).std(axis=0).tolist()
                for i, loss in enumerate(lm):
                    wandb.log({
                        f"C{self._id} {k} avg": loss,
                        f"C{self._id} {k} std": ls[i],
                        "iteration": i,
                    })
        del vae
        torch.cuda.empty_cache()
        return ret_x, ret_y
    
    # FedDIFD 核心方法
    def dif_synthesis_stage(self, model):
        """
        使用生成模型（diffusion）合成新样本（改为使用 DiT）
        特点：
        - 基于特征匹配优化潜在空间
        - 使用预训练 SDXL-VAE 模型
        - 多层特征损失计算
        - 这里使用 HuggingFace 的 DiTPipeline 来替换原来基于 StableDiffusion 的 UNet 流程
        - DiT 是以 ImageNet 类别条件生成为主（使用 class_labels）
        - 注意 ControlNet / SDXL-specific LoRA 可能无法直接在 DiT 上使用
        """
        logger = Logger()
        logger = logger.get_logger()

        ret_x = []
        ori_x = []
        ret_y = []
        ret_z = []
        loss_list = []
        loss_dict_list = {}

        # model init
        model = deepcopy(model)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # 加载预训练 VAE
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        for p in vae.parameters():
            p.requires_grad = False
        

        # 加载ControlNet Pipe 、加载预训练 DIFFUSION
        from diffusers import DiTPipeline, DPMSolverMultistepScheduler, DDIMScheduler
        
        # 选择合适的 DiT 模型，比如 facebook/DiT-XL-2-256
        diffusion_model = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256",torch_dtype=torch.float16).to(self._device)

        diffusion_model.vae = diffusion_model.vae.float()
        vae = diffusion_model.vae

        # 确保 transformer 是 float 类型（某些场景下需要精度对齐）
        diffusion_model.transformer = diffusion_model.transformer.float()
        # diffusion_model.load_lora_weights(
        #     "ArneNix/imagenet-lora",
        #     adapter_name = "sd21_lora_imagenet"
        # )
        # diffusion_model.set_adapters(["sd21_lora_imagenet"], adapter_weights=[0.8])
        
        scheduler = DDIMScheduler.from_config(diffusion_model.scheduler.config)
        diffusion_model.scheduler = scheduler

        print("======开始运行dif_synthesis_stage=====")
        # 加载预训练CLIP
        import clip as clipModel
        clip_model, preprocess = clipModel.load("ViT-B/32", device=self._device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        
        """ CLIP打分的分类及构造自然语言prompt及一系列操作开始：
        """
        # imagenet
        class_names  = {
            "imagenet" : {
                # imagenette的原始 10 类
                "n01440764": "fish",
                #"n02102040": "English springer",
                "n02102040": "dog",
                "n02979186": "cassette player",
                "n03000684": "chain saw",
                "n03028079": "church",
                "n03394916": "French horn",
                "n03417042": "garbage truck",
                "n03425413": "gas pump",
                "n03445777": "golf ball",
                "n03888257": "parachute"
            }
        }
        imagenette_class_ids = [
            0,      # 0 -> n01440764 (fish)
            207,    # 1 -> n02102040 (dog)
            314,    # 2 -> n02979186 (cassette player)
            477,    # 3 -> n03000684 (chain saw)
            536,    # 4 -> n03028079 (church)
            559,    # 5 -> n03394916 (French horn)
            569,    # 6 -> n03417042 (garbage truck)
            671,    # 7 -> n03425413 (gas pump)
            675,    # 8 -> n03445777 (golf ball)
            701     # 9 -> n03888257 (parachute)
        ]
        class_names = class_names["imagenet"]
        prompts = [f"a photo of a {class_names[label]}" for label in class_names]
        df_prompts = [f"a high quality, realistic photograph of a {class_names[label]} in natural lighting" for label in class_names]
        
        # 编码文本
        text_tokens = clipModel.tokenize(prompts).to(self._device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        """ CLIP打分的分类及构造自然语言prompt及一系列操作结束
        """

        # hook for loss computation init 特征钩子设置（捕获中间层特征）
        loss_r_feature_layers = []
        if isinstance(model, ResNet):
            loss_r_feature_layers.append(OutputHook(model.maxpool))
            for name, module in model.named_modules():
                if name in [f"layer{j}" for j in range(1, 5)]:
                    # print(f"Adding hook to {name}")
                    loss_r_feature_layers.append(OutputHook(module))
            # 为不同层添加钩子
            loss_r_feature_layers.append(OutputHook(model.avgpool))
        elif isinstance(model, ConvNet):
            for j in range(4):
                # print(f"Adding hook to {j} pool")
                loss_r_feature_layers.append(OutputHook(model.layers["pool"][j]))
        else:
            raise NotImplementedError()
        loss_r_bn_layers = []
        if self.r_bn > 0:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    loss_r_bn_layers.append(BNFeatureHook(module))
        # 构建合成数据集
        synset, batch_size = self._build_synset()
        synloader = torch.utils.data.DataLoader(synset, batch_size=batch_size, shuffle=False)
        for i, batch in enumerate(synloader):
            original_img, perturbed_img, y = batch

            # === 统一 float 并 resize ===
            target_size = (256, 256)  # <-- 对齐 DiT 输出尺寸
            original_img = original_img.to(self._device).float() / 255.0  # <-- 防止 Byte 导致 interpolate 报错
            perturbed_img = perturbed_img.to(self._device).float() / 255.0
            original_img = torch.nn.functional.interpolate(original_img, size=target_size, mode="bilinear", align_corners=False)
            perturbed_img = torch.nn.functional.interpolate(perturbed_img, size=target_size, mode="bilinear", align_corners=False)

            ### === 主要改动代码开始 ===
            ### 还原 original_img 为 [0,1] 转 PIL
            images = denormalize(original_img)  # [B, 3, H, W]
            pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]

            processed_images = []  # 存放处理后的图片tensor
            text_prompts_batch = []  # 存放每张图片对应的text_prompt

            for img_idx, pil_image in enumerate(pil_images):
                # 单张图输入 CLIP
                image_input = preprocess(pil_image).unsqueeze(0).to(self._device)
                
                with torch.no_grad():
                    image_feature = clip_model.encode_image(image_input)
                    image_feature /= image_feature.norm(dim=-1, keepdim=True)
                
                    # 计算与文本相似度
                    similarity = image_feature @ text_features.T  # [1, N]
                    best_idx = similarity.argmax(dim=1).item()
                    text_prompt = df_prompts[best_idx]

                    # 记录这个图片的文本描述
                    text_prompts_batch.append(text_prompt)

                    # 生成 Patch 后的新图
                    from feddifd.locate.captureKey import run_bbox_generator
                    centre_img = run_bbox_generator(
                        model_id="stabilityai/stable-diffusion-2-1-base",
                        source_image=pil_image,
                        source_prompt=text_prompt,
                        iters=3,
                        guidance_scale=3,
                        word_idx=5
                    )
                    
                    import torch.nn.functional as F
                    target_size = original_img.shape[-2:]  # (H, W)
                    if centre_img.dim() == 3:
                        centre_img = centre_img.unsqueeze(0)  # [1, 3, H_c, W_c]
                    # 转成float并归一化
                    centre_img = centre_img.float() / 255.0
                    # 调整尺寸
                    centre_img_resized = F.interpolate(centre_img, size=target_size, mode='bilinear', align_corners=False)
                    # 如果后续需要恢复到[0,255]和Byte类型，可以再转回
                    centre_img_resized = (centre_img_resized * 255).byte()
                    processed_images.append(centre_img_resized)

            # 拼回原batch
            original_img = torch.cat(processed_images, dim=0).to(self._device)  # [B, 3, H, W]
            ### === 主要改动代码结束 ===

            # distillate initialization with fourier transformation
            with torch.no_grad():
                # 将 VAE 和模型移动到设备
                vae.to(self._device)
                diffusion_model.to(self._device)
                target_size = (224, 224)  # 对齐 DiT 输出
    
                # 1. 潜在空间初始化，将扰动图像归一化并转换为潜在空间的表示
                if "fourier" in self.inputs_init:
                    # resized_img = torch.nn.functional.interpolate(
                    #     denormalize(perturbed_img).to(self._device),
                    #     size=target_size,
                    #     mode="bilinear",
                    #     align_corners=False
                    # )
                    z = vae.encode(denormalize(perturbed_img).to(self._device)).latent_dist.mode().clone().detach()
                else:
                    # resized_img = torch.nn.functional.interpolate(
                    #     denormalize(original_img).to(self._device),
                    #     size=target_size,
                    #     mode="bilinear",
                    #     align_corners=False
                    # )
                    z = vae.encode(denormalize(original_img).to(self._device)).latent_dist.mode().clone().detach()

            # 设置扩散步数
            num_inference_steps = 10
            scheduler.set_timesteps(num_inference_steps)

            # 初始噪声（可选：如果 z 已经是带噪声的，跳过此步）
            noise = torch.randn_like(z)
            timestep = scheduler.timesteps[torch.randint(0, len(scheduler.timesteps), (1,))] # 随机选择时间步，增强多样性
            noisy_z = scheduler.add_noise(z, noise, timestep)

            if isinstance(y, torch.Tensor):
                y_list = y.tolist()
            else:
                y_list = list(y)
            # 把 0~9 的 imagenette 索引映射到 ImageNet id 列表
            class_ids = [imagenette_class_ids[idx] for idx in y_list]  # list[int], len == batch_size

            with torch.no_grad():
                generator = torch.manual_seed(42)
                output = diffusion_model(
                    class_labels=class_ids,        # 类 ID 条件
                    num_inference_steps=25,        # 迭代步数
                    guidance_scale=4.0,            # classifier-free guidance scale
                    generator=generator,
                    output_type="pil"              # 直接输出 PIL 图像
                )

                pil_imgs = output.images          # list[PIL.Image]
                to_tensor = transforms.ToTensor()
                img_tensors = torch.stack([to_tensor(p).to(self._device) for p in pil_imgs])  # [B,3,256,256]

            # 直接作为最终生成结果
            denoised_imgs = img_tensors
                
            targets = y.to(self._device)
            entropy_criterion = nn.CrossEntropyLoss()
            z.requires_grad = True
            optimizer = torch.optim.AdamW([z], lr=self.sre2l_lr, betas=(0.5, 0.9), eps=1e-8)
            lr_scheduler = lr_cosine_policy(self.sre2l_lr, 0, self.iterations_per_layer)

            best_inputs = None
            best_z = None
            best_cost = 1e4
            losses = []
            loss_dicts = {}
            # 优化循环
            for iteration in range(self.iterations_per_layer):
                lr_scheduler(optimizer, iteration, iteration)

                # 使用 denoised 图像作为 inputs
                inputs = denoised_imgs.clone().detach().requires_grad_(True)

                # 统一目标尺寸为 ResNet 默认输入
                target_size = (224, 224)  # ResNet 预训练的默认输入
                
                # 1. 原图也要转 float / 归一化，并 resize
                im = original_img.clone().to(self._device).float() / 255.0
                im = F.interpolate(im, size=target_size, mode="bilinear", align_corners=False)

                # 2. 将 denoised inputs 也 resize 到 target_size，避免尺寸不匹配
                inputs_resized = F.interpolate(inputs.float(), size=target_size, mode="bilinear", align_corners=False)

                # 3. 拼接 inputs 与原图，保证 batch 拼接不会报错
                _inputs = torch.cat([inputs_resized, im], dim=0)

                # DEBUG 输出尺寸
                print("DEBUG: _inputs shape before model =", _inputs.shape)  # [B_inputs+B_original, 3, 224, 224]

                # 合成图像回传前，加入轻度数据增强
                aug_function = transforms.Compose([
                    transforms.RandomResizedCrop(self.input_size),
                    transforms.RandomHorizontalFlip(),
                ])
                #_inputs = aug_function(_inputs)

                 # 4. 取原图部分，送入模型计算特征
                im_part = _inputs[inputs_resized.shape[0]:]  # 取原图部分
                im_part = F.interpolate(im_part.float(), size=target_size, mode='bilinear', align_corners=False)
                
                print("DEBUG: im_part.shape =", im_part.shape)
                print("DEBUG: im_part.dtype =", im_part.dtype)
                print("DEBUG: im_part.min =", im_part.min().item(), "max =", im_part.max().item())

                with torch.no_grad():
                    model(im_part)
                    #target_feat_lists = [mod.r_feature.clone().detach() for mod in loss_r_feature_layers]
                    target_feat_lists = []
                    for mod in loss_r_feature_layers:
                        # 获取特征并立即归一化
                        feat = mod.r_feature.clone().detach()
                        # L2 归一化特征（核心修复）
                        feat = feat / (feat.norm() + 1e-8)  # 避免除以0
                        target_feat_lists.append(feat)

                # 5. 可选：如果需要对 inputs 做进一步处理，保留 _inputs 前半部分
                _inputs = _inputs[:inputs_resized.shape[0]]

                # outputs = model(_inputs)
                # input_feat_lists = [mod.r_feature for mod in loss_r_feature_layers]
                # 修改后：添加特征归一化
                outputs = model(_inputs)
                input_feat_lists = []
                for mod in loss_r_feature_layers:
                    feat = mod.r_feature  # 不需要 detach，因为需要梯度
                    feat = feat / (feat.norm() + 1e-8)  # L2 归一化
                    input_feat_lists.append(feat)
                key_words = self.args.fedsd2c_loss.split("_")

                # 计算特征匹配损失
                loss = 0
                loss_dict = {}
                for key_word in key_words:
                    cf = key_word.split("-")
                    if "gram" in cf:
                        loss_fn = gram_mse_loss
                    elif "factorization" in cf:
                        loss_fn = factorization_loss
                    else:
                        loss_fn = mse_loss

                    loss_feat = loss_fn(input_feat_lists[-1], target_feat_lists[-1], reduction="mean")
                    loss += loss_feat
                    loss_dict["feat"] = loss_feat.item()

                if self.r_bn > 0:
                    rescale = [self.first_bn_multiplier] + [1. for _ in range(len(loss_r_bn_layers) - 1)]
                    loss_r_bn = sum(
                        [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_bn_layers)])
                    loss += self.r_bn * loss_r_bn
                    loss_dict["r_bn"] = loss_r_bn.item()
                if self.r_c > 0:
                    loss_r_c = entropy_criterion(outputs, targets)
                    loss += self.r_c * loss_r_c
                    loss_dict["r_ce"] = loss_r_c.item()
                if self.r_adv > 0:
                    loss_r_adv = -mse_loss(inputs, original_img.clone().to(self._device), reduction="mean")
                    loss += self.r_adv * loss_r_adv
                    loss_dict["r_adv"] = loss_r_adv.item()
                assert loss != 0

                if best_cost > loss.item() or iteration >= 0:
                    best_inputs = inputs.data.cpu().clone()
                    if z is not None:
                        best_z = z.data.detach().cpu().clone()
                    best_cost = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                inputs.data = clip(inputs.data, dataset=self._dataset_id)
                losses.append(loss.item())

                for k, v in loss_dict.items():
                    if k not in loss_dicts:
                        loss_dicts[k] = [v]
                    else:
                        loss_dicts[k].append(v)
            if len(losses) == 0:
                losses = [0]

            # To simplify the implementation, we pass the decoded best_inputs directly back to the server,
            # skipping the process of decoding and decoding on the server 保存最佳合成结果
            ret_x.extend([data.squeeze() for data in torch.split(best_inputs, 1)])
            ori_x.extend([data.squeeze() for data in torch.split(original_img.data.cpu().clone(), 1)])
            ret_y.extend([data.squeeze() for data in torch.split(y.clone(), 1)])
            if "vae" in self.inputs_init:
                ret_z.extend([data.squeeze() for data in torch.split(best_z, 1)])

            logger.info("------------idx {} / {}----------".format(i * batch_size, len(self._distill_data['x'])))
            logger.info("loss avg: {}, final: {}, ".format(np.mean(losses), losses[-1]) + ", ".join(
                [f"{k}: {v[-1]}" for k, v in loss_dicts.items()]))
            
            # import pandas as pd
            # df = pd.DataFrame(loss_dicts)
            # print("\n=== Loss History ===")
            # print(df.tail(10))  # 打印最后 10 次迭代的损失
            loss_list.append(losses)

            loss_list.append(losses)
            for k, v in loss_dicts.items():
                if k not in loss_dict_list:
                    loss_dict_list[k] = [v]
                else:
                    loss_dict_list[k].append(v)

        if self.args.using_wandb:
            loss_mean = np.array(loss_list).mean(axis=0).tolist()
            loss_std = np.array(loss_list).std(axis=0).tolist()
            for i, loss in enumerate(loss_mean):
                wandb.log({
                    f"C{self._id} comp loss avg": loss,
                    f"C{self._id} comp loss std": loss_std[i],
                    "iteration": i,
                })
            for k, v in loss_dict_list.items():
                lm = np.array(v).mean(axis=0).tolist()
                ls = np.array(v).std(axis=0).tolist()
                for i, loss in enumerate(lm):
                    wandb.log({
                        f"C{self._id} {k} avg": loss,
                        f"C{self._id} {k} std": ls[i],
                        "iteration": i,
                    })
        del vae
        torch.cuda.empty_cache()

        return ret_x, ret_y

    def decode_latents(self, latents):
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()
        vae.to(self._device)
        bs = self.ipc * self.factor
        samples = []
        rng = np.random.default_rng(self.args.sys_i_seed)
        with torch.no_grad():
            for kk in range(0, len(latents), bs):
                z = latents[kk:kk + bs].to(self._device)
                if self.noise_type == "gaussian":
                    noise = torch.tensor(rng.normal(size=z.numel()), dtype=z.dtype).reshape(z.shape).to(
                        self._device) * self.noise_s
                    z = (1 - self.noise_p) * z + noise
                elif self.noise_type == "laplace":
                    noise = torch.tensor(rng.laplace(size=z.numel()), dtype=z.dtype).reshape(z.shape).to(
                        self._device) * self.noise_s
                    z = (1 - self.noise_p) * z + noise
                elif self.noise_type == "None":
                    pass
                else:
                    raise NotImplementedError()
                sample = vae.decode(z).sample.detach().clone().cpu()
                sample = self.normalizer(sample)
                samples.extend([data.squeeze() for data in torch.split(sample, 1)])

        return samples

    @property
    def all_select(self):
        """
        The client uploads all of the original dataset
        :return: All of the original images
        """
        return self._train_data

    def save_distilled_dataset(self, exp_dir='client_models', res_root='results'):
        """
        The client saves the distilled images in corresponding directory
        :param exp_dir: Experiment directory name
        :param res_root: Result directory root for saving the result files
        """
        agent_name = 'clients'
        model_save_dir = os.path.join(res_root, exp_dir, agent_name)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        torch.save(self._distill_data, os.path.join(model_save_dir, self._id + '_distilled_img.pt'))

    def _build_synset(self):
        dx1, dx2, dy = [], [], []
        for i in range(0, len(self._distill_data['x']), self.ipc * self.factor):
            idxs = np.random.permutation(self.ipc * self.factor).tolist()
            subset_x = torch.stack([self._distill_data['x'][i + idx] for idx in idxs])
            subset_y = torch.stack([self._distill_data['y'][i + idx] for idx in idxs])

            corres_idxs = np.where(np.array(self._rest_data['y']) == subset_y[0].item())[0]
            rest_x = torch.stack([self._rest_data['x'][idx] for idx in corres_idxs])
            rest_dists = torch.stack([self._rest_data['dist'][idx] for idx in corres_idxs])
            rest_preds = torch.stack([self._rest_data['pred'][idx] for idx in corres_idxs])

            indices = np.where(torch.argmax(rest_preds).numpy() == subset_y[0].item())[0]
            if indices.shape[0] != 0:
                rest_x, rest_dists = rest_x[indices], rest_dists[indices]
            indices = torch.argsort(rest_dists, descending=True)[:subset_x.shape[0]]
            if indices.shape[0] < subset_x.shape[0]:
                indices = indices.repeat((subset_x.shape[0] // indices.shape[0]) + 1)[:subset_x.shape[0]]
            rest_x = rest_x[indices]

            dx1.append(subset_x)
            dx2.append(rest_x)
            dy.append(subset_y)
        dx1 = torch.stack(dx1, dim=0)
        dx2 = torch.stack(dx2, dim=0)
        dy = torch.stack(dy, dim=0)

        if self.iter_mode == "random" or self.iter_mode == "label":
            bs = self.ipc * self.factor
        elif self.iter_mode == "ipc":
            bs = dx1.shape[0]

        return SynDataset(dx1, dx2, dy, self.iter_mode, fourier="fourier" in self.inputs_init,
                          fourier_lambda=self.args.fourier_lambda, dataset=self._dataset_id), bs