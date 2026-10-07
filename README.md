# FedDIFD

FedDIFD 是基于 FedSD2C 扩展的单次通信联邦学习实验项目。客户端先训练本地分类模型，再选择或合成少量代表图像；服务器收集客户端模型与图像，以知识蒸馏训练全局分类模型。

本项目的扩散扩展尝试结合 CLIP 语义选择、Stable Diffusion 注意力定位和 DiT 图像生成。**当前扩散分支仍在开发，存在未定义变量和未接通的优化流程，不能直接完成端到端训练。** 以下文档区分现有实现与待完成的扩展，不代表已验证的实验复现结果。

## 方法概览

```text
真实训练数据
    │ Dirichlet 非 IID 划分
    ▼
各客户端本地分类模型
    │ 加载已有权重，或本地训练
    ▼
代表数据选择：Coreset / Random
    │
    ├── 直接使用代表图像
    ├── SDXL VAE 潜变量优化
    └── CLIP + 目标定位 + DiT 扩散合成（开发中）
    ▼
上传客户端模型与压缩图像
    │ 来源客户端模型生成软标签
    ▼
服务器以 KL 散度训练全局学生模型
```

客户端与服务器目前在同一进程中顺序模拟，没有实际网络通信协议。通信内容包含本地模型和压缩图像，并非仅上传模型参数或图像。

### 代表图像选择

`coreset_stage()` 对每类候选图像生成多个裁剪，以本地模型的分类交叉熵打分。每张图像保留损失最低的裁剪，再选择低损失样本。`mipc` 控制每类候选数量，`ipc` 控制每类目标图像数量。

### VAE 图像合成

`synthesis_stage()` 冻结本地分类模型与 SDXL VAE，将代表图像或 Fourier 扰动图像编码为潜变量，优化潜变量，使解码图像的分类模型特征接近原图特征。支持 MSE、Gram MSE、factorization 等特征损失，以及可选的 BN 统计和分类约束。默认特征损失主要比较最后一个已注册钩子的输出。

### 扩散合成（开发中）

`dif_synthesis_stage()` 尝试使用 CLIP 从 Imagenette 的类别提示词中选择语义，通过 Stable Diffusion 注意力定位目标区域，再由 DiT 生成图像并进行特征匹配。当前定位结果尚未接入生成，生成变量与优化器尚未连接，详见“已知限制”。

### 服务器蒸馏

服务器合并各客户端图像并保留来源 ID。每张图像由其来源客户端模型提供软标签，全局学生模型使用带温度的 KL 散度训练，当前温度为 1。训练损失没有直接使用图像的硬标签。

## 目录结构

```text
FedDIFD/
├── feddifd_main.py              # FedDIFD 主入口
├── fedsd2c_main.py              # FedSD2C 对照入口
├── oneshot_main.py              # 本地模型训练与单次通信基线
├── feddifd/
│   ├── client.py                # 本地训练、数据选择、VAE/扩散合成
│   ├── server.py                # 教师软标签与全局模型蒸馏
│   ├── util.py                  # 数据包装、Fourier 扰动、损失与钩子
│   └── locate/captureKey.py     # Stable Diffusion 注意力目标定位
├── fedsd2c/                     # FedSD2C 实现及实验扩展
├── fed_baselines/               # ENSEMBLE、FedAVG、DENSE、CoBoost 等
├── configs/imagenette/          # Imagenette 配置，包括 feddifd.yaml
├── configs/tinyimagenet/        # TinyImageNet 对照实验配置
├── preprocessing/              # 数据集加载与客户端划分
├── utils/                      # 分类模型、参数解析、配置、日志
├── postprocessing/             # 指标读取与绘图工具
├── shells/                     # 基础实验 Bash 脚本
├── shells-My/                  # 自定义实验 Bash 脚本
├── syntheticData/              # 已有合成数据文件
├── DiT/、ControlNet/、dift/      # 引入的生成/扩散特征工程
├── InstantSwap/、X-Adapter/     # 引入的图像编辑工程
└── origin/                     # 参考代码副本
```

第三方工程目录的存在不表示全部已接入 FedDIFD 主流程。当前客户端直接使用 Diffusers 模型接口与 `feddifd/locate/` 定位模块。带 `backup` 的文件是实验副本，主入口导入的是 `feddifd/client.py`。

## 环境准备

根目录 `requirements.txt` 声明的主要版本：

| 组件 | 版本 |
| --- | --- |
| PyTorch / Torchvision | 2.4.0 / 0.19.0 |
| Diffusers | 0.29.2 |
| NumPy / SciPy | 1.24.3 / 1.14.1 |
| Pillow | 10.4.0 |
| PyYAML | 6.0.2 |
| Weights & Biases | 0.18.7 |

建议使用独立的 Python 3.10 环境，避免沿用旧 README 的 Python 3.8 环境说明。PyTorch 安装需要与实际 CUDA 环境匹配。

```bash
python -m pip install -r requirements.txt
```

**根目录依赖列表尚不完整。** FedDIFD 顶层还会导入 OpenAI CLIP 和 OpenCV；Diffusers 模型加载也需要兼容的 Transformers、Accelerate 等依赖。补充安装方式如下，具体版本组合仍需在目标环境验证：

```bash
python -m pip install opencv-python transformers accelerate
python -m pip install git+https://github.com/openai/CLIP.git
```

即使选择 `coreset`，顶层导入仍要求扩散、CLIP 和定位模块相关依赖。不要将所有子工程的 requirements 混合安装：X-Adapter、InstantSwap 等声明了不同的 Diffusers/PyTorch 版本。

合成阶段会加载以下预训练模型，需要网络访问或本地缓存，并占用额外显存：

| 用途 | 模型标识 |
| --- | --- |
| VAE 编码/解码 | `stabilityai/sdxl-vae` |
| 扩散生成 | `facebook/DiT-XL-2-256` |
| 目标区域定位 | `stabilityai/stable-diffusion-2-1-base` |
| 语义匹配 | CLIP `ViT-B/32` |

## 数据与配置

当前 FedDIFD 示例面向 **Imagenette：10 类、RGB、128×128 分类输入**。数据默认放在项目根目录的 `data/` 下，加载器在需要时下载 Imagenette。客户端使用 Dirichlet 类别分布划分，并要求每个客户端至少有 8 个样本。

基础加载器还包含 CIFAR10、CIFAR100、TinyImageNet、OpenImage、COVID 等分支，但扩散提示词和部分图像处理逻辑固定为 Imagenette，不能据此认为 FedDIFD 扩散分支支持全部数据集。

配置文件：`configs/imagenette/feddifd.yaml`。命令行参数由 `utils/arg_parser2.py` 定义。

| 参数 | 含义 |
| --- | --- |
| `-c` | YAML 配置路径 |
| `-sn` | 实验名称前缀，必填 |
| `-nc` | 客户端数量 |
| `-dda` | Dirichlet alpha，主流程要求提供 |
| `-md` | 分类模型，例如 `Conv5`、`ResNet18` |
| `-is` | 实验随机种子 |
| `-g` | GPU 编号，显式传 `0` 等数字字符串 |
| `-cmr` | 已训练客户端权重目录，内部为 `c0.pt` 等 |
| `-cis` | 数据选择/合成分支 |
| `--feddifd_mipc` | 每类候选图像数量 |
| `--feddifd_ipc` | 每类目标压缩图像数量 |
| `--feddifd_num_crop` | Coreset 候选裁剪数量 |
| `--feddifd_iteration` | 合成优化步数 |
| `--feddifd_lr` | 合成优化学习率 |
| `--feddifd_inputs_init` | 初始化方式，例如 `vae+fourier` |
| `--feddifd_iter_mode` | 合成批次组织方式；当前建议显式使用 `label` |
| `--feddifd_loss` | 特征损失，默认 `mse` |
| `--fourier_lambda` | Fourier 扰动强度 |
| `-sne / -sbs / -slr` | 服务器训练 epoch、batch size、学习率 |
| `-so` | 服务器优化器：`SGD`、`Adam`、`AdamW` |

配置合并规则是：**命令行解析结果只要不为 `None`，就不会被 YAML 覆盖。** 因而 argparse 默认值也会覆盖 YAML，例如 YAML 中的 `feddifd_iter_mode: ipc` 实际会被默认 `label` 覆盖。启动日志会输出最终配置，应以日志为准。

## 运行流程

从项目根目录运行。以下命令使用单行形式，可在 Bash 或 PowerShell 中使用；完整训练尚未在当前环境验证。先补齐环境，并检查文末已知限制。

### 1. 训练并保存客户端模型

以 Imagenette、Conv5、10 个客户端、alpha=0.1、seed=42 为例：

```bash
python oneshot_main.py -c configs/imagenette/ensemble.yaml -dda 0.1 -md Conv5 -is 42 -sn Baseline_Ensemble_dir0.1_nc10 -g 0 -nc 10 --save_client_model
```

保存目录为：

```text
train_results/Baseline_Ensemble_dir0.1_nc10_ENSEMBLE_Conv5_Imagenette_s42/
├── c0.pt
├── c1.pt
└── ...
```

后续实验须保持数据集、客户端数量、划分参数、种子和模型结构一致。当前 FedDIFD 数据划分种子固定为 42，因此此处也使用 42。

### 2. 使用 Coreset 作为流程参照

该分支绕过合成阶段，用于检查权重加载、代表图像选择与服务器蒸馏。

```bash
python feddifd_main.py -c configs/imagenette/feddifd.yaml -dda 0.1 -md Conv5 -is 42 -sn FedDIFD_coreset_dir0.1_nc10 -g 0 -nc 10 -cmr train_results/Baseline_Ensemble_dir0.1_nc10_ENSEMBLE_Conv5_Imagenette_s42 -cis coreset --feddifd_ipc 5
```

### 3. 普通 VAE 合成分支

```bash
python feddifd_main.py -c configs/imagenette/feddifd.yaml -dda 0.1 -md Conv5 -is 42 -sn FedDIFD_vae_dir0.1_nc10 -g 0 -nc 10 -cmr train_results/Baseline_Ensemble_dir0.1_nc10_ENSEMBLE_Conv5_Imagenette_s42 -cis coreset+dist_syn --feddifd_ipc 5 --feddifd_inputs_init vae+fourier --feddifd_iter_mode label
```

此分支仍需检查类别样本量、VAE 图像数值范围和 `clip` 名称冲突，不能视为已验证可复现命令。

### 4. 扩散扩展分支（修复后使用）

```bash
python feddifd_main.py -c configs/imagenette/feddifd.yaml -dda 0.1 -md Conv5 -is 42 -sn FedDIFD_diffusion_dir0.1_nc10 -g 0 -nc 10 -cmr train_results/Baseline_Ensemble_dir0.1_nc10_ENSEMBLE_Conv5_Imagenette_s42 -cis coreset+_dif_dist_syn --feddifd_ipc 5 --feddifd_inputs_init vae+fourier --feddifd_iter_mode label
```

`coreset+_dif_dist_syn` 是当前代码中的准确参数拼写。现有实现会报错，必须先完成扩散方法修复。`shells-My/nette_feddifd_conv5.sh` 同样使用这一分支，其 GPU、客户端数和权重路径应按实际实验调整；现有两个自定义脚本并非可直接串联的一组配置。

其他分支：`random`、`random+dist_syn`。`coreset_clip` 在入口中出现，但客户端尚未实现该方法。

省略 `-cmr` 时，FedDIFD 会在本次运行中训练本地模型；指定该参数时，会直接加载对应权重文件，文件缺失不会自动回退到训练。

## 输出与评估

- 日志写入 `train_records/train_record_<完整实验名>.log`，记录最终配置和训练准确率。
- 完整实验名按 `<前缀>_<分支>_<模型>_<数据集>_s<种子>` 拼接。
- `oneshot_main.py --save_client_model` 保存客户端分类权重。
- FedDIFD 常规 `train_distill()` 在内存中记录服务器指标，并可通过 `--using_wandb` 上报；当前没有自动保存最终全局权重或指标 JSON。
- `save_distilled_dataset()` 是辅助方法，主流程没有调用。
- `postprocessing/eval_main.py` 需要符合 Recorder 格式的结果文件，不能直接读取训练日志。
- 根目录 `test.py` 是独立 MNIST 测试示例，不是 FedDIFD 回归测试或全局模型推理入口。

## 已知限制与待完成工作

### 扩散分支阻断问题

`feddifd/client.py::dif_synthesis_stage()` 当前包含：

1. 使用 `self.device`，而客户端定义的是 `self._device`。
2. 使用未定义的 `z`、`inputs`，末尾删除未定义的 `vae`。
3. 优化器操作 `z`，生成过程却使用 `noise`，梯度链尚未接通。
4. `diffusion_model.decode(noise)` 未实现完整反向扩散采样流程。
5. 将分类 logits 与中间层特征进行匹配，比较对象不一致。
6. `centre_img` 定位结果没有参与生成或损失；定位只处理 batch 的第一张图像。
7. `import clip` 覆盖工具模块的同名裁剪函数，随后 `clip(...)` 调用会产生名称冲突。

### 配置与实验正确性

- GPU 参数默认是 `cuda:0`，设备构造又添加 `cuda:` 前缀；请显式传入 `-g 0`。CPU 回退与多 GPU 定位流程尚未统一。
- 定位模块固定使用 `cuda:0`，并重复加载 Stable Diffusion 模型，存在设备和显存管理问题。
- FedDIFD/FedSD2C 主入口的客户端划分固定使用 `seed=42`。
- “最佳合成图像”判断包含恒真条件 `iteration >= 0`，实际保留最后一次候选。
- `SynDataset` 的 `ipc` 模式对图像进行维度交换，但标签没有同步交换；修复前使用 `label` 模式。
- 合成数据构造假定每类有足够候选及剩余图像，低样本类别可能发生越界、空张量拼接或除零。
- 主入口传入的 ColorJitter 等增强在服务器合并数据时被绕过，实际主要使用水平翻转。
- VAE 输入/输出范围与分类模型标准化之间的转换需要进一步核对。

建议先验证 Coreset 流程，再修复普通 VAE 合成，最后接通扩散采样和优化；实验结果应在这些问题修复并完成端到端验证后报告。

## 阅读顺序

1. `shells-My/nette_feddifd_conv5.sh` 与 `configs/imagenette/feddifd.yaml`
2. `feddifd_main.py`
3. `utils/arg_parser2.py` 与 `utils/config.py`
4. `preprocessing/baselines_dataloader.py::divide_data_with_dirichlet()`
5. `feddifd/client.py::coreset_stage()` 与 `feddifd/util.py::selector_coreset()`
6. `synthesis_stage()`，再对照 `dif_synthesis_stage()`
7. `feddifd/locate/captureKey.py`
8. `feddifd/server.py::train_distill()`

## 来源与许可

本项目基于 FedSD2C 扩展，原 README 对应论文 *One-shot Federated Learning via Synthetic Distiller-Distillate Communication*（NeurIPS 2024）。FedDIFD 扩展与原论文实现应分别说明，原论文引用不能作为扩展分支的验证依据。

```bibtex
@inproceedings{NEURIPS2024_ba0ad9d1,
  author = {Zhang, Junyuan and Liu, Songhua and Wang, Xinchao},
  title = {One-shot Federated Learning via Synthetic Distiller-Distillate Communication},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  pages = {102611--102633},
  year = {2024}
}
```

基础代码还参考了 [FedD3](https://github.com/rruisong/FedD3)。项目许可证见 [LICENSE](LICENSE)；引入的第三方代码和模型须分别遵循其许可证及使用条款。
