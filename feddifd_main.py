import copy
import os
import random
import signal
import sys
from typing import Dict, Set, Tuple

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils import Logger, fed_args2, read_config, log_config, AverageMeter
from utils.fed_utils import init_model, assign_dataset
from preprocessing.baselines_dataloader import divide_data_with_dirichlet
from feddifd.client import FedDIFDClient
from feddifd.server import FedDIFDServer
from feddifd.util import DistilledDataset

# Initialize parameters
args = fed_args2()
args = read_config(args.config, args)

# Set up logging
if Logger.logger is None:
    logger = Logger()
    os.makedirs("train_records/", exist_ok=True)
    logger.set_log_name(os.path.join("train_records", f"train_record_{args.save_name}.log"))
    logger = logger.get_logger()
    log_config(args)

using_wandb = args.using_wandb

# Clear GPU cache
torch.cuda.empty_cache()

# 标准测试流程，返回整体准确率
def test_model(model: nn.Module, testset, batch_size: int, device: str) -> float:
    """Test model accuracy on test set"""
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False) 
    model.to(device)
    correct = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            
    return correct / len(testset)

# 针对特定标签的测试，返回整体准确率+各类别详细准确率
def test_specified(model: nn.Module, testset, batch_size: int, device: str, 
                  specified_labels: list) -> Tuple[float, Dict]:
    """Test model accuracy on specified labels"""
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    model.to(device)
    correct = 0
    specified_acc = {label: AverageMeter() for label in specified_labels}
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            
            for label in specified_labels:
                mask = (y == label)
                if mask.any():
                    acc = (pred[mask] == y[mask]).float().mean().item()
                    specified_acc[label].update(acc, mask.sum().item())
                    
    return correct / len(testset), specified_acc

# 筛选需要蒸馏的类别：
    # 1. 统计客户端数据包含的所有类别
    # 2. 标记样本量大于 ipc*factor² 的类别进行蒸馏
def get_specified_labels(client, ipc: int) -> Tuple[Set, Set]:
    """Get label sets that need to be distilled"""
    dataset = client._train_data
    indices = np.array(dataset.indices)
    targets = np.array(dataset.dataset.targets, dtype=np.int64)[indices]
    unique_classes = np.unique(targets)
    
    containing_labels = set()
    distilled_labels = set()
    
    for c in unique_classes:
        containing_labels.add(c)
        if (targets == c).sum() > ipc:
            distilled_labels.add(c)
            
    return distilled_labels, containing_labels

def setup_environment():
    """Set up runtime environment"""
    # Set random seeds
    random.seed(args.sys_i_seed)
    np.random.seed(args.sys_i_seed)
    torch.manual_seed(args.sys_i_seed)
    torch.cuda.manual_seed(args.sys_i_seed)
    
    # Set CUDNN
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Set number of threads
    torch.set_num_threads(10)

def main():
    """Main function"""
    setup_environment()
    
    # Initialize client dictionary
    client_dict = {}
    
    # Data partitioning
    logger.info('======================Setup Clients==========================')
    if args.sys_dataset_dir_alpha is None:
        raise NotImplementedError("sys_dataset_dir_alpha is None")
        
    logger.info('Using divide data with dirichlet')
    # 使用狄利克雷分布划分非IID数据
    trainset_config, testset, cls_record = divide_data_with_dirichlet(
        n_clients=args.sys_n_client,
        beta=args.sys_dataset_dir_alpha,
        dataset_name=args.sys_dataset,
        seed=42
    )
    logger.info(f'Clients in Total: {len(trainset_config["users"])}')

    # Initialize server
    server = FedDIFDServer(
        args, trainset_config['users'],  # 客户端列表
        epoch=args.server_n_epoch,
        batch_size=args.server_bs,
        lr=args.server_lr,
        momentum=args.server_momentum,
        num_workers=args.server_n_worker,
        dataset_id=args.sys_dataset,
        model_name=args.sys_model,
        i_seed=args.sys_i_seed
    )
    server.load_testset(testset) # 加载测试集

    # Get dataset parameters
    num_class, img_dim, image_channel = assign_dataset(args.sys_dataset)

    # Client training loop
    for client_id in trainset_config['users']:
        # Initialize client 1. 初始化客户端
        client = FedDIFDClient(args, client_id, dataset_id=args.sys_dataset)
        client_dict[client_id] = client
        client.load_train(trainset_config['user_data'][client_id])
        client.load_cls_record(cls_record[client_id])

        # Initialize model 2. 模型初始化/加载
        model = init_model(args.sys_model, num_class, image_channel, im_size=img_dim)

        # 3. 选择蒸馏标签
        specified_labels, containing_labels = get_specified_labels(
            client,
            client.ipc * client.factor ** 2
        )

        # Load or train model 4. 本地训练或加载预训练模型
        if args.client_model_root is not None:
            model_path = os.path.join(args.client_model_root, f"c{client_id}.pt")
            weight = torch.load(model_path, map_location="cpu")
            logger.info(f"Load Client {client_id} from {model_path}")
            model.load_state_dict(weight)
            model = model.to(client._device)
        else:
            logger.info(f"Client {client_id} local training")
            model, _ = client.train(model)

        # Execute distillation 5. 知识蒸馏执行
        if args.client_instance == "coreset":
            ret_x, ret_y = client.coreset_stage(model)
        elif args.client_instance == "coreset_clip":
            ret_x, ret_y = client.coreset_stage_clip(model)
        elif args.client_instance == "random":
            ret_x, ret_y = client.random_stage(model)
        elif args.client_instance == "coreset+dist_syn":
            ret_x, ret_y = client.coreset_stage(model)
            ret_x, ret_y = client.synthesis_stage(model)
        elif args.client_instance == "coreset+_dif_dist_syn":
            ret_x, ret_y = client.coreset_stage(model)
            ret_x, ret_y = client.dif_synthesis_stage(model)
        elif args.client_instance == "random+dist_syn":
            ret_x, ret_y = client.random_stage(model)
            ret_x, ret_y = client.synthesis_stage(model)
        else:
            raise NotImplementedError("Not implemented yet.")

        
        # Server receives distillation results
        # Data augmentation 6. 数据增强处理
        augment = transforms.Compose([
            transforms.RandomResizedCrop(size=img_dim, scale=(1, 1), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2)
        ])
        server.rec_distill(
            client._id,
            model,
            DistilledDataset(ret_x, ret_y, augment),
            list(specified_labels)
        )

    # Server training 7. 上传蒸馏数据到服务器
    server.train_distill()  # 使用所有客户端蒸馏数据训练全局模型

# 处理程序终止信号（Ctrl+C）
def term_sig_handler(signum, frame):
    """Signal handler function"""
    print(f'caught signal: {signum}')
    if using_wandb:
        wandb.finish()  # 优雅关闭wandb
    sys.exit()

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, term_sig_handler)
    signal.signal(signal.SIGINT, term_sig_handler)
    
    # Run main program
    main()
    
    if using_wandb:
        wandb.finish()
