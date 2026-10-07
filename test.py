import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import json

# 设置随机种子保证可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

class SimpleCNN(nn.Module):
    """简单的CNN模型用于测试"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class ModelTester:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"使用设备: {self.device}")
        
    def load_data(self, batch_size=64):
        """加载测试数据"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 使用MNIST数据集
        self.test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"测试集大小: {len(self.test_dataset)}")
        
    def basic_inference_test(self, num_samples=5):
        """基础推理测试"""
        print("\n=== 基础推理测试 ===")
        self.model.eval()
        
        # 测试推理时间
        dummy_input = torch.randn(1, 1, 28, 28).to(self.device)
        
        # Warm up
        for _ in range(10):
            _ = self.model(dummy_input)
        
        # 时间测试
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = self.model(dummy_input)
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) * 1000 / 100
        print(f"平均推理时间: {avg_inference_time:.2f} ms")
        
        # 内存使用
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_cached = torch.cuda.memory_reserved() / 1024**2
            print(f"GPU内存使用 - 已分配: {memory_allocated:.2f} MB, 缓存: {memory_cached:.2f} MB")
        
        return avg_inference_time
    
    def performance_test(self):
        """模型性能测试"""
        print("\n=== 模型性能测试 ===")
        self.model.eval()
        all_predictions = []
        all_targets = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.test_loader, desc="性能测试")):
                data, target = data.to(self.device), target.to(self.device)
                
                start_time = time.time()
                output = self.model(data)
                end_time = time.time()
                
                inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
                
                pred = output.argmax(dim=1, keepdim=True)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets)
        
        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"平均批次推理时间: {np.mean(inference_times):.2f} ± {np.std(inference_times):.2f} ms")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_inference_time': np.mean(inference_times)
        }
    
    def confusion_matrix_analysis(self):
        """混淆矩阵分析"""
        print("\n=== 混淆矩阵分析 ===")
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        cm = confusion_matrix(all_targets, all_predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def robustness_test(self, noise_level=0.1):
        """鲁棒性测试 - 添加噪声"""
        print(f"\n=== 鲁棒性测试 (噪声水平: {noise_level}) ===")
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="鲁棒性测试"):
                # 添加高斯噪声
                noise = torch.randn_like(data) * noise_level
                noisy_data = torch.clamp(data + noise, 0, 1)
                
                noisy_data, target = noisy_data.to(self.device), target.to(self.device)
                output = self.model(noisy_data)
                pred = output.argmax(dim=1)
                
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        robust_accuracy = 100. * correct / total
        print(f"噪声条件下的准确率: {robust_accuracy:.2f}%")
        
        return robust_accuracy
    
    def model_size_analysis(self):
        """模型大小分析"""
        print("\n=== 模型大小分析 ===")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        
        # 保存模型并计算文件大小
        torch.save(self.model.state_dict(), 'temp_model.pth')
        import os
        model_size = os.path.getsize('temp_model.pth') / 1024**2  # MB
        print(f"模型文件大小: {model_size:.2f} MB")
        os.remove('temp_model.pth')
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size
        }
    
    def comprehensive_test(self, save_results=True):
        """综合测试"""
        print("开始综合测试...")
        
        results = {}
        
        # 加载数据
        self.load_data()
        
        # 运行各种测试
        results['inference_time'] = self.basic_inference_test()
        results['performance'] = self.performance_test()
        results['model_size'] = self.model_size_analysis()
        results['robustness'] = self.robustness_test()
        
        # 混淆矩阵
        results['confusion_matrix'] = self.confusion_matrix_analysis()
        
        # 保存结果
        if save_results:
            # 转换为可JSON序列化的格式
            saveable_results = {}
            for key, value in results.items():
                if key == 'confusion_matrix':
                    saveable_results[key] = value.tolist()
                elif isinstance(value, dict):
                    saveable_results[key] = value
                else:
                    saveable_results[key] = float(value) if isinstance(value, (int, float)) else str(value)
            
            with open('test_results.json', 'w', encoding='utf-8') as f:
                json.dump(saveable_results, f, indent=2, ensure_ascii=False)
            print("\n测试结果已保存到 test_results.json")
        
        return results

def main():
    """主函数"""
    # 初始化模型
    model = SimpleCNN(num_classes=10)
    
    # 加载预训练权重（如果有的话）
    # 如果没有，可以训练一个简单的模型或使用随机权重测试
    
    # 创建测试器
    tester = ModelTester(model)
    
    # 运行综合测试
    results = tester.comprehensive_test()
    
    # 打印总结
    print("\n" + "="*50)
    print("测试总结:")
    print(f"设备: {tester.device}")
    print(f"准确率: {results['performance']['accuracy']:.4f}")
    print(f"模型大小: {results['model_size']['model_size_mb']:.2f} MB")
    print(f"参数数量: {results['model_size']['total_params']:,}")
    print("="*50)

if __name__ == "__main__":
    main()