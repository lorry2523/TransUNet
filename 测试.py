import numpy as np
import torch
from utils import compute_metrics

# 1. 检查一个验证集样本的标签
data = np.load('G:/Strain_Project/TransUNet/data/Synapse/val_npz/0.npz')
label = data['label']
print("Label unique values:", np.unique(label))          # 应该包含 0 和 1
print("Foreground pixel ratio:", (label == 1).sum() / label.size)

# 2. 测试 compute_metrics 函数
pred = torch.tensor([0, 0, 1, 1])      # 模拟预测
true = torch.tensor([0, 1, 0, 1])      # 模拟真值
iou, dice, pa = compute_metrics(pred, true, num_classes=2)
print(f"IoU: {iou}, Dice: {dice}, PA: {pa}")  # 应该非零