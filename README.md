# PCG-DL: Physics-Constrained Deep Learning for Rolling Bearing Diagnosis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

物理约束深度学习方法在滚动轴承故障诊断中的应用 | Physics-Constrained Deep Learning for Bearing Fault Diagnosis

## 📖 项目概述

本仓库提出了一种新型物理约束深度学习框架（PCG-DL），用于解决工业场景中滚动轴承故障诊断的以下挑战：
- **零样本诊断**：无需真实故障样本即可检测未知故障类型
- **多物理场融合**：融合振动信号与热力学特征
- **物理约束引导**：通过改进的物理模型约束神经网络训练

## 🚀 核心功能

### 主要模块
| 文件 | 功能描述 |
|------|----------|
| `PCG_NEW.py` | 主程序入口，包含完整训练流程 |
| `phynet.py` | 物理信息神经网络架构 |
| `improved_physmodel.py` | 改进的轴承物理动力学模型 |
| `data_set.py` | 多源数据加载与预处理 |

### 创新特性
```python
# 物理约束损失函数示例
def physical_constraint_loss(output, physics_pred):
    # 基于改进物理模型的约束
    return torch.mean((output[:,:3] - physics_pred)**2 + 0.1*output[:,3:]**2)
