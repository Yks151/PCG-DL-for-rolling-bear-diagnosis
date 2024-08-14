import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from data_set import MyDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# 创建logs目录
if not os.path.exists("./hybrid_exp_kl"):
    os.makedirs("./hybrid_exp_kl")

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # 定义Transformer的网络结构
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Transformer前向传播逻辑
        x = x.permute(2, 0, 1)  # 调整为(seq_length, batch_size, input_size)
        x = self.transformer_encoder(x)  # 输入Transformer编码器
        x = x.permute(1, 0, 2)  # 将输出维度调整为(batch_size, seq_length, input_size)
        x = self.fc(x[:, -1, :])  # 全连接层，并只保留最后一个时间步的输出
        x = F.relu(x)  # 使用ReLU激活函数
        output = x.squeeze(1)  # 去除第1个维度，得到形状为(batch_size, output_size)的输出

        return output

def physical_model(input, temperature=25.0, lubrication=0.5):
    # ...
    # 定义模型参数
    mu_0 = 1.0
    c_mu = 0.01
    kb = 1.0
    cb = 0.1
    cl = 0.01
    m_b = 1.0
    # ...

    # 考虑温度和润滑状况对物理参数的影响
    c_kb = 0.02
    c_cb = 0.01
    c_cl = 0.005
    c_kb_l = 0.1
    c_cb_l = 0.05
    c_cl_l = 0.02
    kb = kb * (1 + c_kb * temperature) * (1 + c_kb_l * lubrication)
    cb = cb * (1 + c_cb * temperature) * (1 + c_cb_l * lubrication)
    cl = cl * (1 + c_cl * temperature) * (1 + c_cl_l * lubrication)

    # 计算滚动体的摩擦力
    mu = mu_0 * (1 + c_mu * temperature)
    Ff = mu * input

    # 考虑润滑状况对滚动体的阻尼的影响
    eta = lubrication
    acceleration = torch.zeros_like(input)
    velocity = torch.zeros_like(input)

    for i in range(1, len(input)):
        acceleration[i] = (input[i] - kb * input[i - 1] - (cb + eta * cl) * velocity[i - 1]) / m_b
        velocity[i] = velocity[i - 1] + acceleration[i]

    output = velocity

    output = output.squeeze()  # 去除维度为1的维度

    return output
def loss_function(y_pred, y_phys, y_target, y_fault, lambda1, lambda2, lambda3, lambda4):
    # 计算特征学习损失
    feature_loss = torch.mean((y_pred - y_target) ** 2)

    # 计算物理一致性损失
    physics_loss = torch.mean((y_pred - y_phys) ** 2)

    # 计算不确定性建模损失（KL散度）
    kl_div_loss = torch.nn.functional.kl_div(torch.log_softmax(y_pred, dim=0), torch.softmax(y_phys, dim=0), reduction='batchmean')

    # 计算故障状态评估损失
    fault_loss = torch.mean((y_pred - y_fault) ** 2)

    # 组合损失
    total_loss = lambda1 * feature_loss + lambda2 * physics_loss + lambda3 * kl_div_loss + lambda4 * fault_loss
    return total_loss

# Load data
batch_size = 32

train_path = r'E:\Transformer_BearingFaultDiagnosis-master\Transformer_BearingFaultDiagnosis-master\data\train\train.csv'
val_path = r'E:\Transformer_BearingFaultDiagnosis-master\Transformer_BearingFaultDiagnosis-master\data\val\val.csv'

train_dataset = MyDataset(train_path, 'fd')
val_dataset = MyDataset(val_path, 'fd')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

input_data_shape = train_dataset[0]['data'].shape
target_data_shape = train_dataset[0]['label'].shape

input_size = input_data_shape[0]
output_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU加速训练，如果可用

model = TransformerModel(input_size, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
lambda1 = 1.0  # 特征学习损失权重
lambda2 = 1.0  # 物理一致性损失权重
lambda3 = 1.0  # 不确定性建模损失权重
lambda4 = 1.0  # 故障状态评估损失权重

# 训练模型
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
velocity_curves = []  # 累积所有批次的速度曲线
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0

    # 在每个epoch之前创建一个空列表来保存所有时间步的速度曲线
    velocity_curves = []
    for batch_idx, sample in enumerate(tqdm(train_loader)):
        data, label = sample['data'].to(device), sample['label'].to(device)

        optimizer.zero_grad()
        data = data.unsqueeze(1).permute(0, 2, 1)  # 添加和调整维度，将数据移至GPU
        output = model(data)
        y_phys = physical_model(output.to("cpu"))

        # 将当前时间段的速度曲线数据添加到列表中
        velocity_curves.append((output.squeeze().detach().numpy(), y_phys.detach().numpy()))
        # 计算综合输出速度曲线并添加到列表中
        combined_output = (output + y_phys) / 2
        combined_curves.append(combined_output.squeeze().detach().numpy())

        # 计算损失并进行反向传播
        loss = loss_function(output, y_phys, label, label, lambda1, lambda2, lambda3, lambda4)
        loss.backward()
        optimizer.step()

        # 更新总损失和正确预测的数量
        total_loss += loss.item()
        total_correct += (output.argmax(1) == label).sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc = total_correct / len(train_dataset)

    # 在训练之后保存当前epoch的速度曲线
    np.save(f"./hybrid_exp_kl/velocity_curves_epoch_{epoch}.npy", velocity_curves)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    # 在每个epoch之后可以进行验证集上的评估
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_correct = 0

        for sample in val_loader:
            data, label = sample['data'].to(device), sample['label'].to(device)
            data = data.unsqueeze(1).permute(0, 2, 1)
            output = model(data)
            y_phys = physical_model(output.to("cpu"))

            loss = loss_function(output, y_phys, label, label, lambda1, lambda2, lambda3, lambda4)
            val_loss += loss.item()
            val_correct += (output.argmax(1) == label).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_dataset)

        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
# 输出训练和验证结果
print('Epoch [{}/{}], Training Loss: {:.4f}, Training Accuracy: {:.2f}%'.format(
    epoch + 1, num_epochs, total_loss / len(train_loader), total_correct / len(train_dataset) * 100
))
print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'.format(
    epoch + 1, num_epochs, val_loss / len(val_loader), val_correct / len(val_dataset) * 100
))


# 保存模型和训练结果
torch.save(model.state_dict(), './hybrid_exp_kl/phys_transformer_model.pth')
np.savetxt('./hybrid_exp_kl/train_loss.txt', train_loss_list)
np.savetxt('./hybrid_exp_kl/val_loss.txt', val_loss_list)
np.savetxt('./hybrid_exp_kl/train_acc.txt', train_acc_list)
np.savetxt('./hybrid_exp_kl/val_acc.txt', val_acc_list)

# 绘制损失和准确率曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Acc')
plt.plot(val_acc_list, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('./hybrid_exp_kl/loss_acc_curve.png')
plt.show()
    loss = loss_function(output, y_phys, label, label, lambda1, lambda2, lambda3, lambda4)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    _, predicted = output.unsqueeze(1).max(1)
    total_correct += predicted.eq(label).sum().item()

train_loss_list.append(total_loss / len(train_loader))
train_acc_list.append(total_correct / len(train_dataset) * 100)

model.eval()
val_loss = 0
val_correct = 0

with torch.no_grad():
    for sample in tqdm(val_loader):
        data, label = sample['data'].to(device), sample['label'].to(device)
        data = data.unsqueeze(1).permute(0, 2, 1)
        output = model(data)
        y_phys = physical_model(output.to("cpu"))

        loss = loss_function(output, y_phys, label, label, lambda1, lambda2, lambda3, lambda4)
        val_loss += loss.item()

        _, predicted = output.unsqueeze(1).max(1)
        val_correct += predicted.eq(label).sum().item()

val_loss_list.append(val_loss / len(val_loader))
val_acc_list.append(val_correct / len(val_dataset) * 100)

# 输出训练和验证结果
print('Epoch [{}/{}], Training Loss: {:.4f}, Training Accuracy: {:.2f}%'.format(
    epoch + 1, num_epochs, total_loss / len(train_loader), total_correct / len(train_dataset) * 100
))
print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'.format(
    epoch + 1, num_epochs, val_loss / len(val_loader), val_correct / len(val_dataset) * 100
))

# 保存模型和训练结果
torch.save(model.state_dict(), './hybrid_exp_kl/phys_transformer_model.pth')
np.savetxt('./hybrid_exp_kl/train_loss.txt', train_loss_list)
np.savetxt('./hybrid_exp_kl/val_loss.txt', val_loss_list)
np.savetxt('./hybrid_exp_kl/train_acc.txt', train_acc_list)
np.savetxt('./hybrid_exp_kl/val_acc.txt', val_acc_list)

# 绘制损失和准确率曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Acc')
plt.plot(val_acc_list, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('./hybrid_exp_kl/loss_acc_curve.png')
plt.show()