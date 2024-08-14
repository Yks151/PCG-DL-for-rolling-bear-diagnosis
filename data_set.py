import torch
from torch.utils.data import Dataset
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, path, task, transform=None, target_transform=None, loader=None):
        # 读取csv文件
        dataFrame = pd.read_csv(path, encoding='utf-8', header=None)
        martrix = dataFrame.values
        datas = martrix[:, :-3]
        # 在 MyDataset 的构造函数中，task 是一个字符串类型的参数，它用于指定数据集的任务类型，可以是 'fd'、'loc'、'dia' 或 'multi'，分别代表故障诊断、故障定位、故障直径预测和多任务学习。
        if task == 'fd':
            labels = martrix[:, -3]
        elif task == 'loc':
            labels = martrix[:, -2]
        elif task == 'dia':
            labels = martrix[:, -1]
        elif task == 'multi':
            labels = martrix[:, -3:]

        self.datas = torch.Tensor(datas)
        self.labels = torch.Tensor(labels).long()  # 将标签转换为整数类型
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        # 获取数据和标签
        # data = self.datas[index]
        data_tensor = self.datas[index]
        # label = self.labels[index]
        label_tensor = self.labels[index]
        # 处理数据
        if self.transform is not None:
            data = self.transform(data)
        # return data, label
        return {
            'data': data_tensor,  # 替换为你的数据张量
            'label': label_tensor  # 替换为你的标签张量
        }

    def __len__(self):
        return len(self.datas)

