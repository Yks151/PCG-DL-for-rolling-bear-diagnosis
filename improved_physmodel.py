# #################以加速度作为输出的物理模型仿真
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from data_set import MyDataset
# from torch.utils.data import DataLoader
#
# # 定义真实的物理模型
# class PhysicalModel:
#     def __init__(self, input_size, temperature=25.0, lubrication=0.5):
#         # 模型参数
#         self.mu_0 = 1.0
#         self.c_mu = 0.01
#         self.kb = 1.0
#         self.cb = 0.1
#         self.cl = 0.01
#         self.m_b = 1.0
#
#         # 考虑温度和润滑状况对物理参数的影响
#         c_kb = 0.02
#         c_cb = 0.01
#         c_cl = 0.005
#         c_kb_l = 0.1
#         c_cb_l = 0.05
#         c_cl_l = 0.02
#         self.kb = self.kb * (1 + c_kb * temperature) * (1 + c_kb_l * lubrication)
#         self.cb = self.cb * (1 + c_cb * temperature) * (1 + c_cb_l * lubrication)
#         self.cl = self.cl * (1 + c_cl * temperature) * (1 + c_cl_l * lubrication)
#
#     def simulate(self, input_data):
#         velocity = np.gradient(input_data)
#
#         # 考虑润滑状况对滚动体的阻尼的影响
#         eta = lubrication
#
#         acceleration = np.zeros_like(velocity)
#         for i in range(1, len(velocity)):
#             kb_with_time = self.kb * (1 + 0.02 * i)  # 添加时间依赖的系数
#             acceleration[i] = (velocity[i] - kb_with_time * velocity[i - 1] - (self.cb + self.cl * velocity[i - 1])) / self.m_b
#
#         output = acceleration
#         return output
#
# # Load data
# batch_size = 32
#
# train_path = r'E:\Transformer_BearingFaultDiagnosis-master\Transformer_BearingFaultDiagnosis-master\data\train\train.csv'
# val_path = r'E:\Transformer_BearingFaultDiagnosis-master\Transformer_BearingFaultDiagnosis-master\data\val\val.csv'
#
# train_dataset = MyDataset(train_path, 'fd')
# val_dataset = MyDataset(val_path, 'fd')
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#
# # Assuming input_data_shape is the shape of the input data in your dataset
# input_data_shape = train_dataset[0]['data'].shape
# input_size = input_data_shape[0]
# output_size = 1
#
# # Define temperature and lubrication
# temperature = 25.0
# lubrication = 0.5
#
# # Create an instance of the PhysicalModel class with appropriate parameters
# physical_model = PhysicalModel(input_size, temperature, lubrication)
#
# # Simulate the physical model using your dataset
# all_output_data = []
# all_input_data = []
# with torch.no_grad():
#     for batch in train_loader:
#         data = batch['data'].numpy()
#         output_data = physical_model.simulate(data)
#         all_output_data.extend(output_data)
#         all_input_data.extend(data)
#
#     # Convert the lists to numpy arrays
#     all_output_data = np.array(all_output_data)
#     all_input_data = np.array(all_input_data)
#
#     # Plot the simulated output data and real output data
#     plt.plot(all_output_data.flatten(), label='Simulated Output Data')
#     plt.plot(all_input_data.flatten(), label='Real Output Data')
#     plt.xlabel('Time')
#     plt.ylabel('Acceleration')
#     plt.legend()
#     plt.show()

##########下面是速度作为输出的物理模型仿真
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from data_set import MyDataset
from torch.utils.data import DataLoader

# 定义真实的物理模型
class PhysicalModel:
    def __init__(self, input_size, temperature=25.0, lubrication=0.5):
        # 模型参数
        self.mu_0 = 1.0
        self.c_mu = 0.01
        self.kb = 1.0
        self.cb = 0.1
        self.cl = 0.01
        self.m_b = 1.0

        # 考虑温度和润滑状况对物理参数的影响
        c_kb = 0.02
        c_cb = 0.01
        c_cl = 0.005
        c_kb_l = 0.1
        c_cb_l = 0.05
        c_cl_l = 0.02
        self.kb = self.kb * (1 + c_kb * temperature) * (1 + c_kb_l * lubrication)
        self.cb = self.cb * (1 + c_cb * temperature) * (1 + c_cb_l * lubrication)
        self.cl = self.cl * (1 + c_cl * temperature) * (1 + c_cl_l * lubrication)

    def simulate(self, input_data):
        # 转换为振动速度
        velocity = np.gradient(input_data)

        # 考虑润滑状况对滚动体的阻尼的影响
        eta = lubrication

        acceleration = np.zeros_like(velocity)
        for i in range(1, len(velocity)):
            kb_with_time = self.kb * (1 + 0.02 * i)  # 添加时间依赖的系数
            acceleration[i] = (velocity[i] - kb_with_time * velocity[i - 1] - (self.cb + self.cl * velocity[i - 1])) / self.m_b

        output = velocity
        return output

# Load data
batch_size = 32

train_path = r'E:\Transformer_BearingFaultDiagnosis-master\Transformer_BearingFaultDiagnosis-master\data\train\train.csv'
val_path = r'E:\Transformer_BearingFaultDiagnosis-master\Transformer_BearingFaultDiagnosis-master\data\val\val.csv'

train_dataset = MyDataset(train_path, 'fd')
val_dataset = MyDataset(val_path, 'fd')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Assuming input_data_shape is the shape of the input data in your dataset
input_data_shape = train_dataset[0]['data'].shape
input_size = input_data_shape[0]
output_size = 1

# Define temperature and lubrication
temperature = 25.0
lubrication = 0.5

# Create an instance of the PhysicalModel class with appropriate parameters
physical_model = PhysicalModel(input_size, temperature, lubrication)

# Simulate the physical model using
all_output_data = []
all_input_data = []

with torch.no_grad():
    for batch in train_loader:
        data = batch['data'].numpy()
        output_data = physical_model.simulate(data)
        all_output_data.extend(output_data)
        all_input_data.extend(data)

# Convert the lists to numpy arrays
all_output_data = np.array(all_output_data)
all_input_data = np.array(all_input_data)

# Plot the simulated output data and real input data
plt.plot(all_output_data.flatten(), label='Simulated Output Data')
plt.plot(all_input_data.flatten(), label='Real Input Data')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()
plt.show()
