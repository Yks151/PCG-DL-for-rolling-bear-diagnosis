import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_set import MyDataset
import torch.nn.functional as F

# Define the physical model
class PhysicalModel(nn.Module):
    def __init__(self):
        super(PhysicalModel, self).__init__()
        # Define the learnable parameters
        self.c_mu = nn.Parameter(torch.tensor(0.01))
        self.c_kb = nn.Parameter(torch.tensor(0.02))
        self.c_cb = nn.Parameter(torch.tensor(0.01))
        self.c_cl = nn.Parameter(torch.tensor(0.005))

    def forward(self, input_data):
        feature1 = input_data[0]
        feature2 = input_data[1]

        kb_with_time = self.c_kb * (1 + 0.02 * feature1)
        output = self.c_mu * feature1 + kb_with_time * feature2 + self.c_cb + self.c_cl * feature2

        return output

# Define the CNN-LSTM-Attention model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # Reshape input to (batch_size, seq_len, input_size)
        x, _ = self.lstm(x)
        attention_weights = F.softmax(self.attention(x), dim=1)
        x = torch.bmm(attention_weights.permute(0, 2, 1), x).squeeze(1)
        x = self.fc(x)
        return x


# Set random seed for reproducibility
torch.manual_seed(42)

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
input_size = input_data_shape[0] + 4  # Add 4 for the 4 learnable parameters
hidden_size = 64
output_size = 10

# Create instances of the physical model and the CNN-LSTM-Attention model
physical_model = PhysicalModel()
model = Model(input_size, hidden_size, output_size)

# Define the optimizer and loss function
optimizer = optim.Adam(list(model.parameters()) + list(physical_model.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    physical_model.train()

    for batch in train_loader:
        input_data = batch['data']
        target_label = batch['label']

        physical_inputs = input_data[:, :2]
        learnable_parameters = torch.stack([
            physical_model.c_mu,
            physical_model.c_kb,
            physical_model.c_cb,
            physical_model.c_cl
        ]).unsqueeze(0).repeat(batch_size, 1)

        # Make sure the dimensions match for concatenation
        physical_inputs = physical_inputs.unsqueeze(1)  # Add an extra dimension
        learnable_parameters = learnable_parameters.unsqueeze(1)  # Add an extra dimension
        input_features = torch.cat((physical_inputs, learnable_parameters), dim=2)

        optimizer.zero_grad()
        output = model(input_features)
        loss = criterion(output, target_label)
        loss.backward()
        optimizer.step()

    model.eval()
    physical_model.eval()

    with torch.no_grad():
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in val_loader:
            input_data = batch['data']
            target_label = batch['label']

            physical_inputs = input_data[:, :2]
            learnable_parameters = torch.stack([
                physical_model.c_mu,
                physical_model.c_kb,
                physical_model.c_cb,
                physical_model.c_cl
            ]).unsqueeze(0).repeat(batch_size, 1)
            input_features = torch.cat((physical_inputs, learnable_parameters.squeeze(1)), dim=1)

            output = model(input_features.unsqueeze(1))
            loss = criterion(output, target_label)
            total_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += target_label.size(0)
            correct += (predicted == target_label).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}: Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
