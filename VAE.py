import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 加载数据函数
def load_data_from_folder(folder_path, key_to_split):
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                data = json.load(f)
                data_list.extend(data[key_to_split])  # 使用实际键名
    return np.array(data_list, dtype=np.float32)

# 加载数据
key_to_split = 'data'  # 替换为实际键名

train_data = load_data_from_folder('output_data/train/20240709_133000', key_to_split)  # 替换 'output_data/train_data' 为实际训练集文件夹路径
test_data = load_data_from_folder('output_data/test/20240709_133000', key_to_split)    # 替换 'output_data/test_data' 为实际测试集文件夹路径
val_data = load_data_from_folder('output_data/validation/20240709_133000', key_to_split)    # 替换 'output_data/val_data' 为实际验证集文件夹路径

# 创建数据集
train_dataset = TensorDataset(torch.from_numpy(train_data))
test_dataset = TensorDataset(torch.from_numpy(test_data))
val_dataset = TensorDataset(torch.from_numpy(val_data))

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4_mu = nn.Linear(hidden_dim3, latent_dim)
        self.fc4_logvar = nn.Linear(hidden_dim3, latent_dim)
        # Decoder
        self.fc5 = nn.Linear(latent_dim, hidden_dim3)
        self.fc6 = nn.Linear(hidden_dim3, hidden_dim2)
        self.fc7 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc8 = nn.Linear(hidden_dim1, input_dim)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim1)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim2)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim3)

    def encode(self, x):
        h = torch.relu(self.batch_norm1(self.fc1(x)))
        h = torch.relu(self.batch_norm2(self.fc2(h)))
        h = torch.relu(self.batch_norm3(self.fc3(h)))
        return self.fc4_mu(h), self.fc4_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc5(z))
        h = torch.relu(self.fc6(h))
        h = torch.relu(self.fc7(h))
        return torch.sigmoid(self.fc8(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train_vae(train_loader, val_loader, model, optimizer, scheduler, epochs=20, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    train_losses = []
    val_losses = []
    train_reconstruction_errors = []
    val_reconstruction_errors = []
    
    model.to(device)  # 将模型移到指定设备
    model.train()  # 设置模型为训练模式

    for epoch in range(epochs):
        train_loss = 0
        epoch_reconstruction_error = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data[0].to(device)  # 将数据移到指定设备
            optimizer.zero_grad()  # 清除梯度缓存
            recon_batch, mu, logvar = model(data)  # 前向传播
            loss = loss_function(recon_batch, data, mu, logvar)  # 计算损失
            loss.backward()  # 反向传播
            train_loss += loss.item()  # 累计损失
            epoch_reconstruction_error += ((recon_batch - data) ** 2).sum().item()  # 累计重构误差
            optimizer.step()  # 优化步骤
        
        scheduler.step()  # 更新学习率
        train_loss /= len(train_loader.dataset)  # 平均训练损失
        train_losses.append(train_loss)  # 记录训练损失
        train_reconstruction_errors.append(epoch_reconstruction_error / len(train_loader.dataset))  # 记录重构误差

        # 验证阶段
        val_loss, val_reconstruction_error = evaluate_vae(val_loader, model, device)  # 调用验证函数
        val_losses.append(val_loss)  # 记录验证损失
        val_reconstruction_errors.append(val_reconstruction_error)  # 记录验证重构误差

        # 打印当前epoch结果
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Recon Error: {train_reconstruction_errors[-1]:.4f}, Val Recon Error: {val_reconstruction_errors[-1]:.4f}")
    
    return train_losses, val_losses, train_reconstruction_errors, val_reconstruction_errors


def evaluate_vae(data_loader, model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    total_loss = 0
    total_reconstruction_error = 0
    reconstruction_errors = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data[0].to(device)  # 将数据移动到指定设备
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            total_loss += loss.item()
            reconstruction_error = ((recon_batch - data) ** 2).sum().item()
            total_reconstruction_error += reconstruction_error
            reconstruction_errors.extend(((recon_batch - data) ** 2).sum(dim=1).cpu().numpy())
    
    avg_loss = total_loss / len(data_loader.dataset)
    avg_reconstruction_error = total_reconstruction_error / len(data_loader.dataset)
    
    return avg_loss, avg_reconstruction_error

def extract_latent_features(data_loader, model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.to(device)
    model.eval()
    latent_features = []
    reconstruction_errors = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data[0].to(device)  # 将数据移动到指定设备
            recon_batch, mu, logvar = model(data)
            z = model.reparameterize(mu, logvar)
            latent_features.append(z.cpu().numpy())
            reconstruction_errors.extend(((recon_batch - data) ** 2).sum(dim=1).cpu().numpy())
    
    return np.vstack(latent_features), reconstruction_errors


# 参数设置
input_dim = 63  # 输入维度
hidden_dim1 = 256  # 第一个隐藏层维度
hidden_dim2 = 512  # 第二个隐藏层维度
hidden_dim3 = 256  # 第三个隐藏层维度
latent_dim = 10  # 潜在空间维度
learning_rate = 1e-5  # 学习率
epochs = 20  # 训练轮数

# 模型、优化器和学习率调度器
model = VAE(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, latent_dim)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # 使用AdamW优化器，并添加权重衰减
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每5个epoch学习率降低一半


# 训练模型
train_losses, val_losses, train_reconstruction_errors, val_reconstruction_errors = train_vae(train_loader, val_loader, model, optimizer, scheduler, epochs)

# 调试信息：打印损失值
print("Train Losses:", train_losses)
print("Validation Losses:", val_losses)
print("Train Reconstruction Errors:", train_reconstruction_errors)
print("Validation Reconstruction Errors:", val_reconstruction_errors)

# 保存模型
torch.save(model.state_dict(), 'vae_model_10.pth')  # 保存模型的路径和文件名，可根据实际情况调整

# 评估模型
test_loss, test_reconstruction_error = evaluate_vae(test_loader, model)
print(f"Test Loss: {test_loss}")
print(f"Test Reconstruction Error: {test_reconstruction_error}")

# 提取潜在特征
train_latent_features, train_reconstruction_errors_all = extract_latent_features(train_loader, model)
test_latent_features, test_reconstruction_errors_all = extract_latent_features(test_loader, model)
val_latent_features, val_reconstruction_errors_all = extract_latent_features(val_loader, model)

print("Train Latent features shape:", train_latent_features.shape)
print("Test Latent features shape:", test_latent_features.shape)
print("Validation Latent features shape:", val_latent_features.shape)

# 可视化训练和验证损失
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# 可视化重构误差随训练轮数变化
plt.figure(figsize=(10, 5))
plt.plot(train_reconstruction_errors, label='Train Reconstruction Error')
plt.plot(val_reconstruction_errors, label='Validation Reconstruction Error')
plt.xlabel('Epochs')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.title('Reconstruction Error over Epochs')
plt.show()

# 可视化潜在空间（前两个维度）
plt.figure(figsize=(10, 5))
plt.scatter(train_latent_features[:, 0], train_latent_features[:, 1], label='Train', alpha=0.5)
plt.scatter(test_latent_features[:, 0], test_latent_features[:, 1], label='Test', alpha=0.5)
plt.scatter(val_latent_features[:, 0], val_latent_features[:, 1], label='Validation', alpha=0.5)
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend()
plt.title('Latent Space Visualization')
plt.show()

# 可视化重构误差分布
plt.figure(figsize=(10, 5))
plt.hist(train_reconstruction_errors_all, bins=50, alpha=0.5, label='Train')
plt.hist(test_reconstruction_errors_all, bins=50, alpha=0.5, label='Test')
plt.hist(val_reconstruction_errors_all, bins=50, alpha=0.5, label='Validation')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.title('Reconstruction Error Distribution')
plt.show()
