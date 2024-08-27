import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import distance
import time
import matplotlib.pyplot as plt
import scipy.stats as stats

# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim=63, hidden_dim1=256, hidden_dim2=512, hidden_dim3=256, latent_dim=5):
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
        std = torch.clamp(std, max=1.0)  # 限制标准差的最大值为1.0
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

# 加载三个不同潜在空间的VAE模型
vae_model_5 = VAE(latent_dim=5)
vae_model_5.load_state_dict(torch.load('vae_model_5.pth', map_location=torch.device('cpu')))
vae_model_5.eval()

vae_model_10 = VAE(latent_dim=10)
vae_model_10.load_state_dict(torch.load('vae_model_10.pth', map_location=torch.device('cpu')))
vae_model_10.eval()

vae_model_15 = VAE(latent_dim=15)
vae_model_15.load_state_dict(torch.load('vae_model_15.pth', map_location=torch.device('cpu')))
vae_model_15.eval()

def encode_samples(vae_models, samples):
    encoded_samples = []
    for model in vae_models:
        with torch.no_grad():
            mu, logvar = model.encode(torch.tensor(samples, dtype=torch.float32))
            z = model.reparameterize(mu, logvar)
        encoded_samples.append(z.numpy())
    return encoded_samples

def mean_distance(d, n_samples=1000):
    n_zero = np.zeros(d)
    samples = stats.multivariate_normal(mean=n_zero).rvs(n_samples)
    d1 = np.random.randint(0, len(samples)-1, len(samples))
    d2 = np.random.randint(0, len(samples)-1, len(samples))
    d_len = np.linalg.norm(samples[d1,:]-samples[d2,:], ord=2, axis=1)
    return np.mean(d_len)

# 生成 63 维向量样本
n_samples = 1000
d = 63
n_zero = np.zeros(d)
samples = stats.multivariate_normal(mean=n_zero, cov=np.eye(d)).rvs(n_samples)

# 使用三个VAE模型进行降维
vae_models = [vae_model_5, vae_model_10, vae_model_15]
latent_vectors = encode_samples(vae_models, samples)

# 打印潜在向量范围以进行调试
for i, latent_vector in enumerate(latent_vectors):
    print(f'Latent vectors for model with latent dimension {vae_models[i].fc4_mu.out_features}:')
    print('Mean:', np.mean(latent_vector))
    print('Standard deviation:', np.std(latent_vector))
    print('Min:', np.min(latent_vector))
    print('Max:', np.max(latent_vector))

# 计算降维后向量的平均距离
mean_distances = [mean_distance(vae_models[i].fc4_mu.out_features, n_samples=len(latent_vectors[i])) for i in range(3)]
print('mean_distance_vae:',mean_distances)
# 理论均值距离
ds = [5, 10, 15]
theoretical_distances = [np.sqrt(2 * d) for d in ds]

# 可视化
fig, ax = plt.subplots()
ax.plot(ds, mean_distances, label="Experimental")
ax.plot(ds, theoretical_distances, label="Theoretical", linestyle='--')
ax.legend()
ax.set_xlabel("Dimensions")
ax.set_ylabel("Mean distance")
ax.set_title("Average distance of latent vectors as a function of dimension")
plt.show()
