import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import PCA

def calculate_mean_distance(samples):
    n_samples = len(samples)
    d1 = np.random.randint(0, n_samples, n_samples)
    d2 = np.random.randint(0, n_samples, n_samples)
    d_len = np.linalg.norm(samples[d1] - samples[d2], axis=1)
    return np.mean(d_len)

# 生成 63 维向量样本
n_samples = 10000  # 增加样本数量
d = 63
n_zero = np.zeros(d)
samples = stats.multivariate_normal(mean=n_zero, cov=np.eye(d)).rvs(n_samples)

# 定义和加载 PCA 模型
pca_model_5 = PCA(n_components=5)
pca_model_10 = PCA(n_components=10)
pca_model_15 = PCA(n_components=15)

# 使用 PCA 模型进行降维
pca_models = [pca_model_5, pca_model_10, pca_model_15]
latent_vectors_pca = [pca.fit_transform(samples) for pca in pca_models]

# 打印潜在向量范围以进行调试
for i, latent_vector in enumerate(latent_vectors_pca):
    print(f'Latent vectors for PCA model with latent dimension {pca_models[i].n_components}:')
    print('Mean:', np.mean(latent_vector))
    print('Standard deviation:', np.std(latent_vector))
    print('Min:', np.min(latent_vector))
    print('Max:', np.max(latent_vector))

# 计算降维后向量的平均距离
mean_distances_pca = [calculate_mean_distance(latent_vectors_pca[i]) for i in range(3)]
print("Mean distances for PCA:", mean_distances_pca)

# 理论均值距离
ds = [5, 10, 15]
theoretical_distances = [np.sqrt(2 * d) for d in ds]

# 可视化
fig, ax = plt.subplots()
ax.plot(ds, mean_distances_pca, label="PCA Experimental")
ax.plot(ds, theoretical_distances, label="Theoretical", linestyle='--')
ax.legend()
ax.set_xlabel("Dimensions")
ax.set_ylabel("Mean distance")
ax.set_title("Average distance of latent vectors as a function of dimension")
plt.show()
