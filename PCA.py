import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import joblib  # 导入joblib库

# Function to load data from JSON file
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data["data"])

# Load training data
train_data_path = "output_data/train/20240709_133000/all_vectors_train_3.json"  # 训练数据文件路径
train_data = load_data(train_data_path)

# Load validation data
val_data_path = "output_data/validation/20240709_133000/all_vectors_validation_3.json"  # 验证数据文件路径
val_data = load_data(val_data_path)

# Standardize the data
scaler = StandardScaler()
train_data_standardized = scaler.fit_transform(train_data)
val_data_standardized = scaler.transform(val_data)  # 使用同一个scaler对验证数据进行标准化

# Save the StandardScaler model to a .pkl file
scaler_path = "scaler_model.pkl"
joblib.dump(scaler, scaler_path)
print(f"StandardScaler model saved to {scaler_path}")

# Perform PCA
n_components = 15
pca = PCA(n_components=n_components)
principal_components_train = pca.fit_transform(train_data_standardized)

# Save the PCA model to a .pkl file
model_path = "pca_model_15.pkl"
joblib.dump(pca, model_path)
print(f"PCA model saved to {model_path}")

# Reconstruct the training data
reconstructed_train_data = pca.inverse_transform(principal_components_train)

# Calculate training reconstruction error
train_reconstruction_error = mean_squared_error(train_data_standardized, reconstructed_train_data)
print(f"Training Reconstruction Error: {train_reconstruction_error}")

# Apply PCA transformation to validation data
principal_components_val = pca.transform(val_data_standardized)

# Reconstruct the validation data
reconstructed_val_data = pca.inverse_transform(principal_components_val)

# Calculate validation reconstruction error
val_reconstruction_error = mean_squared_error(val_data_standardized, reconstructed_val_data)
print(f"Validation Reconstruction Error: {val_reconstruction_error}")

# Visualize the explained variance ratio and cumulative explained variance (based on training set)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print(f"Explained variance ratio: {explained_variance_ratio}")
print(f"Cumulative explained variance: {cumulative_explained_variance}")

plt.figure(figsize=(10, 6))
plt.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.6, align='center', label='Individual explained variance')
plt.step(range(1, n_components + 1), cumulative_explained_variance, where='mid', label='Cumulative explained variance')
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.title('Explained Variance Ratio of Principal Components (Training Set)')
plt.legend(loc='best')
plt.show()

# Visualize the projection error
projection_error = np.mean(np.linalg.norm(train_data_standardized - reconstructed_train_data, axis=1))
print(f"Projection Error: {projection_error}")

residuals = train_data_standardized - reconstructed_train_data
residuals_mean = np.mean(residuals)
residuals_std = np.std(residuals)
print(f"Residuals Mean: {residuals_mean}")
print(f"Residuals Std Dev: {residuals_std}")

plt.figure(figsize=(10, 6))
plt.hist(residuals.flatten(), bins=50, alpha=0.7)
plt.xlabel('Reconstruction Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Reconstruction Residuals')
plt.show()

# Visualize the cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_components + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance as a Function of the Number of Principal Components')
plt.grid(True)
plt.show()
