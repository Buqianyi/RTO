import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import distance
import time
import matplotlib.pyplot as plt

# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim=63, hidden_dim1=256, hidden_dim2=512, hidden_dim3=256, latent_dim=10):
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

def load_vae_model(filepath, latent_dim):
    vae_model = VAE(latent_dim=latent_dim)
    vae_model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    vae_model.eval()
    return vae_model

def encode_vector(vae_model, vector):
    vector = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mu, logvar = vae_model.encode(vector)
        z = vae_model.reparameterize(mu, logvar)
        reconstructed_vector = vae_model.decode(z)
    return z.squeeze(0).numpy(), reconstructed_vector.squeeze(0).numpy()

def calculate_reconstruction_error(original_vector, reconstructed_vector):
    return np.mean((original_vector - reconstructed_vector) ** 2)

# RWO 类定义
class RWO(object):
    def __init__(self, d, threshold=0.45, bag=None, metric='euclidean'):
        if bag is not None and "data" in bag and len(bag["data"]) > 0:
            self.bag = np.array(bag["data"])
        else:
            self.bag = None
        self.output = bag
        self.d = d
        self.metric = metric
        self.threshold = threshold
        self.n_vecs = 0

    def update(self, vector, t):
        min_distance = float('inf')
        if self.bag is None:
            self.bag = np.array(vector[None, :])
            self.n_vecs = 1
            if self.output is not None:
                self.output["data"].append(vector.tolist())  # Convert to list
                self.output["time"].append(t)
            return True, 0
        ds = distance.cdist(self.bag, vector[None, :], self.metric)
        min_distance = np.min(ds)
        if min_distance > self.threshold:
            self.bag = np.concatenate((self.bag, vector[None, :]), axis=0)
            self.n_vecs = len(self.bag)
            if self.output is not None:
                self.output["data"].append(vector.tolist())  # Convert to list
                self.output["time"].append(t)
            return True, min_distance
        return False, min_distance

# 初始化三个 VAE 模型和 RWO 对象
vae_model_5 = load_vae_model('vae_model_5.pth', latent_dim=5)
vae_model_10 = load_vae_model('vae_model_10.pth', latent_dim=10)
vae_model_15 = load_vae_model('vae_model_15.pth', latent_dim=15)

rwo_5 = RWO(d=5, threshold=1.35, bag={"data": [], "time": []})
rwo_10 = RWO(d=10, threshold=2.95, bag={"data": [], "time": []})
rwo_15 = RWO(d=15, threshold=4.35, bag={"data": [], "time": []})

# 初始化 MediaPipe 手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_count = 0
total_vectors = 0

novel_vectors_5 = 0
novel_vectors_10 = 0
novel_vectors_15 = 0

novel_vectors_over_time_5 = []
novel_vectors_over_time_10 = []
novel_vectors_over_time_15 = []
timestamps = []

min_distances_5 = []
min_distances_10 = []
min_distances_15 = []

start_time = time.time()  # 记录程序开始运行的时间

positive_feedback = ["Perfect!", "Excellent!", "Great!", "Awesome!"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 翻转图像
    frame = cv2.flip(frame, 1)
    
    # 转换颜色空间
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 处理图像并获取结果
    results = hands.process(image)
    
    # 绘制手部关键点并提取63维向量
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 提取63维向量
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            assert landmarks.shape[0] == 63, "The landmark vector should be 63-dimensional"
            
            # 使用 VAE 模型进行降维和重构
            latent_vector_5, _ = encode_vector(vae_model_5, landmarks)
            latent_vector_10, _ = encode_vector(vae_model_10, landmarks)
            latent_vector_15, _ = encode_vector(vae_model_15, landmarks)
            
            # 使用 RWO 类处理降维后的向量
            updated_5, min_distance_5 = rwo_5.update(latent_vector_5, frame_count)
            updated_10, min_distance_10 = rwo_10.update(latent_vector_10, frame_count)
            updated_15, min_distance_15 = rwo_15.update(latent_vector_15, frame_count)
            
            # 增加向量计数
            total_vectors += 1
            feedback_displayed = False  # 记录是否显示反馈

            if updated_5 or updated_10 or updated_15:
                feedback = np.random.choice(positive_feedback)
                cv2.putText(frame, feedback, (frame.shape[1] // 2 - 50, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                feedback_displayed = True

            if updated_5:
                novel_vectors_5 += 1
            if updated_10:
                novel_vectors_10 += 1
            if updated_15:
                novel_vectors_15 += 1
            
            # 记录最小距离
            min_distances_5.append(min_distance_5)
            min_distances_10.append(min_distance_10)
            min_distances_15.append(min_distance_15)
    
    frame_count += 1
    
    # 记录当前 novel_vectors 数量和时间戳
    novel_vectors_over_time_5.append(novel_vectors_5)
    novel_vectors_over_time_10.append(novel_vectors_10)
    novel_vectors_over_time_15.append(novel_vectors_15)
    timestamps.append(time.time() - start_time)
    
    # 显示总向量数、新颖向量数、最小距离和程序运行时间
    cv2.putText(frame, f'Total Vectors: {total_vectors}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Novel Vectors 5: {novel_vectors_5}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Min Distance 5: {min_distance_5:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Novel Vectors 10: {novel_vectors_10}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Min Distance 10: {min_distance_10:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Novel Vectors 15: {novel_vectors_15}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Min Distance 15: {min_distance_15:.2f}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    elapsed_time = time.time() - start_time
    elapsed_time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    cv2.putText(frame, f'Time: {elapsed_time_str}', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 显示处理后的帧
    cv2.imshow('Hand Tracking', frame)

    # 定时 3 分钟，程序运行超过 3 分钟自动关闭
    if elapsed_time >= 180:
        print("Program has been running for 3 minutes. Exiting...")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

# 绘制 novel_vectors 随时间变化的图像
plt.figure(figsize=(12, 6))
plt.plot(timestamps, novel_vectors_over_time_5, label='VAE Model 5 (Threshold=1.35)', color='b')
plt.plot(timestamps, novel_vectors_over_time_10, label='VAE Model 10 (Threshold=2.95)', color='g')
plt.plot(timestamps, novel_vectors_over_time_15, label='VAE Model 15 (Threshold=4.35)', color='r')
plt.xlabel('Time (seconds)')
plt.ylabel('Novel Vectors')
plt.title('Novel Vectors Over Time for Different VAE Models')
plt.legend()
plt.grid(True)
plt.show()

# 计算并打印最小距离的平均值
average_min_distance_5 = np.mean(min_distances_5)
average_min_distance_10 = np.mean(min_distances_10)
average_min_distance_15 = np.mean(min_distances_15)

print(f'Average minimum distance (Latent Dim 5): {average_min_distance_5}')
print(f'Average minimum distance (Latent Dim 10): {average_min_distance_10}')
print(f'Average minimum distance (Latent Dim 15): {average_min_distance_15}')
