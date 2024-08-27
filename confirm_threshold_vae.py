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

# 加载训练好的模型
vae_model = VAE()
vae_model.load_state_dict(torch.load('vae_model_5.pth', map_location=torch.device('cpu')))
vae_model.eval()

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

# 初始化 RWO 对象
rwo = RWO(d=5, threshold=1.45, bag={"data": [], "time": []})

# 存储所有63维向量的列表
all_vectors = []

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
show_feedback = False
feedback_timer = 0
feedback_messages = ["Perfect!", "Excellent!", "Great!", "Awesome!"]
total_vectors = 0
novel_vectors = 0
min_distance = 0
min_dist_before_dim_reduction = 0  # 初始化变量
novel_vectors_over_time = []  # 记录 novel_vectors 随时间变化的列表
timestamps = []  # 记录时间戳的列表

# 初始化最小距离列表
min_distances = []

start_time = time.time()  # 记录程序开始运行的时间

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
            
            # 存储新的63维向量
            all_vectors.append(landmarks)
            
            # 打印降维前的向量数据
            print(f'Original 63-dimensional vector: {landmarks}')
            
            # 计算降维前的最小距离
            if len(all_vectors) > 1:
                min_dist_before_dim_reduction = np.min(distance.cdist(all_vectors[:-1], [landmarks], 'euclidean'))
            
            # 使用 VAE 模型进行降维和重构
            latent_vector, reconstructed_vector = encode_vector(vae_model, landmarks)
            
            # 计算重构误差
            reconstruction_error = calculate_reconstruction_error(landmarks, reconstructed_vector)
            
            # 使用 RWO 类处理降维后的向量
            updated, min_distance = rwo.update(latent_vector, frame_count)
            
            # 增加向量计数
            total_vectors += 1
            if updated:
                novel_vectors += 1
                show_feedback = True
                feedback_timer = 30  # 设定反馈信息显示时间（帧数）
                feedback_message = np.random.choice(feedback_messages)
            
            # 记录最小距离
            min_distances.append(min_distance)
            
            # 打印相关信息
            print(f'Latent vector: {latent_vector}')
            print(f'Reconstruction error: {reconstruction_error}')
            print(f'Updated: {updated}')
            print(f'Min distance: {min_distance}')
            print(f'Min distance before dimensionality reduction: {min_dist_before_dim_reduction}')
    
    frame_count += 1
    
    # 记录当前 novel_vectors 数量和时间戳
    novel_vectors_over_time.append(novel_vectors)
    timestamps.append(time.time() - start_time)
    
    # 显示反馈信息
    if show_feedback:
        text_size = cv2.getTextSize(feedback_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, feedback_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        feedback_timer -= 1
        if feedback_timer <= 0:
            show_feedback = False
    
    # 显示总向量数、新颖向量数、最小距离和程序运行时间
    cv2.putText(frame, f'Total Vectors: {total_vectors}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Novel Vectors: {novel_vectors}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Min Distance: {min_distance:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Min Dist (Pre-Reduction): {min_dist_before_dim_reduction:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elapsed_time = time.time() - start_time
    elapsed_time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    cv2.putText(frame, f'Time: {elapsed_time_str}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 显示处理后的帧
    cv2.imshow('Hand Tracking', frame)

    # 定时 10 分钟，程序运行超过 10 分钟自动关闭
    if elapsed_time >= 600:
        print("Program has been running for 3 minutes. Exiting...")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

# 绘制 novel_vectors 随时间变化的图像
plt.figure(figsize=(10, 5))
plt.plot(timestamps, novel_vectors_over_time, label='Novel Vectors under vae_model_5')
plt.xlabel('Time (seconds)')
plt.ylabel('Novel Vectors')
plt.title('Novel Vectors Over Time under threshold 1.35')
plt.legend()
plt.grid(True)
plt.show()

# 计算并打印最小距离的平均值
average_min_distance = np.mean(min_distances)
print(f'Average minimum distance: {average_min_distance}')
