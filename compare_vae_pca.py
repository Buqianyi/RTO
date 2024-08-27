import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import distance
import time
import matplotlib.pyplot as plt
import joblib

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

# 加载训练好的标准化模型和 PCA 模型
scaler = joblib.load('scaler_model.pkl')
pca_models = [
    {'model': joblib.load('pca_model_5.pkl'), 'threshold': 3.65, 'd': 5, 'novel_vectors': [], 'timestamps': []},
    {'model': joblib.load('pca_model_10.pkl'), 'threshold': 4.95, 'd': 10, 'novel_vectors': [], 'timestamps': []},
    {'model': joblib.load('pca_model_15.pkl'), 'threshold': 5.45, 'd': 15, 'novel_vectors': [], 'timestamps': []}
]

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
rwo_objects = [RWO(d=pca['d'], threshold=pca['threshold'], bag={"data": [], "time": []}) for pca in pca_models]

# 存储所有63维向量的列表
all_vectors = []

# 初始化 MediaPipe 手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 初始化三个 VAE 模型和 RWO 对象
vae_model_5 = load_vae_model('vae_model_5.pth', latent_dim=5)
vae_model_10 = load_vae_model('vae_model_10.pth', latent_dim=10)
vae_model_15 = load_vae_model('vae_model_15.pth', latent_dim=15)

rwo_5 = RWO(d=5, threshold=1.35, bag={"data": [], "time": []})
rwo_10 = RWO(d=10, threshold=2.95, bag={"data": [], "time": []})
rwo_15 = RWO(d=15, threshold=4.35, bag={"data": [], "time": []})

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_count = 0
total_vectors = 0
novel_vectors = [0, 0, 0, 0, 0, 0]
min_distance = [0, 0, 0, 0, 0, 0]
min_dist_before_dim_reduction = 0  # 初始化变量
start_time = time.time()  # 记录程序开始运行的时间

# 用于存储每10帧的信息
frame_data = []
min_distances = [[], [], [], [], [], []]  # 记录每帧的最小距离

# 存储每个 VAE 模型的新颖向量数和时间戳
novel_vectors_over_time_vae_5 = []
novel_vectors_over_time_vae_10 = []
novel_vectors_over_time_vae_15 = []
timestamps_vae = []

# 存储 VAE 模型的最小距离
min_distances_vae_5 = []
min_distances_vae_10 = []
min_distances_vae_15 = []

# 定义反馈消息列表
feedback_messages = ["Perfect", "Great", "Awesome", "Excellent"]
feedback_index = 0  # 反馈消息索引
feedback_display_time = 2  # 反馈消息显示时间（秒）
last_feedback_time = 0  # 上一次反馈消息显示的时间

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
            
            # 计算降维前的最小距离
            if len(all_vectors) > 1:
                min_dist_before_dim_reduction = np.min(distance.cdist(all_vectors[:-1], [landmarks], 'euclidean'))
            
            # 使用标准化模型对数据进行标准化
            standardized_vector = scaler.transform(landmarks.reshape(1, -1)).squeeze()
            
            # 对每个 PCA 模型进行降维和 RWO 更新
            for i, pca in enumerate(pca_models):
                latent_vector = pca['model'].transform(standardized_vector.reshape(1, -1)).squeeze()
                updated, min_distance[i] = rwo_objects[i].update(latent_vector, frame_count)
                
                # 增加向量计数
                total_vectors += 1
                if updated:
                    novel_vectors[i] += 1
                    feedback_index = (feedback_index + 1) % len(feedback_messages)  # 更新反馈消息索引
                    last_feedback_time = time.time()  # 更新上一次反馈消息显示的时间
                
                # 记录最小距离
                min_distances[i].append(min_distance[i])
                
                # 记录当前 novel_vectors 数量和时间戳
                pca['novel_vectors'].append(novel_vectors[i])
                pca['timestamps'].append(time.time() - start_time)
                
                # 打印相关信息（仅每10帧打印一次）
                if frame_count % 10 == 0:
                    frame_data.append({
                        'frame_count': frame_count,
                        'latent_vector': latent_vector,
                        'updated': updated,
                        'min_distance': min_distance[i],
                        'min_dist_before_dim_reduction': min_dist_before_dim_reduction
                    })
            
            # 使用 VAE 模型进行降维和重构
            latent_vector_5, _ = encode_vector(vae_model_5, landmarks)
            latent_vector_10, _ = encode_vector(vae_model_10, landmarks)
            latent_vector_15, _ = encode_vector(vae_model_15, landmarks)
            
            # 使用 RWO 类处理降维后的向量
            updated_5, min_distance[3] = rwo_5.update(latent_vector_5, frame_count)
            updated_10, min_distance[4] = rwo_10.update(latent_vector_10, frame_count)
            updated_15, min_distance[5] = rwo_15.update(latent_vector_15, frame_count)
            
            # 增加向量计数
            total_vectors += 1

            if updated_5:
                novel_vectors[3] += 1
                feedback_index = (feedback_index + 1) % len(feedback_messages)  # 更新反馈消息索引
                last_feedback_time = time.time()  # 更新上一次反馈消息显示的时间
            if updated_10:
                novel_vectors[4] += 1
                feedback_index = (feedback_index + 1) % len(feedback_messages)  # 更新反馈消息索引
                last_feedback_time = time.time()  # 更新上一次反馈消息显示的时间
            if updated_15:
                novel_vectors[5] += 1
                feedback_index = (feedback_index + 1) % len(feedback_messages)  # 更新反馈消息索引
                last_feedback_time = time.time()  # 更新上一次反馈消息显示的时间
            
            # 记录最小距离
            min_distances[3].append(min_distance[3])
            min_distances[4].append(min_distance[4])
            min_distances[5].append(min_distance[5])
            
            # 记录当前 novel_vectors 数量和时间戳
            novel_vectors_over_time_vae_5.append(novel_vectors[3])
            novel_vectors_over_time_vae_10.append(novel_vectors[4])
            novel_vectors_over_time_vae_15.append(novel_vectors[5])
            timestamps_vae.append(time.time() - start_time)
    
    frame_count += 1
    
    # 显示总向量数、新颖向量数和程序运行时间
    info_texts = [
        f'Total Vectors: {total_vectors}',
        f'Novel Vectors 5 (PCA): {novel_vectors[0]}',
        f'Novel Vectors 10 (PCA): {novel_vectors[1]}',
        f'Novel Vectors 15 (PCA): {novel_vectors[2]}',
        f'Novel Vectors 5 (VAE): {novel_vectors[3]}',
        f'Novel Vectors 10 (VAE): {novel_vectors[4]}',
        f'Novel Vectors 15 (VAE): {novel_vectors[5]}',
        f'Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}'
    ]
    
    y0, dy = 30, 30
    for i, text in enumerate(info_texts):
        cv2.putText(frame, text, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 显示积极反馈消息
    if time.time() - last_feedback_time < feedback_display_time:
        cv2.putText(frame, feedback_messages[feedback_index], (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    # 显示处理后的帧
    cv2.imshow('Hand Tracking', frame)

    # 定时 3 分钟，程序运行超过 3 分钟自动关闭
    if time.time() - start_time >= 180:
        print("Program has been running for 3 minutes. Exiting...")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

# 计算最小距离的平均值
average_min_distances = []
for i, pca in enumerate(pca_models):
    average_min_distance = np.mean(min_distances[i])
    average_min_distances.append(average_min_distance)
    print(f"Average Minimum Distance for PCA {pca['d']}: {average_min_distance:.2f}")

average_min_distance_vae_5 = np.mean(min_distances[3])
average_min_distance_vae_10 = np.mean(min_distances[4])
average_min_distance_vae_15 = np.mean(min_distances[5])
average_min_distances.append(average_min_distance_vae_5)
average_min_distances.append(average_min_distance_vae_10)
average_min_distances.append(average_min_distance_vae_15)

print(f'Average Minimum Distance for VAE 5: {average_min_distance_vae_5}')
print(f'Average Minimum Distance for VAE 10: {average_min_distance_vae_10}')
print(f'Average Minimum Distance for VAE 15: {average_min_distance_vae_15}')

# 绘制三个模型在各自阈值下的 novel_vectors 随时间变化的趋势图
plt.figure(figsize=(12, 6))
colors_pca = ['b', 'g', 'r']
colors_vae = ['c', 'm', 'y']
for i, pca in enumerate(pca_models):
    plt.plot(pca['timestamps'], pca['novel_vectors'], label=f'PCA Model {pca["d"]} (Threshold={pca["threshold"]})', color=colors_pca[i])
plt.plot(timestamps_vae, novel_vectors_over_time_vae_5, label='VAE Model 5 (Threshold=1.35)', color=colors_vae[0])
plt.plot(timestamps_vae, novel_vectors_over_time_vae_10, label='VAE Model 10 (Threshold=2.95)', color=colors_vae[1])
plt.plot(timestamps_vae, novel_vectors_over_time_vae_15, label='VAE Model 15 (Threshold=4.35)', color=colors_vae[2])
plt.xlabel('Time (seconds)')
plt.ylabel('Novel Vectors')
plt.title('Novel Vectors Over Time for Different PCA and VAE Models')
plt.legend()
plt.grid(True)
plt.show()

