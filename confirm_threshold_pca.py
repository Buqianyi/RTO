import cv2
import mediapipe as mp
import numpy as np
import joblib
from scipy.spatial import distance
import time
import matplotlib.pyplot as plt

# 加载训练好的标准化模型和 PCA 模型
scaler = joblib.load('scaler_model.pkl')
pca_model = joblib.load('pca_model_5.pkl')

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
rwo = RWO(d=5, threshold=3.45, bag={"data": [], "time": []})

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

start_time = time.time()  # 记录程序开始运行的时间

# 用于存储每10帧的信息
frame_data = []
min_distances = []  # 记录每帧的最小距离

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
            
            # 使用 PCA 模型进行降维
            latent_vector = pca_model.transform(standardized_vector.reshape(1, -1)).squeeze()
            
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
            
            # 打印相关信息（仅每10帧打印一次）
            if frame_count % 10 == 0:
                frame_data.append({
                    'frame_count': frame_count,
                    'latent_vector': latent_vector,
                    'updated': updated,
                    'min_distance': min_distance,
                    'min_dist_before_dim_reduction': min_dist_before_dim_reduction
                })
    
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

# 打印存储的帧数据
for data in frame_data:
    print(f"Frame {data['frame_count']}:")
    print(f"  Latent vector: {data['latent_vector']}")
    print(f"  Updated: {data['updated']}")
    print(f"  Min distance: {data['min_distance']}")
    print(f"  Min distance before dimensionality reduction: {data['min_dist_before_dim_reduction']}")

# 计算最小距离的平均值
average_min_distance = np.mean(min_distances)
print(f"Average Minimum Distance: {average_min_distance:.2f}")

# 绘制 novel_vectors 随时间变化的图像
plt.figure(figsize=(10, 5))
plt.plot(timestamps, novel_vectors_over_time, label='Novel Vectors under pca_model_5')
plt.xlabel('Time (seconds)')
plt.ylabel('Novel Vectors')
plt.title('Novel Vectors Over Time under threshold 3.65')
plt.legend()
plt.grid(True)
plt.show()
