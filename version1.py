import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 转换颜色空间
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 处理图像并获取结果
    results = hands.process(image)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 获取每个数据点的坐标
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            # 将数据点转化为向量形式
            landmark_vector = np.array(landmarks).flatten()
            print("Hand landmarks as vector:", landmark_vector)
            
            # 绘制手部标记
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # 显示处理后的图像
    cv2.imshow('Hand Tracking', frame)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
