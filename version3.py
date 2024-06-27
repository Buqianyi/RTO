import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import json

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
        if self.bag is None:
            self.bag = np.array(vector[None, :])
            self.n_vecs = 1
            if self.output is not None:
                self.output["data"].append(vector.tolist())
                self.output["time"].append(t)
            return True
        ds = distance.cdist(self.bag, vector[None, :], self.metric)
        if np.min(ds) > self.threshold:
            self.bag = np.concatenate((self.bag, vector[None, :]), axis=0)
            self.n_vecs = len(self.bag)
            if self.output is not None:
                self.output["data"].append(vector.tolist())
                self.output["time"].append(t)
            return True
        return False

# Initialize MediaPipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize RWO
rwo = RWO(d=63, threshold=0.45, bag={"data": [], "time": []})

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize counters and storage for all vectors
vector_count = 0
qualified_vector_count = 0
all_vectors = {"data": [], "time": []}
show_feedback = False
feedback_timer = 0
feedback_messages = ["Perfect!", "Great!", "Awesome!", "Well done!", "Excellent!"]
feedback_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # flip the image horizontally
    frame = cv2.flip(frame, 1)
    
    # Convert color space
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and get results
    results = hands.process(image)
    
    # Draw hand landmarks and calculate vectors
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            vector_count += 1
            all_vectors["data"].append(landmarks.tolist())
            all_vectors["time"].append(vector_count)
            if rwo.update(landmarks, vector_count):
                qualified_vector_count += 1
                show_feedback = True
                feedback_timer = 20  # Show feedback for 20 frames
                feedback_index = (feedback_index + 1) % len(feedback_messages)
    
    # Display vector count and qualified vector count
    cv2.putText(frame, f'Total Vectors: {vector_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Qualified Vectors: {qualified_vector_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display feedback if a qualified vector is detected
    if show_feedback:
        cv2.putText(frame, feedback_messages[feedback_index], (frame.shape[1]//2 - 100, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        feedback_timer -= 1
        if feedback_timer <= 0:
            show_feedback = False

    # Display the resulting frame
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Add vector counts to the data
rwo.output["total_vectors"] = len(rwo.output["data"])
all_vectors["total_vectors"] = len(all_vectors["data"])

# Save the collected data to JSON files
with open('qualified_vectors.json', 'w') as f:
    json.dump(rwo.output, f, indent=4)

with open('all_vectors.json', 'w') as f:
    json.dump(all_vectors, f, indent=4)

# Release resources
cap.release()
cv2.destroyAllWindows()
