import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

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
                self.output["data"].append(vector.tolist())  # Convert to list
                self.output["time"].append(t)
            return True
        ds = distance.cdist(self.bag, vector[None, :], self.metric)
        if np.min(ds) > self.threshold:
            self.bag = np.concatenate((self.bag, vector[None, :]), axis=0)
            self.n_vecs = len(self.bag)
            if self.output is not None:
                self.output["data"].append(vector.tolist())  # Convert to list
                self.output["time"].append(t)
            return True
        return False

# Function to open camera and process hand gestures
def open_camera():
    # Initialize MediaPipe hand model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize RWO
    all_vectors = {"data": [], "time": []}
    qualified_vectors = {"data": [], "time": []}
    rwo = RWO(d=63, threshold=0.45, bag=qualified_vectors)

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize counters
    vector_count = 0
    qualified_vector_count = 0
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
                all_vectors["data"].append(landmarks.tolist())  # Convert to list
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Function to split data and maintain the same structure for both all_vectors and qualified_vectors
    def split_data(all_data, qualified_data, test_size=0.15, val_size=0.15):
        # Split all_data into train, validation, and test sets
        train_data, test_data, train_time, test_time = train_test_split(all_data["data"], all_data["time"], test_size=test_size, random_state=42)
        train_data, val_data, train_time, val_time = train_test_split(train_data, train_time, test_size=val_size / (1 - test_size), random_state=42)
        
        # Create a map from time to index in the all_data split
        time_to_split = {t: "train" for t in train_time}
        time_to_split.update({t: "validation" for t in val_time})
        time_to_split.update({t: "test" for t in test_time})

        # Assign qualified_data to the corresponding split
        qualified_splits = {"train": {"data": [], "time": []}, "validation": {"data": [], "time": []}, "test": {"data": [], "time": []}}
        for vec, t in zip(qualified_data["data"], qualified_data["time"]):
            split = time_to_split[t]
            qualified_splits[split]["data"].append(vec)
            qualified_splits[split]["time"].append(t)
        
        return {
            "all_vectors": {
                "train": {"data": train_data, "time": train_time},
                "validation": {"data": val_data, "time": val_time},
                "test": {"data": test_data, "time": test_time}
            },
            "qualified_vectors": qualified_splits
        }

    # Split the data
    splits = split_data(all_vectors, qualified_vectors)

    # Create a new directory for storing the file
    output_dir = "output_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create subdirectories for train, validation, and test sets
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Save the data to JSON files with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_data(data, base_dir, split_name):
        sub_dir = os.path.join(base_dir, timestamp)
        os.makedirs(sub_dir, exist_ok=True)
        file_path = os.path.join(sub_dir, f"{split_name}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    save_data(splits["all_vectors"]["train"], train_dir, "all_vectors_train")
    save_data(splits["all_vectors"]["validation"], val_dir, "all_vectors_validation")
    save_data(splits["all_vectors"]["test"], test_dir, "all_vectors_test")

    save_data(splits["qualified_vectors"]["train"], train_dir, "qualified_vectors_train")
    save_data(splits["qualified_vectors"]["validation"], val_dir, "qualified_vectors_validation")
    save_data(splits["qualified_vectors"]["test"], test_dir, "qualified_vectors_test")

    print(f"Data saved to {output_dir}")

# Run the open_camera function directly
open_camera()
