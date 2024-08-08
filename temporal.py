import os
import cv2
import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained ResNet50 model (pre-trained on ImageNet)
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def calculate_temporal_consistency(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 2:
        print("Error: Video must have at least two frames.")
        return None

    frame_embeddings = []
    prev_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame for ResNet50
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(frame)

        # Compute ResNet50 embeddings for frame
        frame_embedding = resnet_model.predict(np.expand_dims(frame, axis=0))
        frame_embeddings.append(frame_embedding)

    cap.release()

    # Compute cosine similarity between adjacent frame embeddings
    similarities = []
    for i in range(1, len(frame_embeddings)):
        similarity = cosine_similarity(frame_embeddings[i-1], frame_embeddings[i])[0][0]
        if not np.isnan(similarity):
            similarities.append(similarity)

    # Check if there are valid similarities before calculating the mean
    if similarities:
        # Calculate mean cosine similarity score
        consistency_score = np.mean(similarities)
    else:
        # If all similarities are NaN, set consistency_score to NaN or any default value
        consistency_score = np.nan  # Or set to any default value you prefer

    return consistency_score

if __name__ == "__main__":
    
    # Directory containing video files
    input_folder = r"/content/drive/MyDrive/outputs"
    
    # Text file to save results
    output_file = r"/content/drive/MyDrive/outputs/results.csv"

    # Open text file for writing
    with open(output_file, mode='w') as file:
        # Loop through all mp4 files in input folder
        for root, _, files in os.walk(input_folder):
            for filename in files:
                if filename.endswith(".mp4"):
                    video_path = os.path.join(root, filename)

                    # Use filename (without extension) as prompt
                    prompt = os.path.splitext(filename)[0]

                    print(prompt)

                    # Calculate temporal consistency score for video
                    consistency_score = calculate_temporal_consistency(video_path)

                    print("score saved")
                    print(consistency_score)

                    # Write prompt and score to text file
                    file.write(f"{prompt}: {consistency_score}\n")

    print("Temporal consistency scores exported to:", output_file)
