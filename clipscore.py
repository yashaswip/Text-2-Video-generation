import os
import torch
import clip
import cv2
from PIL import Image
import csv

def get_clip_score(image, text, model, preprocess, device):
    # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text]).to(device)
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input.to(device))
        text_features = model.encode_text(text_input)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()
    
    # Round off the score to two decimal places
    clip_score_rounded = round(clip_score, 2)
    
    return clip_score_rounded


def main():
    # Check available CLIP models
    available_models = clip.available_models()
    print("Available CLIP models:", available_models)
    
    # Load the pre-trained CLIP model
    model, preprocess = clip.load("ViT-B/32", device="cuda")

    # Directory containing video files
    directory = r"/media/namrata/NAMRATA/Independent Study/Video-Generation-Independent-Study3/dataset"
    
    # CSV file to save results
    csv_file = r"/media/namrata/NAMRATA/Independent Study/Video-Generation-Independent-Study3/clipscore.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['video_path', 'prompt', 'clip_score'])  # Header
        
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".mp4"):
                    # Use the filename (without extension) as the prompt
                    prompt = os.path.splitext(filename)[0]
                    video_path = os.path.join(root, filename)
                    
                    # Read video frames
                    cap = cv2.VideoCapture(video_path)
                    success, frame = cap.read()
                    if not success:
                        print(f"Error reading {video_path}")
                        continue
                    
                    # Convert the frame to PIL image
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    clip_score = get_clip_score(image, prompt, model, preprocess, 'cuda')
                    writer.writerow([video_path, prompt, clip_score])
                    print(f"Processed {video_path}")

if __name__ == "__main__":
    main()
