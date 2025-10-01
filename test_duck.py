import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO

import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

# Load CLIP model + processor from Hugging Face
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

img_path = "/home/create.aau.dk/az66ep/duck_classification_toy/duck.jpg"

# Step 2: Candidate labels
labels = ["a photo of donald duck", "a photo of a duck", "a photo of a cat", "a photo of a dog", "a photo of a bird"]

# Step 3: Prepare inputs
image = Image.open(img_path)
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)

# Step 4: Run CLIP
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # similarity score
    probs = logits_per_image.softmax(dim=1)

# Step 5: Print result
best_idx = probs[0].argmax().item()
print(f"Predicted label: {labels[best_idx]} (confidence {probs[0][best_idx]:.2f})")
# print other confidences
for i, label in enumerate(labels):
    if i != best_idx:
        print(f"Other label: {label} (confidence {probs[0][i]:.2f})")
