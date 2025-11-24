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

img_path = "./duck.jpg"

# Step 2: Candidate labels
labels = [
    "a picture of a woman", "a picture of a man",
    "a picture of a queen", "a picture of a king", "a picture of donald duck", 
    "a picture of a duck", "a picture of a bird"
]

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

# --- Visualization: show image and overlay confidence bar chart ---
import matplotlib.pyplot as plt
import numpy as np

# Ensure image is RGB and convert for plotting
image = Image.open(img_path).convert("RGB")
image_np = np.asarray(image)

# Move confidences to CPU and numpy
confidences = probs[0].cpu().numpy()

# Order labels by confidence (descending) for nicer bars
order = np.argsort(confidences)[::-1]
ordered_conf = confidences[order]
ordered_labels = [labels[i] for i in order]

# Create a figure with the image and an overlaid bar chart at the bottom
# Increase left margin so long label text fits, shrink bar area and fonts
fig = plt.figure(figsize=(8,9))
# make room on the left for long label names
fig.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.04)

# Image axes (fills most of figure)
ax_img = fig.add_axes([0.05, 0.28, 0.9, 0.67])  # x, y, w, h
ax_img.imshow(image_np)
ax_img.axis('off')

# Bar chart axes overlaid near the bottom with translucent background
# Move the bar axes right a bit so y-tick labels are inside the figure area
ax_bar = fig.add_axes([0.14, 0.03, 0.82, 0.17])
ax_bar.patch.set_alpha(0.6)

y_pos = np.arange(len(ordered_labels))
bar_containers = ax_bar.barh(y_pos, ordered_conf, color='C0', height=0.6)
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(ordered_labels, fontsize=9)
ax_bar.invert_yaxis()  # highest confidence on top
ax_bar.set_xlim(0, 1)
ax_bar.set_xlabel('Confidence')

# Reduce margins inside the bar axes to avoid clipping labels
for spine in ax_bar.spines.values():
    spine.set_linewidth(0.8)

# Annotate numeric confidence values next to bars (inside or just outside)
for i, v in enumerate(ordered_conf):
    x_pos = min(v + 0.02, 0.99)
    ax_bar.text(x_pos, i, f"{v:.2f}", va='center', fontsize=8)

plt.show()

# --- Embedding space visualization ---
try:
    # Try to get embeddings from the model outputs if available
    image_embed = outputs.image_embeds[0]
    text_embeds = outputs.text_embeds
except Exception:
    # Fallback: use the convenience methods
    with torch.no_grad():
        # inputs may contain pixel_values and input_ids/attention_mask
        if 'pixel_values' in inputs:
            image_embed = model.get_image_features(pixel_values=inputs['pixel_values'].to(device))[0]
        else:
            raise RuntimeError('No image tensor found for embedding computation')
        if 'input_ids' in inputs:
            text_embeds = model.get_text_features(input_ids=inputs['input_ids'].to(device), attention_mask=inputs.get('attention_mask', None).to(device))
        else:
            raise RuntimeError('No text tensor found for embedding computation')

# Move to CPU numpy
img_e = image_embed.detach().cpu().numpy()
txt_e = text_embeds.detach().cpu().numpy()

# Compute Euclidean distances from image embedding to each text embedding
from numpy.linalg import norm
distances = np.array([norm(img_e - t) for t in txt_e])
# --- Regularize latent space for visualization ---
# Make close embeddings closer and far embeddings farther by scaling
# the offset from the image embedding by a power-law multiplier.
def regularize_text_embeddings(txt_embeddings, img_embedding, power=1.8, eps=1e-8):
    """Return transformed text embeddings for visualization only.

    Args:
        txt_embeddings: (N, D) numpy array of text embeddings
        img_embedding: (D,) numpy array of image embedding
        power: float, power to raise the relative distances
        eps: small float to avoid div-by-zero
    """
    # compute distances
    dists = np.linalg.norm(txt_embeddings - img_embedding, axis=1)
    mean_dist = dists.mean() + eps
    # relative ratio to mean
    ratios = (dists / mean_dist)
    # power-law multiplier
    multipliers = (ratios ** power)
    multipliers = np.nan_to_num(multipliers, nan=1.0, posinf=ratios.max(), neginf=1.0)
    # apply multiplier along the direction vectors
    dirs = txt_embeddings - img_embedding[np.newaxis, :]
    transformed = img_embedding[np.newaxis, :] + dirs * multipliers[:, np.newaxis]
    return transformed

# Choose a power (>1 to exaggerate close/far differences)
power = 20
txt_e_reg = regularize_text_embeddings(txt_e, img_e, power=power)

# Recompute distances after regularization (these are the ones we show on the plot)
reg_distances = np.array([norm(img_e - t) for t in txt_e_reg])

# Build 2D projection (PCA via SVD) of the combined embeddings (use regularized text embeddings)
all_emb = np.vstack([img_e[np.newaxis, :], txt_e_reg])
all_emb_mean = all_emb.mean(axis=0)
X = all_emb - all_emb_mean
# SVD for PCA
U, S, Vt = np.linalg.svd(X, full_matrices=False)
coords = X.dot(Vt.T[:, :2])
img_xy = coords[0]
txt_xy = coords[1:]

# Create a new figure for embedding visualization
fig2 = plt.figure(figsize=(8,4))
ax_emb = fig2.add_axes([0.06, 0.10, 0.88, 0.85])

# Color points by regularized distance (closer = warmer)
# Use np.ptp for NumPy 2.0 compatibility (ndarray.ptp was removed)
normed = (reg_distances - reg_distances.min()) / (np.ptp(reg_distances) + 1e-8)
colors = plt.cm.viridis(1 - normed)

ax_emb.scatter(txt_xy[:, 0], txt_xy[:, 1], c=colors, s=80, edgecolor='k')
ax_emb.scatter(img_xy[0], img_xy[1], marker='*', s=200, c='red', edgecolor='k', label='image')

# Draw lines and annotate distances
for i, label in enumerate(labels):
    x1, y1 = img_xy
    x2, y2 = txt_xy[i]
    ax_emb.plot([x1, x2], [y1, y2], color='gray', linewidth=0.8, linestyle='--')
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax_emb.text(mx, my, f"{reg_distances[i]:.2f}", fontsize=8, color='black',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

# Annotate class names beside their points
for i, label in enumerate(labels):
    ax_emb.text(txt_xy[i, 0] + 0.01, txt_xy[i, 1] + 0.01, label, fontsize=9)

ax_emb.set_title(f'2D projection of CLIP embeddings (power={power:.2f} regularization)')
ax_emb.axis('equal')
ax_emb.grid(False)
plt.show()
