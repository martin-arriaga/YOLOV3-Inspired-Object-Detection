import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from models.yolo import YOLOv3
from utils.metrics import DecodePredictionsBatch, non_max_suppression, non_max_suppression_vectorized
from debug_logger.debug import logger

from pathlib import Path
import json

# -----------------------------
# Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 20
image_size = 416

# Keep anchors in pixels here, no need to normalize for decoding
anchors = [
    torch.tensor([[116, 90], [156, 198], [373, 326]]) / image_size,  # scale 13
    torch.tensor([[30, 61], [62, 45], [59, 119]])/ image_size,  # scale 26
    torch.tensor([[10, 13], [16, 30], [33, 23]])/ image_size  # scale 52
]

class_names = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# -----------------------------
# Load model checkpoint
# -----------------------------
checkpoints_root_directory = Path("checkpoints")
session_file_history = checkpoints_root_directory / "latest_sessions.txt"

if not session_file_history.exists():
    raise FileNotFoundError("No previous training sessions found")

with open(session_file_history, "r") as f:
    latest_session = f.read().strip().splitlines()[-1]

latest_checkpoint = checkpoints_root_directory / latest_session / "latest.pth"
metadata_path = checkpoints_root_directory / latest_session / "metadata.json"

model = YOLOv3(num_classes=num_classes).to(device)
checkpoint = torch.load(latest_checkpoint, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with open(metadata_path, "r") as f:
    metadata = json.load(f)

logger.info(f"Model loaded from session {latest_session}")

# -----------------------------
# Load and preprocess image
# -----------------------------
img_path = "./datasets/VOC/images/test2007/000008.jpg"
image = Image.open(img_path).convert("RGB")
image = image.resize((image_size, image_size))
image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

# -----------------------------
# Run inference
# -----------------------------
with torch.no_grad():
    outputs = model(image_tensor)

all_boxes = []
image_width, image_height = image.size

# -----------------------------
# Decode predictions and convert to pixels
# -----------------------------
for scale_idx, output in enumerate(outputs):
    boxes, obj, cls_probs = DecodePredictionsBatch(output, anchors[scale_idx], img_size=image_size)
    B, N, _ = boxes.shape

    for b in range(B):
        for n in range(N):
            conf = obj[b, n].item()
            if conf < 0.05:  # keep low threshold for debugging
                continue
            cls_idx = torch.argmax(cls_probs[b, n]).item()
            x_center = boxes[b, n, 0].item() * image_width
            y_center = boxes[b, n, 1].item() * image_height
            w = boxes[b, n, 2].item() * image_width
            h = boxes[b, n, 3].item() * image_height

            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            all_boxes.append([x1, y1, x2, y2, cls_idx,conf])
# -----------------------------
# Apply Non-Max Suppression
# -----------------------------
all_boxes = torch.tensor(all_boxes, dtype=torch.float32)
image_idx_tensor = torch.full((all_boxes.shape[0],1 ), 0, device=device)
images_boxes_with_idx = torch.cat([all_boxes, image_idx_tensor], dim=1)

final_boxes = non_max_suppression_vectorized(images_boxes_with_idx, iou_threshold=0.7, confidence_threshold=0.9)
logger.info(f"Before NMS: {len(all_boxes)} boxes")
logger.info(f"After NMS: {len(final_boxes)} boxes")


# -----------------------------
# Visualization
# -----------------------------
def plot_boxes(image_path, boxes, class_names=None):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size))
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    ax = plt.gca()

    for box in boxes:
        x1, y1, x2, y2 , cls, conf,_= box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        label = f"{class_names[int(cls)] if class_names else cls}:{conf:.2f}"
        ax.text(x1, y1, label, color='yellow', fontsize=12)

    plt.axis('off')
    plt.show()


plot_boxes(img_path, final_boxes, class_names)






