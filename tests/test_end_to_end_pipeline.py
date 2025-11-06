import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from models.yolo import YOLOv3
from datasets.vocdatasets import VOCDataset
from utils.metrics import DecodePredictions, non_max_suppression

# -------------------- Config --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 20
image_size = 416
anchors = [
    torch.tensor([[116, 90], [156, 198], [373, 326]]) / image_size,
    torch.tensor([[30, 61], [62, 45], [59, 119]]) / image_size,
    torch.tensor([[10, 13], [16, 30], [33, 23]]) / image_size
]

class_names = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]

# -------------------- Dataset --------------------
test_dataset = VOCDataset(
    img_folder="datasets/VOC/images/train2007",
    label_folder="datasets/VOC/labels/train2007",
    anchors=anchors,
    image_size=image_size,
    num_classes=num_classes
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# -------------------- Model --------------------
model = YOLOv3(num_classes=num_classes).to(device)
model.eval()

# If you have a checkpoint, uncomment:
# checkpoint = torch.load("checkpoints/yolov3_epoch1.pth", map_location=device)
# model.load_state_dict(checkpoint)

# -------------------- Inference & Plot --------------------
def plot_boxes(image, boxes, class_names):
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    ax = plt.gca()
    w_img, h_img = image.size

    for box in boxes:
        cls, conf, x, y, bw, bh = box
        x1 = (x - bw/2) * w_img
        y1 = (y - bh/2) * h_img
        x2 = (x + bw/2) * w_img
        y2 = (y + bh/2) * h_img
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1, f"{class_names[int(cls)]}:{conf:.2f}", color='yellow', fontsize=12)

    plt.axis('off')
    plt.show()

with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        outputs = model(images)
        for i, out in enumerate(outputs):
            print(f"Output {i} shape: {out.shape}")

        all_boxes = []
        for scale_idx, output in enumerate(outputs):
            boxes, obj, cls_probs = DecodePredictions(output, anchors[scale_idx], img_size=image_size)
            B, A, S, _, _ = boxes.shape
            for b in range(B):
                for a in range(A):
                    for i in range(S):
                        for j in range(S):
                            conf = obj[b,a,i,j].item()
                            class_idx = torch.argmax(cls_probs[b,a,i,j]).item()
                            if conf > 0.5:
                                x, y, w, h = boxes[b,a,i,j]
                                all_boxes.append([class_idx, conf, x.item(), y.item(), w.item(), h.item()])

        final_boxes = non_max_suppression(all_boxes, iou_threshold=0.3, confidence_threshold=0.5)
        img_pil = Image.fromarray((images[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8))
        plot_boxes(img_pil, final_boxes, class_names)
        break