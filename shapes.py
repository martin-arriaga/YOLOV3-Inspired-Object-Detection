import cv2
import numpy as np
import os
import random

# Configuration
IMG_SIZE = 256
NUM_IMAGES = 500
OUTPUT_DIR = "datasets/shapes"
CLASSES = ["circle", "square", "triangle"]

# Create directory structure
for split in ["train", "val"]:
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)


def generate_shape(img_size):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    # Random background color
    img[:] = [random.randint(0, 50)] * 3

    shape_type = random.randint(0, 2)
    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

    # Random size and position
    w = random.randint(40, 100)
    h = w if shape_type != 2 else random.randint(40, 100)
    x_center = random.randint(w, img_size - w)
    y_center = random.randint(h, img_size - h)

    if shape_type == 0:  # Circle
        cv2.circle(img, (x_center, y_center), w // 2, color, -1)
    elif shape_type == 1:  # Square
        cv2.rectangle(img, (x_center - w // 2, y_center - h // 2), (x_center + w // 2, y_center + h // 2), color, -1)
    elif shape_type == 2:  # Triangle
        pts = np.array([[x_center, y_center - h // 2], [x_center - w // 2, y_center + h // 2],
                        [x_center + w // 2, y_center + h // 2]])
        cv2.drawContours(img, [pts], 0, color, -1)

    # Normalize coordinates for YOLO
    return img, shape_type, x_center / img_size, y_center / img_size, w / img_size, h / img_size


for i in range(NUM_IMAGES):
    split = "train" if i < NUM_IMAGES * 0.8 else "val"
    img, cls, x, y, w, h = generate_shape(IMG_SIZE)

    img_path = f"{OUTPUT_DIR}/images/{split}/shape_{i}.jpg"
    txt_path = f"{OUTPUT_DIR}/labels/{split}/shape_{i}.txt"

    cv2.imwrite(img_path, img)
    with open(txt_path, "w") as f:
        f.write(f"{cls} {x} {y} {w} {h}\n")

print(f"Generated {NUM_IMAGES} images in {OUTPUT_DIR}")