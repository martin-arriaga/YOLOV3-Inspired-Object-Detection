import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

class VOCDataset(Dataset):
    def __init__(self, img_folder, label_folder, anchors, image_size=416, S=[13,26,52], num_classes=20, transform=None):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.image_size = image_size
        self.S = S
        self.anchors = anchors  # list of tensors per scale (3x2)
        self.num_classes = num_classes
        self.transform = transform
        self.images = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_folder, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # C H W

        # Load label
        label_path = os.path.join(self.label_folder, os.path.splitext(self.images[idx])[0] + ".txt")
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            labels = np.loadtxt(label_path).reshape(-1, 5)# (class, x, y, w, h)
            boxes = torch.tensor(labels[:,1:5], dtype=torch.float32)
            classes = torch.tensor(labels[:,0], dtype=torch.long)
        else:
            boxes = np.zeros((0,5))
            classes = np.zeros((0,))

        if self.transform:
            image, boxes = self.transform(image, boxes)


        return image,boxes, classes

    @staticmethod
    def iou_wh(box1, anchors):
        """
        Compute IoU based only on width and height (no center)
        box1: tensor (w,h)
        anchors: tensor (num_anchors,2)
        """
        box1_area = box1[0] * box1[1]
        anchors_area = anchors[:,0] * anchors[:,1]

        inter_w = torch.min(box1[0], anchors[:,0])
        inter_h = torch.min(box1[1], anchors[:,1])
        inter_area = inter_w * inter_h

        return inter_area / (box1_area + anchors_area - inter_area + 1e-6)

    """
            # Create target tensors for 3 scales
            targets = [torch.zeros((3, S, S, self.num_classes + 5)) for S in self.S]

            for box in boxes:
                class_label, x, y, width, height = box

                # Compute IoU with all anchors
                ious = []
                for scale_idx, scale_anchors in enumerate(self.anchors):
                    iou = self.iou_wh(torch.tensor([width, height]), scale_anchors)
                    ious.append(iou)
                ious = torch.cat(ious)
                anchor_idx = ious.argmax()
                scale_idx = anchor_idx // 3
                anchor_on_scale = anchor_idx % 3

                S = self.S[scale_idx]
                i, j = min(int(S * y), S - 1), min(int(S * x), S - 1)

                # tx, ty relative to grid cell
                tx = S * x - j
                ty = S * y - i

                # tw, th relative to anchor
                anchor_w, anchor_h = self.anchors[scale_idx][anchor_on_scale]
                tw = torch.log(width / anchor_w + 1e-10)
                th = torch.log(height / anchor_h + 1e-10)

                targets[scale_idx][anchor_on_scale, i, j, 0] = tx
                targets[scale_idx][anchor_on_scale, i, j, 1] = ty
                targets[scale_idx][anchor_on_scale, i, j, 2] = tw
                targets[scale_idx][anchor_on_scale, i, j, 3] = th
                targets[scale_idx][anchor_on_scale, i, j, 4] = 1  # objectness
                targets[scale_idx][anchor_on_scale, i, j, 5 + int(class_label)] = 1  # class one-hot
            """