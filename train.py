import random

import numpy as np
import torch
import json
from torch.utils.data import DataLoader
from datasets.vocdatasets import VOCDataset
from models.yolo import YOLOv3
from models.loss import YoloLoss
from tqdm import tqdm
from utils.target_encoder import encode_yolo_targets
import os
import time
from debug_logger.debug import logger, set_debug

# -----------------------------
# Hyperparameters
# -----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 20
image_size = 416
batch_size = 8
learning_rate = 0.001
num_epochs = 20
set_debug(True)
print_debug = True
force_new_train_session = False
# -----------------------------
# Anchors (normalized)
# -----------------------------
anchors = [
    torch.tensor([[116, 90], [156, 198], [373, 326]]) / image_size,  # scale 13
    torch.tensor([[30, 61], [62, 45], [59, 119]]) / image_size,  # scale 26
    torch.tensor([[10, 13], [16, 30], [33, 23]]) / image_size  # scale 52
]
anchors = [a.to(device) for a in anchors]

def yolo_collate(batch):
    images = torch.stack([b[0] for b in batch], dim=0)  # stack images
    boxes_batch = [b[1] for b in batch]  # list of boxes per image
    labels_batch = [b[2] for b in batch]  # list of labels per image
    return images, boxes_batch, labels_batch

# -----------------------------
# Dataset & DataLoader
# -----------------------------
train_dataset = VOCDataset(
    img_folder="datasets/VOC/images/train2007",
    label_folder="datasets/VOC/labels/train2007",
    anchors=anchors,
    image_size=image_size,
    num_classes=num_classes,
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=yolo_collate)

# Optional validation dataset
val_dataset = VOCDataset(
    img_folder="datasets/VOC/images/val2007",
    label_folder="datasets/VOC/labels/val2007",
    anchors=anchors,
    image_size=image_size,
    num_classes=num_classes,
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=yolo_collate)

# -----------------------------
# Model, Loss, Optimizer
# -----------------------------
model = YOLOv3(num_classes=num_classes).to(device)
criterion = YoloLoss(anchors=anchors, num_classes=num_classes, imgsize=image_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#---------------
#Track Sessions
#---------------
checkpoints_root_directory = "checkpoints"
os.makedirs(checkpoints_root_directory, exist_ok=True)
sessions_file = os.path.join(checkpoints_root_directory, "latest_sessions.txt")

if os.path.exists(sessions_file) and not force_new_train_session:
    with open(sessions_file, "r") as f:
        list_of_sessions = f.read().strip().splitlines()
    if list_of_sessions:
        sessions = list_of_sessions[-1]
        logger.info(f"Resuming training session: {sessions}")
    else:
        sessions = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
        with open(sessions_file, "a") as f:
            f.write(sessions + "\n")
        logger.info(f"No sessions found, Starting first session {sessions}")
else:
    sessions = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    with open(sessions_file, "a") as f:
        f.write(sessions+"\n")
    logger.info(f"Starting and storing new Session: {sessions}")

checkpoints_dir = os.path.join(checkpoints_root_directory, sessions)
os.makedirs(checkpoints_dir, exist_ok=True)
latest_checkpoint = os.path.join(checkpoints_dir, "latest.pth")

#-------------
# Resume ?
#-------------
start_epoch = 0
if os.path.exists(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    logger.info(f"Resuming training from epoch {start_epoch}")

metadata_path = os.path.join(checkpoints_dir, "metadata.json")
if not os.path.exists(metadata_path):
    metadata = {
        'session_name': sessions,
        'time': time.strftime("%Y-%m-%d-%H:%M:%S"),
        'image_size': image_size,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'anchors': [
            [[116, 90], [156, 198], [373, 326]],  # scale 13
            [[30, 61], [62, 45], [59, 119]],      # scale 26
            [[10, 13], [16, 30], [33, 23]]  ,
            ],
        'notes': 'First training sessions hyperparameters',

    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        logger.info(f"Wrote metadata to {metadata_path}")
else:
    logger.info(f"Metadata file {metadata_path} already exists, skipping metadata save.")


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(start_epoch, num_epochs):
    print_debug = True
    logger.info(f"\nTraining started Epoch {epoch }/{num_epochs}")
    model.train()
    running_loss = 0
    loop = tqdm(train_loader, leave=True)

    for images, boxes_batch, labels_batch in loop:
        images = images.to(device)

        targets = encode_yolo_targets(
            target_boxes=[b.to(device) for b in boxes_batch],
            target_labels=[l.to(device) for l in labels_batch],
            anchors=anchors,
            strides=[13,26,52],
            num_classes=num_classes,
            device=device,
        )

        optimizer.zero_grad()
        outputs = model(images)  # list of 3 scales

        loss = torch.tensor(0.,device  =device)
        for scale_idx, output in enumerate(outputs):
            loss += criterion(output, targets[scale_idx], anchors[scale_idx].to(device))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        loop.set_postfix(batch_loss=loss.item(), avg_loss=running_loss / (loop.n + 1))
        if print_debug:
            with torch.no_grad():
                from utils.metrics import DecodePredictionsBatch
                boxes, obj, cls_probs = DecodePredictionsBatch(outputs[0], anchors[0], img_size=image_size)
                print(f"Sample decoded box [0]: {boxes[0, 0, :]}")  # check first box
                print(f"Objectness [0]: {obj[0, 0].item():.4f}, Class prob [0]: {cls_probs[0, 0, :].max().item():.4f}")
            print_debug = False  # only print once per epoch

    # Print average epoch loss
    avg_epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

    # -----------------------------
    # Save checkpoint
    # -----------------------------
    epoch_path = os.path.join(checkpoints_dir, f"yolov3_epoch_{epoch + 1}.pth")
    torch.save({
        "epoch": epoch ,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_epoch_loss
    }, epoch_path)

    # Save latest epoch
    torch.save({
        "epoch": epoch ,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_epoch_loss
    }, latest_checkpoint)
    logger.info(f"Saved epoch {epoch + 1}/{num_epochs} and updated latest path")


    '''
     # -----------------------------
    # Optional: Validation Step
    # -----------------------------
    model.eval()
    val_loss = torch.tensor(0.,device  =device)
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            outputs = model(images)
            loss = torch.tensor(0.,device  =device)
            for scale_idx, output in enumerate(outputs):
                loss += criterion(output, targets[scale_idx], anchors[scale_idx].to(device))
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    '''

