import random
import numpy as np
import torch
import json
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
from typing import Tuple, List

from datasets.vocdatasets import VOCDataset
from models.yolo import YOLOv3
from models.loss import YoloLoss
from tqdm import tqdm

from utils.mAP import my_mAP
from utils.metrics import DecodePredictionsBatch, non_max_suppression, non_max_suppression_vectorized
from utils.target_encoder import encode_yolo_targets
import time
from debug_logger.debug import logger, set_debug

# -----------------
# Helper Functions
# -----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def convert_to_pixel_coordinates(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:

    # Convert normalized coordinates to image coordinates
    image_height, image_width = image_size
    x_center, y_center, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = (x_center - w / 2) * image_width
    y1 = (y_center - h / 2) * image_height
    x2 = (x_center + w / 2) * image_width
    y2 = (y_center + h / 2) * image_height

    return torch.stack([x1, y1, x2, y2], dim = -1)

    ''' 
    # Convert normalized coordinates to image coordinates
    x_center = boxes[b, n, 0].item() * image_width
    y_center = boxes[b, n, 1].item() * image_height
    w = boxes[b, n, 2].item() * image_width
    h = boxes[b, n, 3].item() * image_height

    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2

    all_detections.append([class_idx, conf, x1, y1, x2, y2])
    '''
def plot_ground_truth_vs_model_predictions(image_tensor, ground_truth_boxes, predicted_boxes, class_names= None):
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    ax = plt.gca()

    for box in ground_truth_boxes:
        x1, y1, x2, y2 = box
        rectangle = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='green', linewidth=2)
        ax.add_patch(rectangle)
    for box in predicted_boxes:
        x1, y1, x2, y2, conf, cls,_ = box
        rectangle = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
        ax.add_patch(rectangle)
        if class_names:
            ax.text(x1, y1, f"{class_names[int(cls)]}:{conf:.2f}", color='yellow', fontsize=8)
    plt.axis('off')
    plt.show()
    plt.close()

def plot_anchors_and_groundtruth(image_tensor, gt_boxes, anchors, strides, class_names= None, max_boxes= 10):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    H , W , C = image.shape

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    ax = plt.gca()

    # Plot GT boxes
    for idx, box in enumerate(gt_boxes):

        x1, y1, x2, y2 = box

        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='green', linewidth=2)
        ax.add_patch(rect)
        if class_names is not None:
            ax.text(x1, y1, f"GT", color='green', fontsize=10)

    # Plot anchor boxes at the corresponding grid cells
    for scale_idx, S in enumerate(strides):
        stride = H / S  # assuming square images
        for anchor in anchors[scale_idx]:
            aw, ah = anchor.tolist()
            aw *= W
            ah *= H

            x1 = W / 2 - aw / 2

            y1 = H / 2 - ah / 2
            rect = plt.Rectangle((x1, y1), aw, ah, fill=False, color=['red', 'blue', 'orange'][scale_idx],
                                 linestyle='--', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x1, y1, f"A{scale_idx}", color=['red', 'blue', 'orange'][scale_idx], fontsize=8)

    plt.axis('off')
    plt.show()
    plt.close()


def safe_move_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def save_metadata(metadata: dict, metadata_path: str):
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def yolo_collate(batch):
    images = torch.stack([b[0] for b in batch], dim=0)  # stack images
    boxes_batch = [b[1] for b in batch]  # list of boxes per image
    labels_batch = [b[2] for b in batch]  # list of labels per image
    return images, boxes_batch, labels_batch

#------------------------------
# Training and Validation loops
#------------------------------

def train_model(model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
                train_loader: DataLoader, anchors:List[ torch.Tensor], device: torch.device,  epochs:int ,num_epochs: int, writer: SummaryWriter, debug: bool):
    model.train()
    running_loss = 0.0
    batches = 0
    loop = tqdm(train_loader, leave=True)
    loop.set_description(f"Train Epoch {epochs }/{num_epochs}")
    anchors = [a.to(device) for a in anchors]
    for images, boxes_batch, labels_batch in loop:
        images = images.to(device)
        targets = encode_yolo_targets(
            target_boxes=[b.to(device) for b in boxes_batch],
            target_labels=[l.to(device) for l in labels_batch],
            anchors=anchors,
            strides=[13, 26, 52],
            num_classes=model.num_classes if hasattr(model, 'num_classes') else 20,
            device=device,
        )

        #with torch.amp.autocast(device_type = 'cuda', enabled=(device.type == 'cuda')):
        with torch.autocast(device_type=device.type , enabled=(device.type != 'cpu')):
            outputs = model(images)
            loss= torch.zeros((), device=device)
            for scale_idx, output in enumerate(outputs):
                loss += criterion(output, targets[scale_idx], anchors[scale_idx])
                logger.debug(f"loss for scale {scale_idx} loss: {loss.item():.6f}")
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        batches += 1
        running_loss += loss.item()
        loop.set_postfix(batch_loss=loss.item(), avg_loss=running_loss/batches)

        if batches % 10 == 0:
            writer.add_scalar("training loss",loss.item(),epochs * len(train_loader) + batches)
        if debug and batches == 1:
            image_tensor = images[0].detach().cpu()
            gt_boxes = boxes_batch[0]
            plot_anchors_and_groundtruth(image_tensor, gt_boxes, anchors, strides=[13,26,52], class_names=None)
            boxes, obj, cls_probs = DecodePredictionsBatch(output, anchors[scale_idx], img_size=images.shape[-1])
            writer.add_scalar("predictions", float(obj[0, 0].detach().cpu().numpy()), epochs)

    avg_loss = running_loss /max(1, batches)
    writer.add_scalar("training loss",avg_loss,epochs)
    return avg_loss

@torch.no_grad()
def validate_model(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    val_loader: DataLoader,
    anchors: List[torch.Tensor],
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    confidence_threshold: float,
    visual_interval: int,
    num_epochs: int,
    debug: bool=False, ):

    model.eval()
    running_validation_loss = 0.0
    batches = 0
    loop = tqdm(val_loader, leave=True)
    #---Predictions---
    all_detections = []
    #---Ground truths-----
    all_ground_truths = []
    print_first_and_second = 2
    anchors = [a.to(device) for a in anchors]

    for images, boxes_batch, labels_batch in loop:
        loop.set_description(f"Validation for Epoch {epoch }/{num_epochs}")


        images = images.to(device, non_blocking=True)

        targets = encode_yolo_targets(
            target_boxes=[b.to(device) for b in boxes_batch],
            target_labels=[l.to(device) for l in labels_batch],
            anchors=anchors,
            strides=[13, 26, 52],
            num_classes=model.num_classes if hasattr(model, "num_classes") else 20,
            device=device,
        )

        #---------------
        # Sanity Checks
        #---------------
        if debug and batches == 0:
            logger.debug("=== Sanity check: anchors & first batch of target boxes ===")
            for scale_idx, scale_anchors in enumerate(anchors):
                for a_idx, a in enumerate(scale_anchors):
                    logger.debug(f"Scale {scale_idx}, Anchor {a_idx}: {a.cpu().numpy()}")
            logger.debug(f"First GT box in batch:{boxes_batch[0][0]}")

        outputs = model(images)
        loss = torch.zeros((), device=device)
        for scale_idx, output in enumerate(outputs):
            loss = loss + criterion(output, targets[scale_idx], anchors[scale_idx].to(device))

        running_validation_loss += loss.item()
        batches += 1

        for scale_idx, output in enumerate(outputs):
            boxes, obj, cls_probs = DecodePredictionsBatch(output, anchors[scale_idx], img_size=images.shape[2])
            B, N, _ = boxes.shape
            for b_idx in range(B):
                logger.debug(f"\n[DEBUG] Boxes batch values should be normalized ie between [0,1] :{boxes_batch[b_idx]}")
                conf = obj[ b_idx]
                class_scores, class_ids = torch.max(cls_probs[b_idx], dim=-1)  # [N]
                final_conf = conf * class_scores
                mask = final_conf > confidence_threshold
                if mask.sum() == 0:
                    continue
                selected_boxes = boxes[b_idx][mask]  # [M, 4]
                selected_scores = final_conf[mask]  # [M]
                selected_classes = class_ids[mask]
                logger.debug(f"\n[DEBUG]Selected boxes {selected_boxes}")

                absolute_coord = convert_to_pixel_coordinates(selected_boxes, (images.shape[2], images.shape[3]))
                logger.debug(f"\n[DEBUG]Selectedboxes:{absolute_coord[:5]}" )
                widths = absolute_coord[:, 2] - absolute_coord[:, 0]
                heights = absolute_coord[:, 3] - absolute_coord[:, 1]
                logger.debug(f"\n[DEBUG]SelectedPredicted box widths min/max: {widths.min():.1f}/{widths.max():.1f}")
                logger.debug(f"\n[DEBUG]SelectedPredicted box heights min/max: {heights.min():.1f}/{heights.max():.1f}")

                if debug and batches % 250 == 0:
                    for i, box in enumerate(absolute_coord):
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1
                        logger.debug(f"\n[DEBUG]Selected Predicted box {i}: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}, width={width:.1f}, height={height:.1f}")

                #---Save in [x1,y1,x2,y2,score,class] format---
                images_boxes = torch.stack([ torch.tensor(
                    [box[0], box[1], box[2], box[3], score, cls],device=device)
                        for box, score, cls in zip(absolute_coord, selected_scores, selected_classes)])
                image_idx_tensor = torch.full((images_boxes.shape[0],1 ), b_idx, device=device)
                images_boxes_with_idx = torch.cat([images_boxes, image_idx_tensor], dim=1) # [num_boxes,7]

                logger.debug(f"\n[DEBUG]SelectedPredicted boxes (normalized) first image: {boxes[0, :5, :4]}")
                logger.debug(f"\n[DEBUG]SelectedAbsolute pixel boxes first image: {absolute_coord[:5]}")
                logger.debug(f"\n[DEBUG]SelectedNumber of selected boxes after conf threshold: {len(selected_boxes)}")
                logger.debug(f"\n[DEBUG]SelectedGround truth boxes first image: {boxes_batch[:5]}")
                logger.debug(f"\n[DEBUG] Before Nms: {len(selected_boxes)}")
                final_prediction_boxes = non_max_suppression_vectorized(images_boxes_with_idx, iou_threshold=0.6, confidence_threshold=0.7)
                logger.debug(f"\n[DEBUG] After NMS: {len(final_prediction_boxes)}")

                all_detections.extend(final_prediction_boxes)

        for b_idx in range(len(boxes_batch)):
            gt_absolute = convert_to_pixel_coordinates(boxes_batch[b_idx], (images.shape[2], images.shape[3]))
            for label, box in zip(labels_batch[b_idx], gt_absolute):
                all_ground_truths.append(torch.tensor([b_idx,label,  box[0], box[1], box[2], box[3]]))

        if batches % visual_interval == 0 : #only run every nth batch
            if len(all_detections) > 0 and len(all_ground_truths) > 0:
                img_idx = 0
                mask = final_prediction_boxes[:,-1] == img_idx
                img_boxes = final_prediction_boxes[mask].cpu().numpy()
                gt_boxes = convert_to_pixel_coordinates(boxes_batch[img_idx], (images.shape[2], images.shape[3])).cpu().numpy()

                plot_ground_truth_vs_model_predictions(images[0],predicted_boxes= img_boxes, ground_truth_boxes= gt_boxes, class_names=None)
    avg_val_loss = running_validation_loss / max(1, batches)
    writer.add_scalar("validation/epoch_loss", avg_val_loss, epoch)
    if len(all_detections) == 0 or len(all_ground_truths) == 0:
        mAP = 0
    else:
        #mAP = my_mAP(all_detections, all_ground_truths, iou_threshold=0.6, num_classes=model.num_classes, device=device)
        mAP = 0
    writer.add_scalar("validation/mAP", mAP, epoch)

    return avg_val_loss , mAP

def main(args):

    set_debug(args.debug)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    set_seed(args.seed)
    anchors = [
        torch.tensor([[116, 90], [156, 198], [373, 326]]) / args.image_size,  # scale 13
        torch.tensor([[30, 61], [62, 45], [59, 119]])/ args.image_size,  # scale 26
        torch.tensor([[10, 13], [16, 30], [33, 23]])/ args.image_size # scale 52
    ]
    #------------------------
    # Training and Validation set
    #------------------------
    train_dataset = VOCDataset(
        img_folder="datasets/VOC/images/train2012",
        label_folder="datasets/VOC/labels/train2012",
        anchors=anchors,
        image_size=args.image_size,
        num_classes=args.num_classes,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=yolo_collate, pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    val_dataset = VOCDataset(
        img_folder="datasets/VOC/images/val2012",
        label_folder="datasets/VOC/labels/val2012",
        anchors=anchors,
        image_size=args.image_size,
        num_classes=args.num_classes,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=max(1, args.num_workers // 2),pin_memory=(device.type == "cuda"), collate_fn=yolo_collate)

    #---------------------------------------
    # Model, Criterion, Optimizer, scheduler
    #---------------------------------------

    model = YOLOv3(num_classes=args.num_classes).to(device)
    criterion = YoloLoss(anchors=anchors, num_classes=args.num_classes, imgsize=args.image_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    if device.type == "cuda":
        scaler = torch.amp.GradScaler( 'cuda',enabled=(device.type == 'cuda'))
    else:
        scaler = None

    # ---------------
    # Session Management
    # ---------------

    checkpoints_root_directory =Path(args.checkpoints_root)
    checkpoints_root_directory.mkdir(parents=True, exist_ok=True)
    session_history_file = checkpoints_root_directory / "latest_sessions.txt"
    current_session_name = None

    if session_history_file.exists() and not args.force_new_training_session:
        with open(session_history_file, "r") as f:
            list_of_sessions = f.read().strip().splitlines()
        if list_of_sessions:
            current_session_name = list_of_sessions[-1]
            logger.info(f"Resuming training session: {current_session_name}")
        else:
            current_session_name = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
            with open (session_history_file, "a") as f:
                f.write(current_session_name+ "\n")
            logger.info(f"No sessions found, Starting first session {current_session_name}")
    else:
        current_session_name = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
        with open(session_history_file, "a") as f:
            f.write(current_session_name + "\n")
        logger.info(f"New training session detected. Review and Revise Hyperparameters \n"
                    f"Storing new session run with name: {current_session_name}")

    checkpoints_session_name = checkpoints_root_directory / current_session_name
    checkpoints_session_name.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = checkpoints_session_name / "latest.pth"

    tensorboard_dir = checkpoints_session_name / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    start_epoch = 0
    best_val_loss = float("inf")

    if latest_checkpoint.exists():
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        safe_move_to_device(optimizer, device)
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"Latest checkpoint found at {latest_checkpoint}, last succesful saved epoch {start_epoch - 1}")

    metadata_path = checkpoints_session_name / "metadata.json"
    metadata = {
        "session_name": current_session_name,
        "time": time.strftime("%Y-%m-%d-%H:%M:%S"),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "anchors": [a.tolist() for a in anchors],
        "seed": args.seed,
        "notes": args.notes,
    }
    if not metadata_path.exists():
        save_metadata(metadata, str(metadata_path))

    #------------------
    # Start Training
    #------------------
    if not args.force_new_training_session:
        logger.info(
            f"Starting Training session from last successful saved epoch {start_epoch - 1 } resuming training with epoch {start_epoch }")
    else:
        logger.info(f"Starting new training session: {current_session_name}")

    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"Starting epoch: {epoch}")
        t0= time.time()
        print_debug_flag = True if epoch == start_epoch else False

        training_loss = train_model(model = model, criterion= criterion, optimizer= optimizer,
        scaler=scaler, train_loader= train_loader, anchors = anchors,
        device=device, writer=writer, epochs= epoch,num_epochs=args.num_epochs,
        debug= print_debug_flag)

        validation_loss, mAP = validate_model(model = model, criterion= criterion,
        val_loader= val_loader, anchors = anchors, epoch=epoch, writer=writer,
        device=device,confidence_threshold= .5, debug= args.debug,
        visual_interval= args.visual_interval, num_epochs=args.num_epochs)

        scheduler.step()

        #----------------
        # Save Progress
        #----------------
        checkpoint= {
            "epoch": epoch,  # last finished epoch (0-based)
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": float(training_loss),
            "val_loss": float(validation_loss),
            "best_val_loss": float(best_val_loss),
            "mAP" : float(mAP),
        }
        save_checkpoints_path = checkpoints_session_name / f"yolov3_epoch_{epoch}.pth"
        torch.save(checkpoint, save_checkpoints_path)
        torch.save(checkpoint, latest_checkpoint)

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_checkpoint = checkpoints_session_name / "best_session.pth"
            torch.save(checkpoint, best_checkpoint)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("train/epoch_loss", training_loss, epoch)
        writer.add_scalar("val/epoch_loss", validation_loss, epoch)
        t1 = time.time()
        logger.info(f"Epoch {epoch}/{args.num_epochs} done. mAP for epoch {epoch}: {mAP} train_loss={validation_loss:.4f} val_loss={validation_loss:.4f} time={(t1-t0):.1f}s")
    writer.close()
    print(f"Training completed, best_val_loss={best_val_loss:.4f}")

#-----------
# Command line interface
#----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-img-folder", type=str, default="datasets/VOC/images/train2012")
    parser.add_argument("--train-label-folder", type=str, default="datasets/VOC/labels/train2012")
    parser.add_argument("--val-img-folder", type=str, default="datasets/VOC/images/val2012")
    parser.add_argument("--val-label-folder", type=str, default="datasets/VOC/labels/val2012")
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--checkpoints-root", type=str, default="checkpoints")
    parser.add_argument("--session-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--notes", type=str, default="changed LR to .0001, run for 200 epochs")
    parser.add_argument("--debug", action="store_true", default=False,help="Sanity checks for end to end pipeline")
    parser.add_argument("--force-new-training-session",action="store_true", default=False, help="Start new training session")
    parser.add_argument("--visual-interval", type=int, default=250)
    args = parser.parse_args()
    main(args)