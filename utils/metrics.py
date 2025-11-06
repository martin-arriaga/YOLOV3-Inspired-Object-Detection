import torch
from torchvision.ops import nms
from debug_logger.debug import logger


'''
each out from yolo is in anchor relative coordinates we need to decode it 
predictions: tensor Bx3xSxSxnumofClasses+ 5 
'''
def DecodePredictions(pred_tensor, anchors, img_size=416):
    """
    predictions: tensor Bx3xSxSx(num_classes+5)
    anchors: tensor (3,2) width,height
    """
    '''
    B -> batch size 
    num_anchors -> number of anchors 3 in yolov3
    S -> grid size 13, 26, 52 depending on the scale 
    final shape : (B, 3, S , S , 5+C
    '''
    B, A, H, W, _ = pred_tensor.shape  # <- dynamic H, W

    device = pred_tensor.device

    # create grid dynamically
    yv, xv = torch.meshgrid([torch.arange(H), torch.arange(W)])
    grid = torch.stack((xv, yv), 2).float().to(device)  # [H,W,2]

    # extract predictions
    pred_tx = pred_tensor[..., 0]
    pred_ty = pred_tensor[..., 1]
    pred_tw = pred_tensor[..., 2]
    pred_th = pred_tensor[..., 3]
    pred_obj = pred_tensor[..., 4]
    pred_cls = pred_tensor[..., 5:]

    # apply grid
    boxes = torch.zeros(B, A, H, W, 4).to(device)
    boxes[..., 0:2] = torch.sigmoid(pred_tensor[..., 0:2]) + grid[None, None, :, :, :]
    boxes[..., 2:4] = torch.exp(pred_tensor[..., 2:4]) * anchors[None, :, None, None, :]

    return boxes, torch.sigmoid(pred_obj), torch.sigmoid(pred_cls)

def non_max_suppression(bounding_boxes, iou_threshold, confidence_threshold):
    # 1. Filter out boxes below confidence threshold
    bounding_boxes = [b for b in bounding_boxes if b[1] > confidence_threshold]

    final_boxes = []

    # Process per class
    classes = set([b[0] for b in bounding_boxes])
    for cls in classes:
        cls_boxes = [b for b in bounding_boxes if b[0] == cls]
        cls_boxes = sorted(cls_boxes, key=lambda x: x[1], reverse=True)

        while cls_boxes:
            chosen_box = cls_boxes.pop(0)
            final_boxes.append(chosen_box)

            cls_boxes = [
                b for b in cls_boxes
                if iou(chosen_box, b) < iou_threshold
            ]

    return final_boxes
#-----------------------
# Using TorchVisions NMS
#------------------------
def non_max_suppression_vectorized(bounding_boxes: torch.Tensor, iou_threshold, confidence_threshold)-> torch.Tensor:

    """
    :param bounding_boxes: list of [ b_idx ,x1, y1, x2, y2, conf, class]
    :param iou_threshold:
    :param confidence_threshold:
    :return:
    """
    #-----------
    # Edge case
    #-----------
    if bounding_boxes.numel() == 0:
        return torch.empty((0,7))
    #----Filter-Low-Confidence-Scores
    mask = bounding_boxes[:, 5] > confidence_threshold
    bounding_boxes = bounding_boxes[mask]

    boxes = bounding_boxes[:, 1:5] #--x1--y1--x2--y2----
    scores = bounding_boxes[:, 5] #---scores---
    classes = bounding_boxes[:, 6].long() #class
    b_idx = bounding_boxes[:, 0].long()

    #----boxes-to-keep----
    filtered = []
    for cls in classes.unique():
        cls_mask = classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        keep = nms(cls_boxes, cls_scores, iou_threshold)
        filtered.append(bounding_boxes[cls_mask][keep])

    if filtered:
        return torch.cat(filtered, dim=0)
    else:
       return torch.empty((0,7))

def iou(box1, box2):
    """
    Compute IoU between two YOLO-style bounding boxes.
    Format: [class, conf, x_center, y_center, w, h]
    """
    # Convert to corners
    x1_min = box1[2] - box1[4] / 2
    y1_min = box1[3] - box1[5] / 2
    x1_max = box1[2] + box1[4] / 2
    y1_max = box1[3] + box1[5] / 2

    x2_min = box2[2] - box2[4] / 2
    y2_min = box2[3] - box2[5] / 2
    x2_max = box2[2] + box2[4] / 2
    y2_max = box2[3] + box2[5] / 2

    # Intersection
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-6)

def iou_x1y1x2y2(box1, box2):
    """
    Compute IoU between two bounding boxes in the form [x1, y1, x2, y2].
    :param box1:
    :param box2:
    :return:
    """

    inter_x1 = torch.max(box1[0], box2[:, 0])
    inter_y1 = torch.max(box1[1], box2[:, 1])
    inter_x2 = torch.min(box1[2], box2[:, 2])
    inter_y2 = torch.min(box1[3], box2[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter_area + 1e-6

    return inter_area / union + 1e-6


def DecodePredictionsBatch(predictions, anchors, img_size=416):
    """
    predictions: tensor (B, A, S, S, 5+C)
    anchors: tensor (A,2)
    Returns:
    --------
    boxes: (B, A*S*S, 4)  # x_center, y_center, w, h normalized
    obj:  (B, A*S*S)      # objectness
    cls_probs: (B, A*S*S, C) # class probabilities
    """

    B, A, S, _, columns = predictions.shape
    C = predictions.shape[-1] - 5
    logger.debug(f"\n[DEBUG] Predictions shape:{predictions.shape}\n"
                 f"\n[DEBUG] ----Breakdown of the shape  ----\n")

    logger.debug(f"\n[DEBUG] There are 8 images in a batch [{B}, _, _, _, _]: \n"
                 f"\n[DEBUG] NUmber of anchors: [-, {A}, _ , _, _]\n"
                 f"\n[DEBUG] Grid Size [-, , {S},{S}, _ ]\n"
                 f"\n[DEBUG] Each Prediction vector has 25 columns [_, _, _, _, {columns}]\n"
                 f"\n[DEBUG] Prediction vector layout: [ tx, ty, tw, th, obj, c0, c1, ..., c19 ]\n"
                 f"\n[DEBUG] Where tx , ty are the offsets for box center\n"
                 f"\n[DEBUG] tw, th are the width and height offsets of box\n"
                 f"\n[DEBUG]obj is the objectness score (object or no object )\n"
                 f"\n[DEBUG] c0 -c19 are the probabilities for each class\n"
                 f"\n---------------------------------------------------"
                )

    logger.debug(f"\n[DEBUG] Anchors shape:{anchors.shape}")


    predictions = predictions.clone()
    predictions[..., 0:2] = torch.sigmoid(predictions[..., 0:2])  # tx, ty
    predictions[..., 4] = torch.sigmoid(predictions[..., 4])  # objectness
    predictions[..., 5:] = torch.sigmoid(predictions[..., 5:])  # class prob
    logger.debug(f"")
    # --------------
    # Sanity checks
    # --------------
    logger.debug(f"\n[DEBUG] tx min/max:{predictions[...,0].min().item()}/{predictions[...,0].max().item()}, should be between [0,1]")
    logger.debug(f"\n[DEBUG] ty min/max:{predictions[...,1].min().item()}/{predictions[...,1].max().item()}, should be between [0,1]")
    logger.debug(f"\n[DEBUG] Objectness min/max:{predictions[...,4].min().item()}/{predictions[...,4].max().item()}, should be between [0,1]")
    logger.debug(f"\n[DEBUG] tx Class Probability min/max:{predictions[...,5].min().item()}/{predictions[...,5].max().item()}, should be between [0,1]")
    # Build grid
    grid_y, grid_x = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).float()  # S, S, 2
    grid = grid.to(predictions.device)


    # Expand grid and anchors
    grid = grid[None, None, :, :, :]  # 1x1xSxSx2
    anchors = anchors[None, :, None, None, :].float()  # 1xAx1x1x2
    # --------------
    # Sanity checks
    # --------------
    logger.debug(f"\n[DEBUG] grid shape:{grid.shape}")
    logger.debug(f"[DEBUG] grid sample:{grid[0,0,0:3,0:3,:]}")

    # Decode positions
    # --------------
    # Sanity checks
    # --------------
    boxes = predictions[..., :4].clone()
    logger.debug(f"\n[DEBUG] tw/th min/max BEFORE exp:{boxes.min().item()}/{boxes.max().item()}")

    boxes[..., :2] += grid  # add offsets
    boxes[..., :2] /= S  # normalize to [0,1]
    boxes[..., 2:4] = (torch.exp(boxes[..., 2:4]) * anchors)

    logger.debug(f"\n[DEBUG] tw/th min/max AFTER exp:{boxes.min().item()}/{boxes.max().item()}")
    logger.debug(f"\n[DEBUG] boxes[0,0,0,0,4]-> , x_center, y_center, w, h:{boxes[0, 0,0, 0:4]}")
    logger.debug(f"\n[DEBUG] boxes min/max:{boxes.min().item()}, {boxes.max().item()}")

    # Flatten grid & anchors
    boxes = boxes.reshape(B, -1, 4)  # B, A*S*S, 4
    obj = predictions[..., 4].reshape(B, -1)  # B, A*S*S
    cls_probs = predictions[..., 5:].reshape(B, -1, C)  # B, A*S*S, C
    # --------------
    # Sanity checks
    # --------------
    logger.debug(f"\n[DEBUG] boxes_flat.shape: {boxes.shape}")
    logger.debug(f"\n[DEBUG] boxes_flat[0,0,:]:{boxes[0, 0, :]}")


    return boxes, obj, cls_probs
