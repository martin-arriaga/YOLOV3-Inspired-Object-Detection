import torch
from debug_logger.debug import logger

def encode_yolo_targets(target_boxes, target_labels, anchors, strides, num_classes, device):
    """
    Encode ground truth for all YOLO scales.

    Args:
        target_boxes: list of tensors [(num_objs, 4), ...] per image
                      Format: (x_center, y_center, w, h), normalized [0,1]
        target_labels: list of tensors [(num_objs,), ...] per image (class indices)
        anchors: list of lists, 3 scales:
                 [
                   [(w,h), (w,h), (w,h)],  # for 13x13
                   [(w,h), (w,h), (w,h)],  # for 26x26
                   [(w,h), (w,h), (w,h)],  # for 52x52
                 ]

        Anchors must be normalized relative to image size
        strides: list of scales [13, 26, 52]
        num_classes: number of classes
        device: torch.device

    Returns:
        List of 3 tensors for each scale:
        [
          (B, A, S, S, 5+num_classes),  # scale 13 large
          (B, A, S, S, 5+num_classes),  # scale 26 medium
          (B, A, S, S, 5+num_classes),  # scale 52 small

        ]
        inside each tensor:
        target[b, anchor_indx, i, j, :]
        where :
        b is which image in the batch
        anchor_indx is the index of the anchor
        i and j are the index of the box in the grid
        : [tx, ty, tw, th, objectness, one hot vector for classes ]
        tx, ty, tw, th = target_tensor[0, 2, 4, 5, :4]
        think: first image, stride 3 ie 52x52, row 4 column 5, up to column 4 of the vector
        retrieves -> [ tx, ty, tw, th]

    """
    B = len(target_boxes)
    all_scales = []

    for scale_idx, S in enumerate(strides):
        A = len(anchors[scale_idx])
        target_tensor = torch.zeros((B, A, S, S, 6 + num_classes), device=device)

        for b in range(B):
            boxes = target_boxes[b]
            labels = target_labels[b]

            for box, cls in zip(boxes, labels):
                x, y, w, h = box
                i, j = int(S * y), int(S * x)  # grid cell indices

                if i >= S or j >= S:
                    continue

                #---IoU-with-anchors---
                ious = []
                for aw, ah in anchors[scale_idx]:
                    inter = torch.min(w, aw) * torch.min(h, ah)
                    union = (w * h) + (aw * ah) - inter
                    ious.append(inter / (union + 1e-6))
                best_anchor = torch.argmax(torch.tensor(ious, device=device))
                best_iou = ious[best_anchor]

                #---offsets---
                tx = S * x - j
                ty = S * y - i
                tw = torch.log((w / anchors[scale_idx][best_anchor][0]) + 1e-6)
                th = torch.log((h / anchors[scale_idx][best_anchor][1]) + 1e-6)

                #--Fill---
                current_best_match = target_tensor[b, best_anchor, i, j, 4]
                current_best_iou = target_tensor[b, best_anchor, i, j, -1]
                if current_best_match == 0 or best_iou > current_best_iou:
                    target_tensor[b, best_anchor, i, j, 0] = tx
                    target_tensor[b, best_anchor, i, j, 1] = ty
                    target_tensor[b, best_anchor, i, j, 2] = tw
                    target_tensor[b, best_anchor, i, j, 3] = th
                    target_tensor[b, best_anchor, i, j, 4] = 1.0
                    target_tensor[b, best_anchor, i, j, 5 + cls] = 1.0
                    target_tensor[b, best_anchor, i, j, -1] = best_iou
                    logger.debug(f"Format tx, ty, tw, th : {tx}, {ty}, {tw}, {th}, {cls}")
        target_tensor = target_tensor[...,:-1]
        all_scales.append(target_tensor)

    return all_scales
