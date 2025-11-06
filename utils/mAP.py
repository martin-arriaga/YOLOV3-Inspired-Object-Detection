
import torch
from debug_logger.debug import logger

@torch.no_grad()
def my_mAP(predictions , ground_truth, iou_threshold, num_classes, device=None):
    """

    :param predictions: list of predictions [x1, y1, x2, y2, conf, class, image_idx]
    :param ground_truth: list of ground truth [img_idx, class, x1, y1, x2, y2]
    :param iou_threshold:
    :param num_classes:
    :return:
    """
    if len(predictions) == 0:
        return 0.0
    #---P: Total predicted boxes after NMS---
    #---G: Total Ground Truth boxes---
    preds = torch.stack(predictions)
    gts = torch.stack(ground_truth)

    if device is None:
        device = preds.device

    preds = preds.to(device)
    gts = gts.to(device)

    ap = []

    for cls in range(num_classes):
        #---Class--- [P] boolean mask
        cls_mask_pred = preds[:, 5] == cls
        #---Class--- [G] boolean mask
        cls_mask_gt = gts[:, 1] == cls
        #---Predicted boxes for this class---
        cls_pred = preds[cls_mask_pred]
        cls_pred= cls_pred[cls_pred[:,4] > .05]
        #---Ground truth boxes for this class---
        cls_gt = gts[cls_mask_gt]

        if cls_pred.shape[0] == 0 and cls_gt.shape[0] == 0:
            continue

        if cls_pred.shape[0] == 0:
            ap.append(0.0)
            continue

        if cls_gt.shape[0] == 0:
            ap.append(0.0)
            continue

        # Sort predictions by confidence
        cls_pred = cls_pred[cls_pred[:, 4].argsort(descending=True)]

        tp = torch.zeros(cls_pred.shape[0], device=device)
        fp = torch.zeros(cls_pred.shape[0], device=device)

        image_indexes = cls_gt[:,0].unique()
        gt_detected = {int(i.item()): torch.zeros((cls_gt[cls_gt[:, 0] == i].shape[0]), dtype=torch.bool, device=device)
                       for i in image_indexes}
        for i, pred in enumerate(cls_pred):
            assert cls_pred.shape[1] == 7
            pred_img_idx = int(pred[-1].item())
            pred_box = pred[:4]

            gt_for_image = cls_gt[cls_gt[:, 0] == pred_img_idx]

            if gt_for_image.shape[0] == 0:
                fp[i] = 1
                continue
            gt_boxes = gt_for_image[:, 2:6]

            inter_x1 = torch.max(pred_box[0], gt_boxes[:, 0])
            inter_y1 = torch.max(pred_box[1], gt_boxes[:, 1])
            inter_x2 = torch.min(pred_box[2], gt_boxes[:, 2])
            inter_y2 = torch.min(pred_box[3], gt_boxes[:, 3])

            inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
            area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            ious = inter_area / (area_pred + area_gt - inter_area + 1e-6)

            iou_max, gt_idx = torch.max(ious, dim=0)
            # Check for valid match
            if iou_max > iou_threshold and not gt_detected[pred_img_idx][gt_idx]:
                tp[i] = 1
                gt_detected[pred_img_idx][gt_idx] = True
            else:
                fp[i] = 1
        # Compute cumulative sums
        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        num_gt = sum(len(v) for v in gt_detected.values())
        recalls = tp_cum / num_gt + 1e-6
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        # interpolation
        recalls = torch.cat([torch.tensor([0.0], device=device), recalls, torch.tensor([1.0], device=device)])
        precisions = torch.cat([torch.tensor([0.0], device=device), precisions, torch.tensor([0.0], device=device)])
        for j in range(precisions.shape[0] - 2, -1, -1):
            precisions[j] = torch.maximum(precisions[j], precisions[j + 1])

        ap.append(torch.sum((recalls[1:] - recalls[:-1]) * precisions[1:]).item())

    if len(ap) == 0:
        return 0.0

    return sum(ap) / len(ap)

def my_mAP_vectorized(predictions , ground_truth, iou_threshold, num_classes, device=None):
    """

    :param predictions: list of predictions [x1, y1, x2, y2, conf, class, image_idx]
    :param ground_truth: list of ground truth [img_idx, class, x1, y1, x2, y2]
    :param iou_threshold:
    :param num_classes:
    :return mean Average Precision

    """

    if len(predictions) == 0:
        return 0.0

    preds = torch.stack(predictions).to(device)
    gts = torch.stack(ground_truth).to(device)

    ap_list = []

    for cls in range(num_classes):
        # --- filter predictions and GTs by class ---
        cls_pred = preds[preds[:, 5] == cls]
        cls_gt = gts[gts[:, 1] == cls]

        if cls_pred.shape[0] == 0 and cls_gt.shape[0] == 0:
            continue
        if cls_pred.shape[0] == 0 or cls_gt.shape[0] == 0:
            ap_list.append(0.0)
            continue

        # --- sort predictions by confidence ---
        cls_pred = cls_pred[cls_pred[:, 4].argsort(descending=True)]

        pred_boxes = cls_pred[:, :4]          # [num_preds, 4]
        pred_img_idx = cls_pred[:, 6].long()  # [num_preds]

        gt_boxes = cls_gt[:, 2:6]             # [num_gts, 4]
        gt_img_idx = cls_gt[:, 0].long()      # [num_gts]

        # --- compute IoU matrix between preds and GTs ---
        # expand dims for broadcasting
        pred_boxes_exp = pred_boxes.unsqueeze(1)  # [num_preds, 1, 4]
        gt_boxes_exp = gt_boxes.unsqueeze(0)      # [1, num_gts, 4]

        inter_x1 = torch.max(pred_boxes_exp[..., 0], gt_boxes_exp[..., 0])
        inter_y1 = torch.max(pred_boxes_exp[..., 1], gt_boxes_exp[..., 1])
        inter_x2 = torch.min(pred_boxes_exp[..., 2], gt_boxes_exp[..., 2])
        inter_y2 = torch.min(pred_boxes_exp[..., 3], gt_boxes_exp[..., 3])

        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area_pred = (pred_boxes_exp[..., 2] - pred_boxes_exp[..., 0]) * (pred_boxes_exp[..., 3] - pred_boxes_exp[..., 1])
        area_gt = (gt_boxes_exp[..., 2] - gt_boxes_exp[..., 0]) * (gt_boxes_exp[..., 3] - gt_boxes_exp[..., 1])
        iou_matrix = inter_area / (area_pred + area_gt - inter_area + 1e-6)  # [num_preds, num_gts]

        # --- mask out GTs from different images ---
        img_mask = (pred_img_idx.unsqueeze(1) == gt_img_idx.unsqueeze(0))  # [num_preds, num_gts]
        iou_matrix = iou_matrix * img_mask.float()

        # --- initialize TP/FP trackers ---
        gt_detected = torch.zeros(cls_gt.shape[0], dtype=torch.bool, device=device)
        tp = torch.zeros(cls_pred.shape[0], device=device)
        fp = torch.zeros(cls_pred.shape[0], device=device)

        for i in range(cls_pred.shape[0]):
            ious = iou_matrix[i]
            iou_max, gt_idx = ious.max(dim=0)

            if iou_max > iou_threshold and not gt_detected[gt_idx]:
                tp[i] = 1
                gt_detected[gt_idx] = True
            else:
                fp[i] = 1

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recalls = tp_cum / (cls_gt.shape[0] + 1e-6)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        #AP
        recalls = torch.cat([torch.tensor([0.0], device=device), recalls, torch.tensor([1.0], device=device)])
        precisions = torch.cat([torch.tensor([0.0], device=device), precisions, torch.tensor([0.0], device=device)])
        for j in range(precisions.shape[0] - 2, -1, -1):
            precisions[j] = torch.maximum(precisions[j], precisions[j + 1])

        ap_list.append(torch.sum((recalls[1:] - recalls[:-1]) * precisions[1:]).item())

    if len(ap_list) == 0:
        return 0.0
    return sum(ap_list) / len(ap_list)
