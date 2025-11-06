import torch
import torch.nn as nn
from debug_logger.debug import logger, set_debug

class YoloLoss(nn.Module):
    def __init__(self, anchors, num_classes, imgsize):
        super(YoloLoss, self).__init__()
        self.mseloss = nn.MSELoss()
        self.bceloss = nn.BCEWithLogitsLoss()
        self.numofclasses = num_classes
        self.imgsize = imgsize
        self.anchors = anchors
        self.lmda_cls = 1.0
        self.lmda_obj = 1.0
        self.lmda_no_obj = .5
        self.lmda_box = 10

    #--- tx,ty,tw,th ---> xcenter, ycenter, width, height (normalized)---
    def box_coordinates_center(self, tx, ty, tw, th, grid_x, grid_y, grid_size, anchors):
        x= (torch.sigmoid(tx) + grid_x) / grid_size
        y = (torch.sigmoid(ty) + grid_y) / grid_size
        w = torch.exp(tw) * anchors[...,0] / grid_size
        h = torch.exp(th) * anchors[...,1] / grid_size
        decoded_boxes = torch.stack([x, y, w, h], dim=-1)
        return decoded_boxes

    #--- xcenter, ycenter, h, w ---> x1, y1, x2, y1---
    def box_coordinates_corner(self, some_box):
        x1 = some_box[..., 0] - some_box[..., 2]/2 #---x_center- w / 2---
        y1 = some_box[..., 1] - some_box[..., 3]/2 #---y_center - h / 2---
        x2 = some_box[..., 0] + some_box[..., 2]/2
        y2 = some_box[..., 1] + some_box[..., 3]/2

        return x1, y1, x2, y2


    #---Generalized-IoU----
    def giou(self, predicted_boxes, target_boxes):
        pbox_x1, pbox_y1, pbox_x2, pbox_y2 = self.box_coordinates_corner(predicted_boxes)
        tbox_x1, tbox_y1, tbox_x2, tbox_y2 = self.box_coordinates_corner(target_boxes)

        intersection_x1 = torch.max(pbox_x1, tbox_x1)
        intersection_y1 = torch.max(pbox_y1, tbox_y1)
        intersection_x2 = torch.min(pbox_x2, tbox_x2)
        intersection_y2 = torch.min(pbox_y2, tbox_y2)
        intersection_h = (intersection_y2 - intersection_y1).clamp(min=0)
        intersection_w = (intersection_x2 - intersection_x1).clamp(min=0)
        intersection_area = intersection_w * intersection_h

        predicted_area = (pbox_x2 - pbox_x1) * (pbox_y2 - pbox_y1)
        target_area = (tbox_x2 - tbox_x1) * (tbox_y2 - tbox_y1)
        union_area = predicted_area + target_area - intersection_area
        iou = intersection_area / (union_area + 1e-5)

        enclosed_x1 = torch.min(pbox_x1, tbox_x1)
        enclosed_y1 = torch.min(pbox_y1, tbox_y1)
        enclosed_x2 = torch.max(pbox_x2, tbox_x2)
        enclosed_y2 = torch.max(pbox_y2, tbox_y2)
        enclosed_area = (enclosed_x2 - enclosed_x1)* (enclosed_y2 - enclosed_y1)
        giou = iou - (enclosed_area - union_area) / (enclosed_area + 1e-5)
        return giou

    def forward(self, prediction, target, anchors):
        """

        :param prediction: [tx, ty, tw,th,cls+5] --raw-predictions---
        :param target: [[tx, ty, tw,th, objectness ,cls+5]---raw-targets---
        :param anchors:
        :return:
        """

        logger.debug(f"\n[DEBUG] prediction shape: {prediction.shape}")
        logger.debug(f"\n[DEBUG] target shape: {target.shape}")

        assert prediction.shape == target.shape , "Prediction and target shape mismatch"
        B , numofanchors, S , _, _ = prediction.shape
        #---Raw Predictions---
        pred_tx = prediction[..., 0]
        pred_ty = prediction[..., 1]
        pred_tw = prediction[..., 2]
        pred_th = prediction[..., 3]
        pred_obj = prediction[..., 4]
        pred_cls = prediction[..., 5:]

        #---Raw Targets---
        target_tx = target[..., 0]
        target_ty = target[..., 1]
        target_tw = target[..., 2]
        target_th = target[..., 3]

        object_mask = target[..., 4] == 1
        noobject_mask = target[..., 4] == 0

        #---Build-Grid and Prep---
        grid_y, grid_x = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
        grid_y = grid_y.to(prediction.device).float().view(1,1,S,S)
        grid_x= grid_x.to(prediction.device).float().view(1,1,S,S)
        anchors = anchors.to(prediction.device)
        anchors = anchors.view(1, numofanchors, 1, 1, 2)

        #---Decode Raw outputs---
        """
        [B, num_anchors, S, S 4] 
        Batch 0:
            Grid cell:
                0,0
                    anchor 0 
                    [ x_center, y_center, width, height] 
        
        """
        decoded_target_boxes = self.box_coordinates_center(tx=target_tx, ty=target_ty,  tw=target_tw,
                                                    th=target_th,grid_x=grid_x, grid_y=grid_y,
                                                    grid_size=S, anchors=anchors)
        decoded_predicted_boxes = self.box_coordinates_center(tx=pred_tx,ty=pred_ty, tw=pred_tw,
                                                    th=pred_th, grid_x=grid_x, grid_y=grid_y,
                                                    grid_size=S,anchors=anchors)

        predicted_boxes_with_obj = decoded_predicted_boxes[object_mask]
        target_boxes_with_obj = decoded_target_boxes[object_mask]
        # --------
        # Losses
        # --------
        if predicted_boxes_with_obj.numel() > 0:
            giou = self.giou(predicted_boxes_with_obj, target_boxes_with_obj)
            loss_box = (1 - giou).mean()
        else:
            loss_box = torch.tensor(0.0, device=prediction.device)
        #---Class---
        target_cls = target[..., 5:]

        # Objectness loss
        loss_obj = self.bceloss(pred_obj[object_mask], torch.ones_like(pred_obj[object_mask]))
        loss_noobj = self.bceloss(pred_obj[noobject_mask], torch.zeros_like(pred_obj[noobject_mask]))

        # Classification loss
        if target_cls[object_mask].numel() > 0:
            loss_cls = self.bceloss(pred_cls[object_mask], target_cls[object_mask])
        else:
            loss_cls = torch.tensor(0.).to(prediction.device)
        # Total loss
        total_loss = (
                self.lmda_box * loss_box
                + self.lmda_obj * loss_obj
                + self.lmda_no_obj * loss_noobj
                + self.lmda_cls * loss_cls
        )

        return total_loss






