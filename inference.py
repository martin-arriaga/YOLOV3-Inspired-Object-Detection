import torch
from utils.metrics import DecodePredictions, non_max_suppression


def infer_image(image_tensor, model, anchors, img_size=416, conf_threshold=0.5, iou_threshold=0.3):
    """
    Runs inference on a single image or a batch of images.

    Parameters:
    -----------
    image_tensor: torch.Tensor
        Shape: (B, C, H, W), normalized [0,1]
    model: torch.nn.Module
        YOLOv3 model
    anchors: list of torch.Tensor
        List of 3 tensors, each (3,2) for each scale
    img_size: int
        Input image size (default 416)
    conf_threshold: float
        Minimum objectness confidence to keep a box
    iou_threshold: float
        IoU threshold for NMS

    Returns:
    --------
    list of lists
        Final bounding boxes per image. Each box: [class_idx, conf, x, y, w, h]
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)

    final_boxes_batch = []

    # Loop over batch
    for batch_idx in range(image_tensor.shape[0]):
        all_boxes = []

        # Decode each scale
        for scale_idx, output in enumerate(outputs):
            boxes, obj, cls_probs = DecodePredictions(output[batch_idx:batch_idx + 1],
                                                      anchors[scale_idx],
                                                      img_size=img_size)
            B, A, S, _, _ = boxes.shape

            for a in range(A):
                for i in range(S):
                    for j in range(S):
                        conf = obj[0, a, i, j].item()
                        class_idx = torch.argmax(cls_probs[0, a, i, j]).item()
                        if conf > conf_threshold:
                            x, y, w, h = boxes[0, a, i, j]
                            all_boxes.append([class_idx, conf, x.item(), y.item(), w.item(), h.item()])

        # Apply NMS
        final_boxes = non_max_suppression(all_boxes, iou_threshold=iou_threshold, confidence_threshold=conf_threshold)
        final_boxes_batch.append(final_boxes)

    return final_boxes_batch
