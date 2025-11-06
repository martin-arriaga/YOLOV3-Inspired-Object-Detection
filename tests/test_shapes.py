import torch

from models.yolo import YOLOv3
from utils.metrics import DecodePredictionsBatch# using your implementation

# -----------------------------
# Fake Decode (stand-in)
# -----------------------------
def fake_decode(pred, anchors, img_size=416):
    """
    Simulate DecodePredictionsBatch:
    - pred: (B,3,S,S,25)
    - anchors: (3,2)
    """
    B, A, S, _, C = pred.shape
    # flatten anchors and grid
    N = A * S * S
    boxes = torch.rand(B, N, 4)       # cx, cy, w, h in [0,1]
    obj   = torch.rand(B, N)          # objectness
    cls   = torch.rand(B, N, 20)      # fake class probs
    return boxes, obj, cls


# -----------------------------
# Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 20
img_size = 416

anchors = [
    torch.tensor([[116, 90], [156, 198], [373, 326]]),  # 13x13
    torch.tensor([[30, 61], [62, 45], [59, 119]]),      # 26x26
    torch.tensor([[10, 13], [16, 30], [33, 23]])        # 52x52
]

# -----------------------------
# Model + Input
# -----------------------------
model = YOLOv3(num_classes=num_classes).to(device)
model.eval()

x = torch.randn(1, 3, img_size, img_size).to(device)

with torch.no_grad():
    outputs = model(x)

print("Input:", x.shape)
for i, out in enumerate(outputs):
    print(f"Scale {i}: {out.shape}")

# -----------------------------
# Decode each scale (fake)
# -----------------------------
print("==== Sanity Check: Decoded Boxes ====")
num_print = 5  # number of boxes to inspect per scale

for scale_idx, output in enumerate(outputs):
    boxes, obj, cls_probs = DecodePredictionsBatch(output, anchors[scale_idx], img_size=img_size)
    B, N, C = boxes.shape[0], boxes.shape[1], cls_probs.shape[-1]
    print(f"[Scale {boxes.shape[1]} boxes] boxes={boxes.shape}, obj={obj.shape}, cls={cls_probs.shape}")

    for b in range(B):
        print(f"-- Batch {b} --")
        for n in range(min(num_print, N)):
            x, y, w, h = boxes[b, n]
            conf = obj[b, n].item()
            top_cls = torch.argmax(cls_probs[b, n]).item()
            top_cls_prob = cls_probs[b, n, top_cls].item()
            print(f"Box {n}: x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}, obj={conf:.3f}, class={top_cls}({top_cls_prob:.3f})")
    print("---------------------------------------------------")