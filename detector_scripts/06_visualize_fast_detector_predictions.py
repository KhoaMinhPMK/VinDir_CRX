from pathlib import Path
import json, random
import cv2
import torch

from mmdet.apis import init_detector, inference_detector

DATA_DIR = Path("/workspace/Hokkaido_collaboration/km/lung_project/CRX/vinbigdata-chest-xray-abnormalities-detection")
CONFIG = DATA_DIR / "trackB_cxr_dino/final_checkpoints/trackB_dino_fast1024_best_config.py"
CKPT = DATA_DIR / "trackB_cxr_dino/final_checkpoints/trackB_dino_fast1024_best_mAP0247_epoch5.pth"
VAL_COCO = DATA_DIR / "trackB_cxr_dino/data/annotations/val_fold0_coco_1280.json"
VAL_IMG_DIR = DATA_DIR / "trackB_cxr_dino/data/images_1280/val"
OUT_DIR = DATA_DIR / "trackB_cxr_dino/visualizations/fast_detector_epoch5_val"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_THR = 0.25
N_IMAGES = 24
SEED = 42

print("=" * 80)
print("VISUALIZE FAST DINO DETECTOR")
print("=" * 80)
print("CONFIG :", CONFIG)
print("CKPT   :", CKPT)
print("VAL    :", VAL_COCO)
print("OUT    :", OUT_DIR)

with open(VAL_COCO, "r") as f:
    coco = json.load(f)

cats = {c["id"]: c["name"] for c in coco["categories"]}
label_names = [c["name"] for c in sorted(coco["categories"], key=lambda x: int(x["id"]))]
imgs = {img["id"]: img for img in coco["images"]}

anns_by_img = {}
for ann in coco["annotations"]:
    anns_by_img.setdefault(ann["image_id"], []).append(ann)

# ưu tiên ảnh có GT bbox để dễ nhìn
candidate_ids = [img_id for img_id, anns in anns_by_img.items() if len(anns) > 0]
random.seed(SEED)
sample_ids = random.sample(candidate_ids, min(N_IMAGES, len(candidate_ids)))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = init_detector(str(CONFIG), str(CKPT), device=device)

def draw_box(img, box, text, color, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        img, text, (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
    )

for idx, img_id in enumerate(sample_ids, 1):
    info = imgs[img_id]
    img_path = VAL_IMG_DIR / info["file_name"]
    img = cv2.imread(str(img_path))

    if img is None:
        print("Skip missing image:", img_path)
        continue

    vis = img.copy()

    # GT: green
    for ann in anns_by_img.get(img_id, []):
        x, y, bw, bh = ann["bbox"]
        cls_name = cats[ann["category_id"]]
        draw_box(vis, [x, y, x + bw, y + bh], f"GT {cls_name}", (0, 255, 0), 2)

    # Prediction: red
    result = inference_detector(model, str(img_path))
    pred = result.pred_instances

    bboxes = pred.bboxes.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()

    keep = scores >= SCORE_THR
    bboxes, labels, scores = bboxes[keep], labels[keep], scores[keep]

    # chỉ vẽ top 10 để ảnh không rối
    order = scores.argsort()[::-1][:10]

    for j in order:
        cls_name = label_names[int(labels[j])]
        score = float(scores[j])
        draw_box(vis, bboxes[j], f"P {cls_name} {score:.2f}", (0, 0, 255), 2)

    out_path = OUT_DIR / f"{idx:02d}_{info['file_name']}"
    cv2.imwrite(str(out_path), vis)
    print(f"[{idx:02d}/{len(sample_ids)}] saved:", out_path.name)

print("\nDone.")
print("Saved visualizations to:", OUT_DIR)