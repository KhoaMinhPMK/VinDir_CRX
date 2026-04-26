from pathlib import Path
import os, textwrap

DATA_DIR = Path(
    "/workspace/Hokkaido_collaboration/km/lung_project/CRX/"
    "vinbigdata-chest-xray-abnormalities-detection"
).resolve()
os.chdir(DATA_DIR)

TRACKB_DIR = DATA_DIR / "trackB_cxr_dino"

BASE_CONFIG = TRACKB_DIR / "final_checkpoints/trackB_dino_fast1024_best_config.py"
BEST_CKPT = TRACKB_DIR / "final_checkpoints/trackB_dino_fast1024_best_mAP0247_epoch5.pth"

OUT_CONFIG = TRACKB_DIR / "detector_configs/cxr_dino_r50_full1024_finetune_from_fast_epoch5.py"
OUT_WORK_DIR = TRACKB_DIR / "detector_work_dirs/cxr_dino_r50_full1024_finetune_from_fast_epoch5"
RUN_SH = TRACKB_DIR / "detector_scripts/run_full1024_finetune_detector.sh"

assert BASE_CONFIG.exists(), BASE_CONFIG
assert BEST_CKPT.exists(), BEST_CKPT

OUT_WORK_DIR.mkdir(parents=True, exist_ok=True)
OUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
RUN_SH.parent.mkdir(parents=True, exist_ok=True)

s = BASE_CONFIG.read_text()

# Dùng full train thay vì fast subset
s = s.replace(
    "ann_file='annotations/train_fast_fold0_coco_1280.json'",
    "ann_file='annotations/train_fold0_coco_1280.json'"
)

# Fine-tune thêm 6 epochs
s = s.replace("max_epochs=5", "max_epochs=6")

# LR nhỏ hơn để fine-tune ổn định
s = s.replace(
    "optimizer=dict(type='AdamW', lr=5e-05, weight_decay=0.0001)",
    "optimizer=dict(type='AdamW', lr=1e-05, weight_decay=0.0001)"
)

# Scheduler mới cho 6 epochs
s = s.replace(
    "dict(type='MultiStepLR', begin=0, end=5, by_epoch=True, milestones=[3, 4], gamma=0.1)",
    "dict(type='MultiStepLR', begin=0, end=6, by_epoch=True, milestones=[4, 5], gamma=0.1)"
)

# Load từ best checkpoint hiện tại, không resume optimizer cũ
s = s.replace("load_from = None", f"load_from = r'{BEST_CKPT}'")
s = s.replace("resume = False", "resume = False")

# Work dir riêng, không đè run cũ
s = s.replace(
    "work_dir = r'/workspace/Hokkaido_collaboration/km/lung_project/CRX/vinbigdata-chest-xray-abnormalities-detection/trackB_cxr_dino/detector_work_dirs/cxr_dino_r50_fast1024_from1280_fold0'",
    f"work_dir = r'{OUT_WORK_DIR}'"
)

OUT_CONFIG.write_text(s)

run_text = f"""#!/usr/bin/env bash
set -e

cd "{DATA_DIR}"

python "{TRACKB_DIR}/mmdetection_py310/tools/train.py" "{OUT_CONFIG}" \\
  --work-dir "{OUT_WORK_DIR}"
"""

RUN_SH.write_text(run_text)
os.chmod(RUN_SH, 0o755)

print("=" * 80)
print("FULL 1024 FINE-TUNE CONFIG CREATED")
print("=" * 80)
print("Base config :", BASE_CONFIG)
print("Load from   :", BEST_CKPT)
print("New config  :", OUT_CONFIG)
print("Work dir    :", OUT_WORK_DIR)
print("Run script  :", RUN_SH)
print()
print("Next command:")
print(f"bash {RUN_SH}")