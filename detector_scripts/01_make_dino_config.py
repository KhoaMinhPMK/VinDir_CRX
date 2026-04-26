# ============================================================
# 01_make_dino_config.py
# Create DINO detector config for VinDr-CXR 1280
# This file is external to notebook.
# Run with:
#   conda activate cxr_mmdet310
#   python trackB_cxr_dino/detector_scripts/01_make_dino_config.py
# ============================================================

from pathlib import Path
import os
import json
import shutil
import subprocess
import textwrap

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

DATA_DIR = Path(
    "/workspace/Hokkaido_collaboration/km/lung_project/CRX/"
    "vinbigdata-chest-xray-abnormalities-detection"
).resolve()

os.chdir(DATA_DIR)

TRACKB_DIR = DATA_DIR / "trackB_cxr_dino"
TRACKB_DATA_DIR = TRACKB_DIR / "data"
TRACKB_ANN_DIR = TRACKB_DATA_DIR / "annotations"

DETECTOR_SCRIPT_DIR = TRACKB_DIR / "detector_scripts"
DETECTOR_CONFIG_DIR = TRACKB_DIR / "detector_configs"
DETECTOR_WORK_DIR = TRACKB_DIR / "detector_work_dirs"

MMDET_SRC_DIR = TRACKB_DIR / "mmdetection_py310"

IMG_SIZE = 1280
FOLD = 0

TRAIN_IMG_DIR = TRACKB_DATA_DIR / f"images_{IMG_SIZE}" / "train"
VAL_IMG_DIR = TRACKB_DATA_DIR / f"images_{IMG_SIZE}" / "val"

TRAIN_COCO = TRACKB_ANN_DIR / f"train_fold{FOLD}_coco_{IMG_SIZE}.json"
VAL_COCO = TRACKB_ANN_DIR / f"val_fold{FOLD}_coco_{IMG_SIZE}.json"

DEBUG_TRAIN_COCO = TRACKB_ANN_DIR / f"train_debug_fold{FOLD}_coco_{IMG_SIZE}.json"
DEBUG_VAL_COCO = TRACKB_ANN_DIR / f"val_debug_fold{FOLD}_coco_{IMG_SIZE}.json"

EXPERIMENT_NAME = f"cxr_dino_r50_{IMG_SIZE}_fold{FOLD}"
CUSTOM_CONFIG = DETECTOR_CONFIG_DIR / f"{EXPERIMENT_NAME}.py"

DEBUG_WORK_DIR = DETECTOR_WORK_DIR / f"{EXPERIMENT_NAME}_debug"
MAIN_WORK_DIR = DETECTOR_WORK_DIR / EXPERIMENT_NAME

DEBUG_RUN_SH = DETECTOR_SCRIPT_DIR / "run_debug_detector.sh"
MAIN_RUN_SH = DETECTOR_SCRIPT_DIR / "run_main_detector.sh"

for p in [
    DETECTOR_SCRIPT_DIR,
    DETECTOR_CONFIG_DIR,
    DETECTOR_WORK_DIR,
    DEBUG_WORK_DIR,
    MAIN_WORK_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)

assert TRAIN_IMG_DIR.exists(), f"Missing: {TRAIN_IMG_DIR}"
assert VAL_IMG_DIR.exists(), f"Missing: {VAL_IMG_DIR}"
assert TRAIN_COCO.exists(), f"Missing: {TRAIN_COCO}"
assert VAL_COCO.exists(), f"Missing: {VAL_COCO}"

print("=" * 80)
print("CREATE DINO DETECTOR CONFIG")
print("=" * 80)
print(f"DATA_DIR        : {DATA_DIR}")
print(f"TRAIN_IMG_DIR   : {TRAIN_IMG_DIR}")
print(f"VAL_IMG_DIR     : {VAL_IMG_DIR}")
print(f"TRAIN_COCO      : {TRAIN_COCO}")
print(f"VAL_COCO        : {VAL_COCO}")
print(f"MMDET_SRC_DIR   : {MMDET_SRC_DIR}")
print(f"CUSTOM_CONFIG   : {CUSTOM_CONFIG}")

# ------------------------------------------------------------
# 2. Check / clone MMDetection source
# ------------------------------------------------------------

if not MMDET_SRC_DIR.exists():
    print("\nMMDetection source not found. Cloning...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "-b",
            "3.x",
            "https://github.com/open-mmlab/mmdetection.git",
            str(MMDET_SRC_DIR),
        ],
        check=True,
    )
else:
    print("\nMMDetection source already exists.")

assert (MMDET_SRC_DIR / "configs").exists(), "Missing mmdetection configs folder."
assert (MMDET_SRC_DIR / "tools" / "train.py").exists(), "Missing mmdetection tools/train.py."

TRAIN_SCRIPT = MMDET_SRC_DIR / "tools" / "train.py"

# ------------------------------------------------------------
# 3. Choose base DINO config
# ------------------------------------------------------------

BASE_CONFIG = MMDET_SRC_DIR / "configs" / "dino" / "dino-4scale_r50_8xb2-12e_coco.py"

assert BASE_CONFIG.exists(), f"Missing base config: {BASE_CONFIG}"

print(f"\nBase config: {BASE_CONFIG}")

# ------------------------------------------------------------
# 4. Load categories/classes from COCO
# ------------------------------------------------------------

with open(TRAIN_COCO, "r") as f:
    train_coco = json.load(f)

with open(VAL_COCO, "r") as f:
    val_coco = json.load(f)

categories = sorted(train_coco["categories"], key=lambda x: int(x["id"]))
CLASS_NAMES = [cat["name"] for cat in categories]
NUM_CLASSES = len(CLASS_NAMES)

assert NUM_CLASSES == 14, f"Expected 14 classes, got {NUM_CLASSES}"

print("\nClasses:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i:2d}: {name}")

# ------------------------------------------------------------
# 5. Create debug COCO subsets
# ------------------------------------------------------------

def make_debug_coco(src_json, dst_json, n_images):
    with open(src_json, "r") as f:
        coco = json.load(f)

    images = coco["images"][:n_images]
    keep_ids = set(img["id"] for img in images)

    anns = [
        ann for ann in coco["annotations"]
        if ann["image_id"] in keep_ids
    ]

    out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": anns,
        "categories": coco["categories"],
    }

    with open(dst_json, "w") as f:
        json.dump(out, f)

    print(f"\nSaved debug COCO: {dst_json}")
    print(f"Images      : {len(images):,}")
    print(f"Annotations : {len(anns):,}")

make_debug_coco(TRAIN_COCO, DEBUG_TRAIN_COCO, n_images=128)
make_debug_coco(VAL_COCO, DEBUG_VAL_COCO, n_images=64)

# ------------------------------------------------------------
# 6. Write custom MMDetection config
# ------------------------------------------------------------

classes_tuple = "(" + ", ".join([repr(x) for x in CLASS_NAMES]) + ",)"

config_text = f"""
# Auto-generated DINO config for VinDr-CXR
# Advanced detector branch
# Data: 1280 letterbox PNG + COCO 1280
# Base: {BASE_CONFIG}

_base_ = r'{BASE_CONFIG}'

custom_imports = dict(imports=['mmdet'], allow_failed_imports=False)

data_root = r'{TRACKB_DATA_DIR}/'
metainfo = dict(classes={classes_tuple})
num_classes = {NUM_CLASSES}

# IMPORTANT:
# Images are already 1280x1280 letterbox PNGs.
# Resize is still added to create scale_factor metadata for DINO validation/prediction.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=({IMG_SIZE}, {IMG_SIZE}), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, color_type='color'),
    dict(type='Resize', scale=({IMG_SIZE}, {IMG_SIZE}), keep_ratio=False),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/train_fold{FOLD}_coco_{IMG_SIZE}.json',
        data_prefix=dict(img='images_{IMG_SIZE}/train/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=train_pipeline,
        backend_args=None
    )
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/val_fold{FOLD}_coco_{IMG_SIZE}.json',
        data_prefix=dict(img='images_{IMG_SIZE}/val/'),
        metainfo=metainfo,
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=test_pipeline,
        backend_args=None
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + 'annotations/val_fold{FOLD}_coco_{IMG_SIZE}.json',
    metric='bbox',
    iou_thrs=[0.4],
    classwise=False,
    format_only=False,
    backend_args=None
)

test_evaluator = val_evaluator

# Change COCO 80 classes to VinDr 14 abnormal classes.
model = dict(
    bbox_head=dict(num_classes=num_classes)
)

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# A100 40GB safe start:
# Real batch = 1, effective batch = 4 via gradient accumulation.
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='float16',
    accumulative_counts=4,
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        rule='greater'
    )
)

env_cfg = dict(cudnn_benchmark=True)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

work_dir = r'{MAIN_WORK_DIR}'

load_from = None
resume = False
auto_scale_lr = dict(enable=False)
"""

CUSTOM_CONFIG.write_text(textwrap.dedent(config_text).strip() + "\n")

print(f"\nSaved custom config: {CUSTOM_CONFIG}")

# ------------------------------------------------------------
# 7. Create debug and main run scripts
# ------------------------------------------------------------

debug_cmd = f"""#!/usr/bin/env bash
set -e

cd "{DATA_DIR}"

python "{TRAIN_SCRIPT}" "{CUSTOM_CONFIG}" \\
  --work-dir "{DEBUG_WORK_DIR}" \\
  --cfg-options \\
    train_cfg.max_epochs=1 \\
    train_cfg.val_interval=1 \\
    default_hooks.checkpoint.interval=1 \\
    default_hooks.logger.interval=1 \\
    train_dataloader.num_workers=2 \\
    val_dataloader.num_workers=1 \\
    train_dataloader.dataset.ann_file=annotations/{DEBUG_TRAIN_COCO.name} \\
    val_dataloader.dataset.ann_file=annotations/{DEBUG_VAL_COCO.name} \\
    test_dataloader.dataset.ann_file=annotations/{DEBUG_VAL_COCO.name} \\
    val_evaluator.ann_file="{DEBUG_VAL_COCO}" \\
    test_evaluator.ann_file="{DEBUG_VAL_COCO}"
"""

main_cmd = f"""#!/usr/bin/env bash
set -e

cd "{DATA_DIR}"

if [ -f "{MAIN_WORK_DIR}/latest.pth" ]; then
  echo "Resuming from latest checkpoint..."
  python "{TRAIN_SCRIPT}" "{CUSTOM_CONFIG}" \\
    --work-dir "{MAIN_WORK_DIR}" \\
    --resume
else
  echo "Starting fresh training..."
  python "{TRAIN_SCRIPT}" "{CUSTOM_CONFIG}" \\
    --work-dir "{MAIN_WORK_DIR}"
fi
"""

DEBUG_RUN_SH.write_text(debug_cmd)
MAIN_RUN_SH.write_text(main_cmd)

os.chmod(DEBUG_RUN_SH, 0o755)
os.chmod(MAIN_RUN_SH, 0o755)

print(f"\nSaved debug run script: {DEBUG_RUN_SH}")
print(f"Saved main run script : {MAIN_RUN_SH}")

# ------------------------------------------------------------
# 8. Parse config check
# ------------------------------------------------------------

try:
    from mmengine.config import Config

    cfg = Config.fromfile(str(CUSTOM_CONFIG))

    print("\nConfig parse check:")
    print(f"data_root : {cfg.data_root}")
    print(f"train ann : {cfg.train_dataloader.dataset.ann_file}")
    print(f"val ann   : {cfg.val_dataloader.dataset.ann_file}")
    print(f"batch     : {cfg.train_dataloader.batch_size}")
    print(f"epochs    : {cfg.train_cfg.max_epochs}")
    print(f"work_dir  : {cfg.work_dir}")

except Exception as e:
    print("\nConfig parse skipped/failed.")
    print("This is not fatal if you run from the correct conda env.")
    print(str(e))

# ------------------------------------------------------------
# 9. Summary
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("01_make_dino_config.py SUMMARY")
print("=" * 80)
print(f"Experiment      : {EXPERIMENT_NAME}")
print(f"Image size      : {IMG_SIZE}")
print(f"Train images    : {len(train_coco['images']):,}")
print(f"Train annots    : {len(train_coco['annotations']):,}")
print(f"Val images      : {len(val_coco['images']):,}")
print(f"Val annots      : {len(val_coco['annotations']):,}")
print(f"Config          : {CUSTOM_CONFIG}")
print(f"Debug script    : {DEBUG_RUN_SH}")
print(f"Main script     : {MAIN_RUN_SH}")
print(f"Debug work dir  : {DEBUG_WORK_DIR}")
print(f"Main work dir   : {MAIN_WORK_DIR}")

print("\nNext command:")
print(f"bash {DEBUG_RUN_SH}")