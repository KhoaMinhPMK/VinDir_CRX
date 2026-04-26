from pathlib import Path
import os, json, random, subprocess, textwrap

# ============================================================
# FAST DETECTOR CONFIG
# - Không tạo lại ảnh
# - Dùng lại images_1280 + COCO_1280
# - Train subset cân bằng hơn: all abnormal + một phần normal
# - Resize on-the-fly 1280 -> 1024 trong MMDetection
# ============================================================

DATA_DIR = Path(
    "/workspace/Hokkaido_collaboration/km/lung_project/CRX/"
    "vinbigdata-chest-xray-abnormalities-detection"
).resolve()
os.chdir(DATA_DIR)

TRACKB_DIR = DATA_DIR / "trackB_cxr_dino"
DATA_SUBDIR = TRACKB_DIR / "data"
ANN_DIR = DATA_SUBDIR / "annotations"
SCRIPT_DIR = TRACKB_DIR / "detector_scripts"
CONFIG_DIR = TRACKB_DIR / "detector_configs"
WORK_DIR = TRACKB_DIR / "detector_work_dirs"
MMDET_DIR = TRACKB_DIR / "mmdetection_py310"

IMG_SOURCE_SIZE = 1280
TRAIN_SIZE = 1024
FOLD = 0
SEED = 42

# Fast subset: all abnormal + N normal
NORMAL_SAMPLE_N = 2000
MAX_EPOCHS = 5
BATCH_SIZE = 2
ACCUMULATIVE_COUNTS = 2
LR = 5e-5

TRAIN_COCO = ANN_DIR / f"train_fold{FOLD}_coco_{IMG_SOURCE_SIZE}.json"
VAL_COCO = ANN_DIR / f"val_fold{FOLD}_coco_{IMG_SOURCE_SIZE}.json"

FAST_TRAIN_COCO = ANN_DIR / f"train_fast_fold{FOLD}_coco_{IMG_SOURCE_SIZE}.json"
DEBUG_TRAIN_COCO = ANN_DIR / f"train_fast_debug_fold{FOLD}_coco_{IMG_SOURCE_SIZE}.json"
DEBUG_VAL_COCO = ANN_DIR / f"val_fast_debug_fold{FOLD}_coco_{IMG_SOURCE_SIZE}.json"

EXP_NAME = f"cxr_dino_r50_fast{TRAIN_SIZE}_from{IMG_SOURCE_SIZE}_fold{FOLD}"
CONFIG_PATH = CONFIG_DIR / f"{EXP_NAME}.py"
FAST_WORK_DIR = WORK_DIR / EXP_NAME
DEBUG_WORK_DIR = WORK_DIR / f"{EXP_NAME}_debug"

RUN_DEBUG_SH = SCRIPT_DIR / "run_fast_debug_detector.sh"
RUN_FAST_SH = SCRIPT_DIR / "run_fast_detector.sh"

for p in [SCRIPT_DIR, CONFIG_DIR, WORK_DIR, FAST_WORK_DIR, DEBUG_WORK_DIR]:
    p.mkdir(parents=True, exist_ok=True)

assert TRAIN_COCO.exists(), TRAIN_COCO
assert VAL_COCO.exists(), VAL_COCO
assert (DATA_SUBDIR / "images_1280/train").exists()
assert (DATA_SUBDIR / "images_1280/val").exists()
assert (MMDET_DIR / "tools/train.py").exists()

TRAIN_SCRIPT = MMDET_DIR / "tools/train.py"
BASE_CONFIG = MMDET_DIR / "configs/dino/dino-4scale_r50_8xb2-12e_coco.py"
assert BASE_CONFIG.exists(), BASE_CONFIG

print("=" * 80)
print("MAKE FAST DINO DETECTOR CONFIG")
print("=" * 80)
print("DATA_DIR        :", DATA_DIR)
print("SOURCE COCO     :", TRAIN_COCO)
print("VAL COCO        :", VAL_COCO)
print("FAST TRAIN COCO :", FAST_TRAIN_COCO)
print("CONFIG          :", CONFIG_PATH)
print("TRAIN_SIZE      :", TRAIN_SIZE)
print("BATCH_SIZE      :", BATCH_SIZE)
print("EPOCHS          :", MAX_EPOCHS)

# ------------------------------------------------------------
# Load COCO
# ------------------------------------------------------------

with open(TRAIN_COCO, "r") as f:
    train_coco = json.load(f)

with open(VAL_COCO, "r") as f:
    val_coco = json.load(f)

categories = sorted(train_coco["categories"], key=lambda x: int(x["id"]))
CLASS_NAMES = [c["name"] for c in categories]
NUM_CLASSES = len(CLASS_NAMES)
assert NUM_CLASSES == 14

# ------------------------------------------------------------
# Build fast train subset: all abnormal + sampled normal
# ------------------------------------------------------------

all_train_images = train_coco["images"]
all_train_anns = train_coco["annotations"]

abnormal_image_ids = set(ann["image_id"] for ann in all_train_anns)
all_image_ids = set(img["id"] for img in all_train_images)
normal_image_ids = sorted(list(all_image_ids - abnormal_image_ids))
abnormal_image_ids = sorted(list(abnormal_image_ids))

rng = random.Random(SEED)
sampled_normal_ids = rng.sample(
    normal_image_ids,
    k=min(NORMAL_SAMPLE_N, len(normal_image_ids))
)

keep_ids = set(abnormal_image_ids) | set(sampled_normal_ids)

fast_images = [img for img in all_train_images if img["id"] in keep_ids]
fast_anns = [ann for ann in all_train_anns if ann["image_id"] in keep_ids]

fast_coco = {
    "info": train_coco.get("info", {}),
    "licenses": train_coco.get("licenses", []),
    "images": fast_images,
    "annotations": fast_anns,
    "categories": train_coco["categories"],
}

with open(FAST_TRAIN_COCO, "w") as f:
    json.dump(fast_coco, f)

print("\nFast subset:")
print("All train images :", len(all_train_images))
print("Abnormal images  :", len(abnormal_image_ids))
print("Sampled normal   :", len(sampled_normal_ids))
print("Fast train images:", len(fast_images))
print("Fast train annots:", len(fast_anns))

# ------------------------------------------------------------
# Debug subsets
# ------------------------------------------------------------

def make_debug_coco(src, dst, n_images):
    with open(src, "r") as f:
        coco = json.load(f)

    images = coco["images"][:n_images]
    keep = set(img["id"] for img in images)
    anns = [ann for ann in coco["annotations"] if ann["image_id"] in keep]

    out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": anns,
        "categories": coco["categories"],
    }

    with open(dst, "w") as f:
        json.dump(out, f)

    print(f"Debug COCO saved: {dst}")
    print(f"  images: {len(images)}")
    print(f"  annots: {len(anns)}")

make_debug_coco(FAST_TRAIN_COCO, DEBUG_TRAIN_COCO, 256)
make_debug_coco(VAL_COCO, DEBUG_VAL_COCO, 64)

# ------------------------------------------------------------
# Write MMDetection config
# ------------------------------------------------------------

classes_tuple = "(" + ", ".join(repr(x) for x in CLASS_NAMES) + ",)"

config_text = f"""
_base_ = r'{BASE_CONFIG}'

custom_imports = dict(imports=['mmdet'], allow_failed_imports=False)

data_root = r'{DATA_SUBDIR}/'
metainfo = dict(classes={classes_tuple})
num_classes = {NUM_CLASSES}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=({TRAIN_SIZE}, {TRAIN_SIZE}), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, color_type='color'),
    dict(type='Resize', scale=({TRAIN_SIZE}, {TRAIN_SIZE}), keep_ratio=False),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    _delete_=True,
    batch_size={BATCH_SIZE},
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/{FAST_TRAIN_COCO.name}',
        data_prefix=dict(img='images_{IMG_SOURCE_SIZE}/train/'),
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
        ann_file='annotations/val_fold{FOLD}_coco_{IMG_SOURCE_SIZE}.json',
        data_prefix=dict(img='images_{IMG_SOURCE_SIZE}/val/'),
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
    ann_file=data_root + 'annotations/val_fold{FOLD}_coco_{IMG_SOURCE_SIZE}.json',
    metric='bbox',
    iou_thrs=[0.4],
    classwise=False,
    format_only=False,
    backend_args=None
)

test_evaluator = val_evaluator

model = dict(
    bbox_head=dict(num_classes=num_classes)
)

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs={MAX_EPOCHS},
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='float16',
    accumulative_counts={ACCUMULATIVE_COUNTS},
    optimizer=dict(type='AdamW', lr={LR}, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end={MAX_EPOCHS}, by_epoch=True, milestones=[3, 4], gamma=0.1)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
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

work_dir = r'{FAST_WORK_DIR}'

load_from = None
resume = False
auto_scale_lr = dict(enable=False)
"""

CONFIG_PATH.write_text(textwrap.dedent(config_text).strip() + "\n")

print("\nConfig saved:", CONFIG_PATH)

# ------------------------------------------------------------
# Run scripts
# ------------------------------------------------------------

debug_sh = f"""#!/usr/bin/env bash
set -e
cd "{DATA_DIR}"

python "{TRAIN_SCRIPT}" "{CONFIG_PATH}" \\
  --work-dir "{DEBUG_WORK_DIR}" \\
  --cfg-options \\
    train_cfg.max_epochs=1 \\
    train_cfg.val_interval=1 \\
    default_hooks.checkpoint.interval=1 \\
    default_hooks.logger.interval=20 \\
    train_dataloader.dataset.ann_file=annotations/{DEBUG_TRAIN_COCO.name} \\
    val_dataloader.dataset.ann_file=annotations/{DEBUG_VAL_COCO.name} \\
    test_dataloader.dataset.ann_file=annotations/{DEBUG_VAL_COCO.name} \\
    val_evaluator.ann_file="{DEBUG_VAL_COCO}" \\
    test_evaluator.ann_file="{DEBUG_VAL_COCO}"
"""

fast_sh = f"""#!/usr/bin/env bash
set -e
cd "{DATA_DIR}"

if [ -f "{FAST_WORK_DIR}/latest.pth" ]; then
  echo "Resuming fast detector..."
  python "{TRAIN_SCRIPT}" "{CONFIG_PATH}" --work-dir "{FAST_WORK_DIR}" --resume
else
  echo "Starting fast detector..."
  python "{TRAIN_SCRIPT}" "{CONFIG_PATH}" --work-dir "{FAST_WORK_DIR}"
fi
"""

RUN_DEBUG_SH.write_text(debug_sh)
RUN_FAST_SH.write_text(fast_sh)

os.chmod(RUN_DEBUG_SH, 0o755)
os.chmod(RUN_FAST_SH, 0o755)

print("\nRun scripts:")
print("Debug:", RUN_DEBUG_SH)
print("Fast :", RUN_FAST_SH)

# ------------------------------------------------------------
# Config parse check
# ------------------------------------------------------------

from mmengine.config import Config
cfg = Config.fromfile(str(CONFIG_PATH))

print("\nParse check:")
print("train ann:", cfg.train_dataloader.dataset.ann_file)
print("val ann  :", cfg.val_dataloader.dataset.ann_file)
print("batch    :", cfg.train_dataloader.batch_size)
print("epochs   :", cfg.train_cfg.max_epochs)
print("work_dir :", cfg.work_dir)

print("\nNext commands:")
print(f"bash {RUN_DEBUG_SH}")
print(f"bash {RUN_FAST_SH}")