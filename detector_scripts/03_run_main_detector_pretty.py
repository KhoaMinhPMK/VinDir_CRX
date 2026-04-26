# ============================================================
# 03_run_main_detector_pretty.py
# Run DINO detector training with cleaner console output.
#
# Usage:
#   cd /workspace/Hokkaido_collaboration/km/lung_project/CRX/vinbigdata-chest-xray-abnormalities-detection
#   conda activate cxr_mmdet310
#   python trackB_cxr_dino/detector_scripts/03_run_main_detector_pretty.py
# ============================================================

from pathlib import Path
import os
import re
import csv
import subprocess
import sys
import time

DATA_DIR = Path(
    "/workspace/Hokkaido_collaboration/km/lung_project/CRX/"
    "vinbigdata-chest-xray-abnormalities-detection"
).resolve()

os.chdir(DATA_DIR)

TRACKB_DIR = DATA_DIR / "trackB_cxr_dino"
MMDET_SRC_DIR = TRACKB_DIR / "mmdetection_py310"

CONFIG_PATH = TRACKB_DIR / "detector_configs" / "cxr_dino_r50_1280_fold0.py"
TRAIN_SCRIPT = MMDET_SRC_DIR / "tools" / "train.py"

WORK_DIR = TRACKB_DIR / "detector_work_dirs" / "cxr_dino_r50_1280_fold0"

LOG_DIR = TRACKB_DIR / "detector_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

RAW_LOG = LOG_DIR / "main_train_raw.log"
PRETTY_LOG = LOG_DIR / "main_train_pretty.log"
SUMMARY_CSV = LOG_DIR / "main_train_summary.csv"

assert CONFIG_PATH.exists(), f"Missing config: {CONFIG_PATH}"
assert TRAIN_SCRIPT.exists(), f"Missing train.py: {TRAIN_SCRIPT}"

WORK_DIR.mkdir(parents=True, exist_ok=True)

latest_ckpt = WORK_DIR / "latest.pth"

cmd = [
    sys.executable,
    str(TRAIN_SCRIPT),
    str(CONFIG_PATH),
    "--work-dir",
    str(WORK_DIR),
]

if latest_ckpt.exists():
    cmd.append("--resume")
    resume_text = "YES"
else:
    resume_text = "NO"

print("=" * 100)
print("TRACK B DINO MAIN TRAIN")
print("=" * 100)
print(f"DATA_DIR   : {DATA_DIR}")
print(f"CONFIG     : {CONFIG_PATH}")
print(f"WORK_DIR   : {WORK_DIR}")
print(f"RAW_LOG    : {RAW_LOG}")
print(f"SUMMARY_CSV: {SUMMARY_CSV}")
print(f"RESUME     : {resume_text}")
print("=" * 100)

train_re = re.compile(
    r"Epoch\(train\) \[(?P<epoch>\d+)\]\[(?P<iter>\d+)/(?P<total>\d+)\].*?"
    r"lr: (?P<lr>[\deE\+\-\.]+).*?"
    r"eta: (?P<eta>[0-9:]+).*?"
    r"time: (?P<time>[\d\.]+).*?"
    r"memory: (?P<memory>\d+).*?"
    r"loss: (?P<loss>[\d\.]+).*?"
    r"loss_cls: (?P<loss_cls>[\d\.]+).*?"
    r"loss_bbox: (?P<loss_bbox>[\d\.]+).*?"
    r"loss_iou: (?P<loss_iou>[\d\.]+)"
)

val_re = re.compile(
    r"Epoch\(val\).*?"
    r"coco/bbox_mAP: (?P<map>[-\d\.]+).*?"
    r"coco/bbox_mAP_s: (?P<map_s>[-\d\.]+).*?"
    r"coco/bbox_mAP_m: (?P<map_m>[-\d\.]+).*?"
    r"coco/bbox_mAP_l: (?P<map_l>[-\d\.]+)"
)

best_re = re.compile(
    r"The best checkpoint with (?P<best>[-\d\.]+) coco/bbox_mAP at (?P<epoch>\d+) epoch"
)

rows = []

def write_pretty(text):
    print(text, flush=True)
    with open(PRETTY_LOG, "a") as f:
        f.write(text + "\n")

with open(RAW_LOG, "a") as raw_f:
    raw_f.write("\n\n" + "=" * 100 + "\n")
    raw_f.write("NEW TRAIN RUN\n")
    raw_f.write("=" * 100 + "\n")
    raw_f.write("CMD: " + " ".join(cmd) + "\n")

with open(PRETTY_LOG, "a") as pretty_f:
    pretty_f.write("\n\n" + "=" * 100 + "\n")
    pretty_f.write("NEW TRAIN RUN\n")
    pretty_f.write("=" * 100 + "\n")

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"

start_time = time.time()

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=env,
)

last_print_key = None

for line in proc.stdout:
    line = line.rstrip("\n")

    with open(RAW_LOG, "a") as raw_f:
        raw_f.write(line + "\n")

    m = train_re.search(line)
    if m:
        d = m.groupdict()

        epoch = int(d["epoch"])
        it = int(d["iter"])
        total = int(d["total"])

        # Print every 100 iters, first iter, and last iter.
        should_print = (it == 1) or (it % 100 == 0) or (it == total)

        if should_print:
            msg = (
                f"[TRAIN] epoch={epoch:02d} "
                f"iter={it:05d}/{total:<5d} "
                f"loss={float(d['loss']):7.3f} "
                f"cls={float(d['loss_cls']):6.3f} "
                f"bbox={float(d['loss_bbox']):6.3f} "
                f"iou={float(d['loss_iou']):6.3f} "
                f"lr={float(d['lr']):.2e} "
                f"mem={int(d['memory']):5d}MB "
                f"eta={d['eta']}"
            )
            write_pretty(msg)

        rows.append({
            "phase": "train",
            "epoch": epoch,
            "iter": it,
            "total_iter": total,
            "loss": float(d["loss"]),
            "loss_cls": float(d["loss_cls"]),
            "loss_bbox": float(d["loss_bbox"]),
            "loss_iou": float(d["loss_iou"]),
            "lr": float(d["lr"]),
            "memory": int(d["memory"]),
            "bbox_mAP": "",
            "bbox_mAP_s": "",
            "bbox_mAP_m": "",
            "bbox_mAP_l": "",
        })

        continue

    m = val_re.search(line)
    if m:
        d = m.groupdict()
        msg = (
            f"[VAL]   bbox_mAP={float(d['map']):.4f} "
            f"small={float(d['map_s']):.4f} "
            f"medium={float(d['map_m']):.4f} "
            f"large={float(d['map_l']):.4f}"
        )
        write_pretty(msg)

        rows.append({
            "phase": "val",
            "epoch": "",
            "iter": "",
            "total_iter": "",
            "loss": "",
            "loss_cls": "",
            "loss_bbox": "",
            "loss_iou": "",
            "lr": "",
            "memory": "",
            "bbox_mAP": float(d["map"]),
            "bbox_mAP_s": float(d["map_s"]),
            "bbox_mAP_m": float(d["map_m"]),
            "bbox_mAP_l": float(d["map_l"]),
        })

        continue

    m = best_re.search(line)
    if m:
        d = m.groupdict()
        msg = f"[BEST]  epoch={d['epoch']} best_bbox_mAP={float(d['best']):.4f}"
        write_pretty(msg)
        continue

    # Important events
    if "Saving checkpoint" in line:
        write_pretty("[CKPT]  " + line.split("INFO -")[-1].strip())

    if "Starting fresh training" in line or "Resuming" in line:
        write_pretty("[INFO]  " + line)

ret = proc.wait()
elapsed = time.time() - start_time

fieldnames = [
    "phase", "epoch", "iter", "total_iter",
    "loss", "loss_cls", "loss_bbox", "loss_iou",
    "lr", "memory",
    "bbox_mAP", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l",
]

with open(SUMMARY_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

write_pretty("=" * 100)
write_pretty(f"FINISHED returncode={ret} elapsed_hours={elapsed/3600:.2f}")
write_pretty(f"RAW_LOG    : {RAW_LOG}")
write_pretty(f"PRETTY_LOG : {PRETTY_LOG}")
write_pretty(f"SUMMARY_CSV: {SUMMARY_CSV}")
write_pretty(f"WORK_DIR   : {WORK_DIR}")
write_pretty("=" * 100)

if ret != 0:
    raise SystemExit(ret)