"""Download and convert real event-camera datasets to the OmniBird layout.

OmniBird's `EventScapeDataset` (in eventscape.py) expects this per-clip layout:

    root/
      clip_000/
        events_0.npy        # (N_raw, 4): x_int, y_int, t_us, polarity ∈ {0, 1}
        label_0.txt         # integer class label
        rgb_0.png           # optional, for multimodal
        events_1.npy
        ...

This file provides helpers for the two most useful source datasets:

  1. EventScape  — CARLA driving simulation, events + RGB + depth + semantic.
                   Project: https://rpg.ifi.uzh.ch/RAMNet.html
                   ~50–250 GB total; per-Town subsets available.

  2. CIFAR10-DVS — event-camera replay of CIFAR-10 images. Small (~1 GB),
                   10-class classification, well-known benchmark.
                   Project: https://www.frontiersin.org/articles/10.3389/fnins.2017.00309
                   Hosted: https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671

Functions:
    download_cifar10_dvs(out_dir)               # download raw archives
    convert_cifar10_dvs(raw_dir, out_dir)        # → OmniBird layout
    eventscape_download_urls()                   # list URLs for manual wget
    convert_eventscape(raw_dir, out_dir)         # → OmniBird layout

Run as a script:
    python -m datasets.download cifar10_dvs --out ./data/cifar10_dvs_omnibird
    python -m datasets.download eventscape --raw /path/to/eventscape --out ./data/eventscape_omnibird
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest: Path):
    """Stream-download `url` to `dest`, printing a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  downloading {url}")
    last_pct = [-1]
    def reporthook(block_num, block_size, total_size):
        if total_size <= 0: return
        pct = int(100 * block_num * block_size / total_size)
        if pct != last_pct[0] and pct % 5 == 0:
            print(f"    {pct:3d}%  ({block_num * block_size / 1e6:.1f}/{total_size / 1e6:.1f} MB)")
            last_pct[0] = pct
    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print(f"  done. saved → {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")


def _extract_zip(zpath: Path, out_dir: Path):
    print(f"  extracting {zpath.name} → {out_dir}")
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(out_dir)


# ---------------------------------------------------------------------------
# CIFAR10-DVS
# ---------------------------------------------------------------------------

CIFAR10_DVS_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Figshare hosts CIFAR10-DVS as a zip per class. URLs verified Mar 2024;
# update if Figshare reorganizes. See:
#   https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671
# In practice users often download the full tarball from the figshare page
# manually and extract; this helper supports both.
CIFAR10_DVS_FIGSHARE_DATASET_ID = 4724671
CIFAR10_DVS_FIGSHARE_API = (
    f"https://api.figshare.com/v2/articles/{CIFAR10_DVS_FIGSHARE_DATASET_ID}/files"
)


def download_cifar10_dvs(out_dir: str | Path):
    """Download the CIFAR10-DVS zips from Figshare.

    NOTE: The Figshare API returns a list of file URLs. This function
    auto-discovers them and downloads each one. If Figshare's API or the
    dataset's hosting changes, follow:

        https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671

    Manually download the .zip files and place them in `out_dir`.
    """
    import json
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"querying Figshare for CIFAR10-DVS file manifest ...")
    try:
        with urllib.request.urlopen(CIFAR10_DVS_FIGSHARE_API, timeout=20) as resp:
            manifest = json.loads(resp.read())
    except Exception as e:
        print(f"  Figshare API call failed ({e}).\n"
              f"  Please manually download the zips from\n"
              f"    https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671\n"
              f"  and place the .zip files in: {out_dir}")
        return
    print(f"  found {len(manifest)} files; downloading...")
    for f in manifest:
        url = f["download_url"]
        name = f["name"]
        _download_with_progress(url, out_dir / name)
    print(f"\nCIFAR10-DVS raw zips downloaded to: {out_dir}")
    print(f"Next: convert with `python -m datasets.download convert_cifar10_dvs --raw {out_dir} --out <omnibird_layout_dir>`")


def _strip_aedat_header(blob: bytes) -> tuple:
    """Skip ASCII header lines (each starts with '#') and return (body, format).
    `format` is 'aedat2', 'aedat3', or 'unknown'."""
    head_text = blob[:1024].decode('ascii', errors='ignore')
    if 'AER-DAT3' in head_text:
        fmt = 'aedat3'
    elif 'AER-DAT2' in head_text or '!AER-DAT' in head_text:
        fmt = 'aedat2'
    else:
        fmt = 'unknown'
    offset = 0
    while offset < len(blob) and blob[offset:offset+1] == b'#':
        eol = blob.find(b'\n', offset)
        if eol < 0:
            break
        offset = eol + 1
    return blob[offset:], fmt


def _parse_aedat2_dvs128_events(body: bytes) -> np.ndarray:
    """AEDAT 2.0 DVS128 polarity-event parser. CIFAR10-DVS uses this format.

    Per event (8 bytes, BIG-ENDIAN):
        uint32: address       — bit 0 = polarity, bits 1-7 = y (0..127), bits 8-14 = x (0..127)
        uint32: timestamp_us  — recording-local microseconds

    Returns (N, 4) float32: (x_int, y_int, t_us, polarity in {0,1}).
    """
    n = len(body) // 8
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    raw = np.frombuffer(body[:n * 8], dtype='>u4').reshape(-1, 2)   # big-endian u32 pairs
    addr = raw[:, 0]
    t_us = raw[:, 1].astype(np.int64)
    pol   = (addr & 1).astype(np.float32)
    y_int = ((addr >> 1) & 0x7F).astype(np.float32)                  # 7-bit y
    x_int = ((addr >> 8) & 0x7F).astype(np.float32)                  # 7-bit x
    return np.stack([x_int, y_int, t_us.astype(np.float32), pol], axis=1)


def _parse_aedat3_polarity_events(body: bytes) -> np.ndarray:
    """AEDAT 3.1 polarity event-packet parser. Each event is 8 bytes (little-endian):
        uint32: data       — bit 0 = polarity, bits 1-10 = y, bits 11-21 = x
        int32:  timestamp_us
    """
    n = len(body) // 8
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    raw = np.frombuffer(body[:n * 8], dtype=np.uint32).reshape(-1, 2)
    data = raw[:, 0]
    t_us = raw[:, 1].view(np.int32).astype(np.int64)
    pol   = (data & 1).astype(np.float32)
    y_int = ((data >> 1) & 0x3FF).astype(np.float32)
    x_int = ((data >> 11) & 0x7FF).astype(np.float32)
    return np.stack([x_int, y_int, t_us.astype(np.float32), pol], axis=1)


def _parse_aedat31_events(aedat_bytes: bytes, sensor_h: int = 128, sensor_w: int = 128):
    """Dispatch AEDAT 2.0 / 3.1 polarity-event parsing based on header.

    CIFAR10-DVS uses AEDAT 2.0 on a DVS128 sensor. Many newer datasets use 3.1.
    Returns (N_raw, 4) float32 with columns (x_int, y_int, t_us, polarity in {0,1}).
    """
    body, fmt = _strip_aedat_header(aedat_bytes)
    if fmt == 'aedat2':
        return _parse_aedat2_dvs128_events(body)
    elif fmt == 'aedat3':
        return _parse_aedat3_polarity_events(body)
    else:
        # Heuristic fallback: CIFAR10-DVS files often lack a discoverable version
        # tag in the first 1KB. The AEDAT 2.0 DVS128 layout is the safer default
        # for CIFAR10-DVS specifically.
        return _parse_aedat2_dvs128_events(body)


def convert_cifar10_dvs(raw_dir: str | Path, out_dir: str | Path,
                        sensor_hw=(128, 128), events_per_clip: int = 8192,
                        time_window_us: int = 50_000):
    """Convert a raw CIFAR10-DVS directory to OmniBird layout.

    Expects per-class AEDAT files in `raw_dir/<class_name>/*.aedat`.
    Writes per-clip directories `out_dir/clip_<idx>/` with:

        events_0.npy  (N_raw, 4) - x_int, y_int, t_us, polarity {0,1}
        label_0.txt   integer class id
    """
    raw_dir = Path(raw_dir); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    h, w = sensor_hw
    clip_idx = 0
    classes = []
    for d in sorted(raw_dir.iterdir()):
        if d.is_dir() and d.name in CIFAR10_DVS_CLASSES:
            classes.append(d.name)
    if not classes:
        # Maybe extracted with capitalized names; try a broader scan
        classes = [d.name for d in sorted(raw_dir.iterdir()) if d.is_dir()]
        print(f"warning: did not find canonical CIFAR10-DVS class folder names; using found: {classes}")

    for class_name in classes:
        cls_id = CIFAR10_DVS_CLASSES.index(class_name) if class_name in CIFAR10_DVS_CLASSES else None
        if cls_id is None:
            print(f"  skipping unknown class: {class_name}"); continue
        files = sorted((raw_dir / class_name).glob("*.aedat"))
        print(f"  class={class_name}  cls_id={cls_id}  files={len(files)}")
        for fp in files:
            try:
                raw = fp.read_bytes()
                events = _parse_aedat31_events(raw, sensor_h=h, sensor_w=w)
            except Exception as e:
                print(f"    skip {fp.name}: {e}"); continue
            if events.shape[0] == 0:
                continue
            # Optionally trim to a single time window around the clip's center
            t = events[:, 2]
            t = t - t.min()
            mid = (t.max() - t.min()) / 2
            mask = (t >= mid - time_window_us/2) & (t <= mid + time_window_us/2)
            events = events[mask]
            if events.shape[0] > events_per_clip:
                sel = np.random.choice(events.shape[0], events_per_clip, replace=False)
                sel.sort()
                events = events[sel]
            clip_dir = out_dir / f"clip_{clip_idx:05d}"; clip_dir.mkdir(parents=True, exist_ok=True)
            np.save(clip_dir / "events_0.npy", events.astype(np.float32))
            (clip_dir / "label_0.txt").write_text(str(cls_id))
            clip_idx += 1
    print(f"\nwrote {clip_idx} clips to {out_dir}")


# ---------------------------------------------------------------------------
# EventScape
# ---------------------------------------------------------------------------

# EventScape per-Town zip URLs. Verified at the RAMNet project page; if these
# 404, check https://rpg.ifi.uzh.ch/RAMNet.html for the current download links.
# EventScape direct download URLs — verified against the rpg_ramnet README
# (https://github.com/uzh-rpg/rpg_ramnet, README.md lines 64-68):
#
#   Training Set   (71 GB):  http://rpg.ifi.uzh.ch/data/RAM_Net/dataset/Town01-03_train.zip
#   Validation Set (12 GB):  http://rpg.ifi.uzh.ch/data/RAM_Net/dataset/Town05_val.zip
#   Test Set       (14 GB):  http://rpg.ifi.uzh.ch/data/RAM_Net/dataset/Town05_test.zip
#
# Note: these are large. Start with the validation set (smallest at 12 GB) for
# development; switch to the full training set once your pipeline is verified.
EVENTSCAPE_URLS = {
    "train": "http://rpg.ifi.uzh.ch/data/RAM_Net/dataset/Town01-03_train.zip",
    "val":   "http://rpg.ifi.uzh.ch/data/RAM_Net/dataset/Town05_val.zip",
    "test":  "http://rpg.ifi.uzh.ch/data/RAM_Net/dataset/Town05_test.zip",
}


def eventscape_download_urls() -> dict:
    """Return the canonical EventScape URLs. If RPG re-hosts the data,
    update the EVENTSCAPE_URLS dict above."""
    return dict(EVENTSCAPE_URLS)


def download_eventscape(out_dir: str | Path, subsets=("val",), extract: bool = True):
    """Download one or more EventScape subsets to `out_dir`.

    Subsets:
        "val"   - 12 GB (smallest, RECOMMENDED first download for dev)
        "test"  - 14 GB
        "train" - 71 GB
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for name in subsets:
        if name not in EVENTSCAPE_URLS:
            print(f"  unknown subset: {name}  (known: {list(EVENTSCAPE_URLS)})"); continue
        url = EVENTSCAPE_URLS[name]
        zip_path = out_dir / f"{name}.zip"
        try:
            _download_with_progress(url, zip_path)
        except Exception as e:
            print(f"  download failed for {name}: {e}\n"
                  f"  → please manually retrieve {url}\n"
                  f"    and place the zip at {zip_path}")
            continue
        if extract:
            _extract_zip(zip_path, out_dir / name)
    print(f"\nEventScape subsets downloaded to: {out_dir}")
    print("Next: convert with `python -m datasets.download convert_eventscape "
          f"--raw {out_dir} --out <omnibird_layout_dir>`")


def convert_eventscape(raw_dir: str | Path, out_dir: str | Path,
                        events_per_clip: int = 8192,
                        sensor_hw=(256, 256)):
    """Convert an extracted EventScape directory to OmniBird's per-clip layout.

    Real EventScape layout (verified from rpg_ramnet/RAM_Net/data_loader/dataset.py):

        <raw_dir>/<sequence>/
            events/   *_NNNN_events.npy      # raw events: (n, 4) columns (t, x, y, polarity)
            frames/   *_NNNN_depth.npy        # per-pixel depth (float32)
            rgb/      *_NNNN_image.png        # RGB frame
            semantic/ *_NNNN_gt_labelIds.png  # per-pixel CARLA semantic class

    Each per-frame `*_NNNN_*` group is one timestep. We treat each timestep as
    one OmniBird clip and label it by the dominant CARLA semantic class
    (np.bincount(seg.ravel()).argmax()). For the canonical depth-prediction
    task, see Note A below.

    Requires Pillow (pip install pillow).

    Note A — labels:
      EventScape's *native* supervision is per-pixel depth (regression) +
      per-pixel semantic seg. There is NO per-clip classification label. The
      single integer this converter writes is the *dominant CARLA semantic
      class in the segmentation map of the same timestep*. This is a
      pragmatic coarse classification target that lets OmniBird's existing
      LinearProbe work; for the canonical RAMNet depth task you'd want a
      per-pixel regression probe head instead.
    """
    try:
        from PIL import Image
    except ImportError:
        print("missing dependency: pillow\n  pip install pillow"); return

    raw_dir = Path(raw_dir); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    clip_idx = 0
    H, W = sensor_hw

    # Find every sequence — defined by having an events/ folder with raw events npys
    sequences = []
    for events_dir in raw_dir.rglob("events"):
        if events_dir.is_dir() and any(events_dir.glob("*_events.npy")):
            sequences.append(events_dir.parent)
    sequences = sorted(set(sequences))
    print(f"found {len(sequences)} EventScape sequences")

    for seq_dir in sequences:
        ev_dir   = seq_dir / "events"
        sem_dir  = seq_dir / "semantic"
        rgb_dir  = seq_dir / "rgb"
        # Index by the 4-digit frame id at the end of the events filename
        ev_files = sorted(ev_dir.glob("*_events.npy"))
        for ev_file in ev_files:
            # *_NNNN_events.npy  →  frame id = the 4-digit stem
            stem = ev_file.stem.replace("_events", "")
            frame_id = stem[-4:]                  # last 4 chars are the zero-padded id
            sem_match = list(sem_dir.glob(f"*_{frame_id}_gt_labelIds.png"))
            rgb_match = list(rgb_dir.glob(f"*_{frame_id}_image.png"))

            # Load raw events. EventScape format: columns (t, x, y, polarity).
            ev_raw = np.load(ev_file)
            if ev_raw.size == 0:
                continue
            # Reorder to OmniBird's expected (x, y, t, polarity)
            ev_full = np.stack([ev_raw[:, 1], ev_raw[:, 2], ev_raw[:, 0], ev_raw[:, 3]],
                                axis=1).astype(np.float32)
            n = ev_full.shape[0]
            if n > events_per_clip:
                sel = np.random.choice(n, events_per_clip, replace=False); sel.sort()
                ev_full = ev_full[sel]

            # Coarse classification label = dominant CARLA semantic class in
            # this timestep's segmentation map.
            label = 0
            if sem_match:
                sem = np.asarray(Image.open(sem_match[0]))
                if sem.ndim == 3:
                    sem = sem[..., 0]    # CARLA stores class id in the R channel
                label = int(np.bincount(sem.ravel()).argmax())

            clip_dir = out_dir / f"clip_{clip_idx:06d}"; clip_dir.mkdir(parents=True, exist_ok=True)
            np.save(clip_dir / "events_0.npy", ev_full)
            (clip_dir / "label_0.txt").write_text(str(label))
            if rgb_match:
                try:
                    Image.open(rgb_match[0]).save(clip_dir / "rgb_0.png")
                except Exception:
                    pass
            clip_idx += 1
            if clip_idx % 500 == 0:
                print(f"  wrote {clip_idx} clips so far ...")

    print(f"\nwrote {clip_idx} clips to {out_dir}")


# ---------------------------------------------------------------------------
# Tonic-based downloads (recommended — verified working URLs)
# ---------------------------------------------------------------------------
#
# https://tonic.readthedocs.io/ — a community-maintained library that wraps
# every well-hosted event-camera dataset with a torchvision-style API and
# handles the download + caching for you. Datasets it knows about (verified):
#
#   tonic.datasets.NMNIST          — N-MNIST (events of MNIST digits)
#   tonic.datasets.CIFAR10DVS      — CIFAR10-DVS (events of CIFAR-10 images)
#   tonic.datasets.NCALTECH101     — N-Caltech101
#   tonic.datasets.DVSGesture      — IBM DVS128 hand-gesture (robotics-ish)
#   tonic.datasets.NCARS           — N-CARS (binary car detection from events)
#   tonic.datasets.POKERDVS        — small playing-card dataset
#   ...etc
#
# Install with:   pip install tonic

TONIC_DATASETS = ("NMNIST", "CIFAR10DVS", "NCALTECH101", "DVSGesture", "NCARS")


def download_via_tonic(dataset_name: str, out_dir: str | Path):
    """Download an event-camera dataset via the Tonic library and convert it
    to OmniBird's per-clip layout.

    Tonic handles the download + extraction; we iterate over each sample,
    grab the raw (x, y, t, p) events, and write events_0.npy + label_0.txt.

    Args:
        dataset_name: one of TONIC_DATASETS (case-sensitive).
        out_dir:      destination root (will contain clip_NNNNN/ subdirs).
    """
    try:
        import tonic
    except ImportError:
        print("missing dependency: tonic. Install with:")
        print("    pip install tonic")
        return

    if not hasattr(tonic.datasets, dataset_name):
        print(f"unknown tonic dataset: {dataset_name}  (try one of: {TONIC_DATASETS})")
        return

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    DataCls = getattr(tonic.datasets, dataset_name)
    save_to = str(out_dir.parent / f"{dataset_name}_raw_cache")

    print(f"downloading {dataset_name} via tonic to {save_to} ...")
    # Most tonic datasets accept (save_to, train=True/False); a few are split-less.
    try:
        ds_train = DataCls(save_to=save_to, train=True)
    except TypeError:
        ds_train = DataCls(save_to=save_to)
    try:
        ds_test = DataCls(save_to=save_to, train=False)
    except TypeError:
        ds_test = ds_train
        print("  (dataset has no train/test split — using the whole set as 'train')")

    sensor_size = getattr(ds_train, "sensor_size", None)
    print(f"  train samples = {len(ds_train)}  test samples = {len(ds_test)}  sensor = {sensor_size}")

    def _dump(ds, prefix, start_idx):
        i = start_idx
        for k in range(len(ds)):
            events, label = ds[k]
            # Tonic events are structured arrays with fields x, y, t, p
            ev = np.stack(
                [events["x"].astype(np.float32),
                 events["y"].astype(np.float32),
                 events["t"].astype(np.float32),
                 events["p"].astype(np.float32)],
                axis=1,
            )
            clip_dir = out_dir / f"clip_{i:06d}"; clip_dir.mkdir(parents=True, exist_ok=True)
            np.save(clip_dir / "events_0.npy", ev)
            (clip_dir / "label_0.txt").write_text(str(int(label)))
            i += 1
            if i % 500 == 0:
                print(f"    {prefix}: wrote {i - start_idx}/{len(ds)} clips")
        return i

    n = _dump(ds_train, "train", 0)
    if ds_test is not ds_train:
        n = _dump(ds_test, "test", n)
    print(f"\nwrote {n} clips to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    p = argparse.ArgumentParser(description="Download / convert event-camera datasets for OmniBird")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_dvs   = sub.add_parser("cifar10_dvs",  help="download CIFAR10-DVS raw zips")
    p_dvs.add_argument("--out", required=True, help="destination directory")

    p_cdvs  = sub.add_parser("convert_cifar10_dvs", help="convert CIFAR10-DVS → OmniBird layout")
    p_cdvs.add_argument("--raw", required=True, help="dir containing class subfolders")
    p_cdvs.add_argument("--out", required=True, help="destination OmniBird layout dir")

    p_es    = sub.add_parser("eventscape",   help="download EventScape (CARLA) subsets")
    p_es.add_argument("--out", required=True, help="destination directory")
    p_es.add_argument("--subsets", nargs="*", default=["val"],
                      choices=list(EVENTSCAPE_URLS.keys()),
                      help="train (71GB) / val (12GB, default) / test (14GB)")
    p_es.add_argument("--no-extract", action="store_true")

    p_ces   = sub.add_parser("convert_eventscape", help="convert EventScape → OmniBird layout")
    p_ces.add_argument("--raw", required=True, help="extracted EventScape root")
    p_ces.add_argument("--out", required=True, help="destination OmniBird layout dir")

    p_list  = sub.add_parser("urls",         help="print canonical EventScape download URLs (placeholders)")

    p_tonic = sub.add_parser("tonic", help="download an event dataset via the tonic library (recommended)")
    p_tonic.add_argument("--name", required=True, choices=list(TONIC_DATASETS),
                          help="tonic dataset name")
    p_tonic.add_argument("--out", required=True, help="destination OmniBird layout dir")

    args = p.parse_args()
    if args.cmd == "cifar10_dvs":
        download_cifar10_dvs(args.out)
    elif args.cmd == "convert_cifar10_dvs":
        convert_cifar10_dvs(args.raw, args.out)
    elif args.cmd == "eventscape":
        download_eventscape(args.out, subsets=args.subsets, extract=not args.no_extract)
    elif args.cmd == "convert_eventscape":
        convert_eventscape(args.raw, args.out)
    elif args.cmd == "tonic":
        download_via_tonic(args.name, args.out)
    elif args.cmd == "urls":
        urls = eventscape_download_urls()
        if not urls:
            print("EVENTSCAPE_URLS is currently empty in download.py.")
            print("Visit https://rpg.ifi.uzh.ch/RAMNet.html to get the latest")
            print("download links (often Google Drive / institutional file servers),")
            print("then populate the EVENTSCAPE_URLS dict at the top of this file.")
            print()
            print("For an immediately downloadable alternative, run:")
            print("    python -m datasets.download tonic --name CIFAR10DVS --out ./data/cifar10_dvs_omnibird")
            print("    python -m datasets.download tonic --name DVSGesture --out ./data/dvs_gesture_omnibird")
        else:
            for name, url in urls.items():
                print(f"{name:24s}  {url}")


if __name__ == "__main__":
    _main()
