"""Standalone data-generation script for correspondence_bench.

OVERVIEW
========

This script ties together the four building blocks of the benchmark:

  1. **Scene generators** — `LinkedPrimitivesGenerator` (static, 2-view),
     `LinkedPrimitivesVideoGenerator` (spatiotemporal, 2-view), or
     `MultiViewLinkedPrimitivesGenerator` (3-view limited-FOV).
  2. **Operating point** — the difficulty knob (`easy / basic / hard /
     extreme / adversarial`, plus Hz-focused points for Dataset B).
  3. **Augmenter pipeline** — observation-channel modifications:
     noise, sparse subsampling, occlusion, image-space FOV crop.
  4. **Output** — saved as NPZ files: one file per scene containing
     RGB, segmentation, keypoints, visibility, IDs, and the latent
     label.

USAGE
=====

As a library:

    from generate_dataset import generate
    scenes = generate(
        dataset="static",
        operating_point="basic",
        n_scenes=100,
        augmenters=[GaussianNoiseAugmenter(0.05)],
        out_dir="./data/basic_n100",
    )

As a CLI:

    python generate_dataset.py --dataset static \
        --operating-point basic --n 100 \
        --noise-sigma 0.05 \
        --out ./data/basic_n100

    python generate_dataset.py --dataset video \
        --operating-point mixed_hz --n 50 \
        --subsample 0.4 \
        --out ./data/mixed_hz_sparse40

    python generate_dataset.py --dataset multiview \
        --operating-point basic --n 100 \
        --occlude 0.3 \
        --out ./data/multiview_occluded

SCENE FILE FORMAT
=================

Each scene is saved as `<out_dir>/scene_{idx:06d}.npz`:

  Static (2-view):
    rgb_A, rgb_B            (H, W, 3) uint8 — observed images after augmenters
    seg_A, seg_B            (H, W)    int32 — primitive IDs (background = -1)
    kpts_A, kpts_B          (N, 2)   float32 — 2D projected centers
    vis_A,  vis_B           (N,)     bool — visibility flags
    ids_A,  ids_B           (N,)     int32 — stable primitive IDs
    label                   scalar
    label_kind              str
    operating_point         str

  Video (2-view, T frames):
    rgb_A, rgb_B            (T, H, W, 3) uint8
    seg_A, seg_B            (T, H, W)    int32
    kpts_A, kpts_B          (T, N, 2)    float32
    vis_A,  vis_B           (T, N)       bool
    ids_A,  ids_B           (N,)         int32
    times                   (T,)         float32 — τ ∈ [0, 1]
    label, label_kind, operating_point

  Multiview (N cameras = 3 by default):
    rgb_0, rgb_1, rgb_2, ... (H, W, 3) uint8 each
    seg_0, ...               (H, W)    int32
    kpts_0, ...              (N, 2)    float32
    vis_0, ...               (N,)      bool
    ids_0, ...               (N,)      int32
    label, label_kind, operating_point, view_angles_deg

A scene index manifest is also written to `<out_dir>/manifest.json`
listing all scene files + summary stats (operating point, augmenters,
label distribution).
"""
from __future__ import annotations
import os, sys, json, argparse, time
from typing import Optional, List, Union
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from linked_primitives import LinkedPrimitivesGenerator
from linked_primitives_video import LinkedPrimitivesVideoGenerator
from multiview_primitives import (
    MultiViewLinkedPrimitivesGenerator,
    coverage_summary,
)
from augmenters import (
    Augmenter, IdentityAugmenter,
    GaussianNoiseAugmenter, SaltPepperNoiseAugmenter,
    RandomSubsampleAugmenter, CenterOcclusionAugmenter, LimitedFOVAugmenter,
    AugmenterPipeline,
)


# ===========================================================================
# Public API
# ===========================================================================

def generate(dataset: str,
             operating_point: Union[str, dict] = "basic",
             n_scenes: int = 100,
             image_size: int = 128,
             augmenters: Optional[List[Augmenter]] = None,
             label_kind: Optional[str] = None,
             label_K: int = 4,
             out_dir: Optional[str] = None,
             base_seed: int = 0,
             write_manifest: bool = True,
             verbose: bool = True) -> dict:
    """Generate `n_scenes` scenes and (optionally) write them to disk.

    Parameters
    ----------
    dataset : "static" | "video" | "multiview"
        Which scene generator to use.
    operating_point : str or dict
        Operating-point name or custom knobs dict.
    n_scenes : int
        How many independent scenes to generate.
    image_size : int
        Output H × W per view.
    augmenters : list[Augmenter] or None
        Applied to each rendered view after rendering. None = identity.
    label_kind : str or None
        Label function to compute. None picks a sensible default per
        dataset ("count_modulo_K" everywhere).
    label_K : int
        K for the count_modulo_K-style labels.
    out_dir : str or None
        If given, write each scene to `<out_dir>/scene_XXXXXX.npz`.
        Manifest written to `<out_dir>/manifest.json`.
    base_seed : int
        Per-scene seeds are `base_seed + scene_idx`.
    write_manifest : bool
        If True (default) write a JSON manifest summarising the run.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    A dict summarising the run:
        {
            "n_scenes": N,
            "label_distribution": {label: count},
            "coverage_stats": {...}   # multiview only
            "files": [list of saved paths or None],
        }
    """
    pipeline = AugmenterPipeline(augmenters or [IdentityAugmenter()])
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    if dataset == "static":
        gen = LinkedPrimitivesGenerator(operating_point=operating_point,
                                          image_size=image_size, base_seed=base_seed)
        default_label = label_kind or "count_modulo_K"
    elif dataset == "video":
        gen = LinkedPrimitivesVideoGenerator(operating_point=operating_point,
                                               image_size=image_size, base_seed=base_seed)
        default_label = label_kind or "count_modulo_K"
    elif dataset == "multiview":
        gen = MultiViewLinkedPrimitivesGenerator(operating_point=operating_point,
                                                   image_size=image_size, base_seed=base_seed)
        default_label = label_kind or "count_modulo_K"
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected static/video/multiview")

    label_counts: dict = {}
    coverage_accum = []
    files = []
    t0 = time.time()
    for idx in range(n_scenes):
        seed = base_seed + idx
        scene = gen.sample_scene(seed=seed)
        if dataset == "static":
            out_A = gen.render(scene, view="A")
            out_B = gen.render(scene, view="B")
            out_A_aug = pipeline(out_A, rng=seed * 2)
            out_B_aug = pipeline(out_B, rng=seed * 2 + 1)
            label = gen.compute_label(scene, kind=default_label, K=label_K)
            payload = {
                "rgb_A": out_A_aug["rgb"], "rgb_B": out_B_aug["rgb"],
                "seg_A": out_A_aug["seg"], "seg_B": out_B_aug["seg"],
                "kpts_A": out_A_aug["kpts"], "kpts_B": out_B_aug["kpts"],
                "vis_A":  out_A_aug["vis"],  "vis_B":  out_B_aug["vis"],
                "ids_A":  out_A_aug["ids"],  "ids_B":  out_B_aug["ids"],
            }
        elif dataset == "video":
            video = gen.render_video_pair(scene)
            for v in ("A", "B"):
                # Apply augmenters per-frame with a per-frame rng for
                # temporally-independent corruption. For temporally-fixed
                # corruption (e.g. constant occlusion mask), pass the same
                # rng to every frame instead.
                T = video[f"view_{v}"]["rgb"].shape[0]
                aug_rgb = np.empty_like(video[f"view_{v}"]["rgb"])
                aug_seg = np.empty_like(video[f"view_{v}"]["seg"])
                for t in range(T):
                    one = {
                        "rgb":  video[f"view_{v}"]["rgb"][t],
                        "seg":  video[f"view_{v}"]["seg"][t],
                        "kpts": video[f"view_{v}"]["kpts"][t],
                        "vis":  video[f"view_{v}"]["vis"][t],
                        "ids":  video[f"view_{v}"]["ids"],
                    }
                    aug = pipeline(one, rng=seed * 1000 + (0 if v == "A" else 500) + t)
                    aug_rgb[t] = aug["rgb"]; aug_seg[t] = aug["seg"]
                video[f"view_{v}"]["rgb"] = aug_rgb
                video[f"view_{v}"]["seg"] = aug_seg
            label = gen.compute_label(scene, kind=default_label, K=label_K)
            payload = {
                "rgb_A": video["view_A"]["rgb"], "rgb_B": video["view_B"]["rgb"],
                "seg_A": video["view_A"]["seg"], "seg_B": video["view_B"]["seg"],
                "kpts_A": video["view_A"]["kpts"], "kpts_B": video["view_B"]["kpts"],
                "vis_A":  video["view_A"]["vis"],  "vis_B":  video["view_B"]["vis"],
                "ids_A":  video["view_A"]["ids"],  "ids_B":  video["view_B"]["ids"],
                "times":  video["times"],
            }
        else:   # multiview
            renders = gen.render(scene)
            renders_aug = [pipeline(r, rng=seed * 100 + v_idx) for v_idx, r in enumerate(renders)]
            label = gen.compute_label(scene, kind=default_label, K=label_K)
            payload = {}
            for v_idx, r in enumerate(renders_aug):
                payload[f"rgb_{v_idx}"]      = r["rgb"]
                payload[f"seg_{v_idx}"]      = r["seg"]
                payload[f"depth_{v_idx}"]    = r["depth"]
                payload[f"kpts_{v_idx}"]     = r["kpts"]
                payload[f"vis_{v_idx}"]      = r["vis"]
                payload[f"ids_{v_idx}"]      = r["ids"]
                payload[f"modality_{v_idx}"] = np.array(r.get("modality", ""))
                payload[f"focal_{v_idx}"]    = float(r.get("focal", 0.0))
            payload["view_angles_deg"] = np.asarray(scene.knobs["view_angles_deg"], dtype=np.float32)
            payload["view_modalities"]  = np.asarray(scene.view_modalities)
            cov = coverage_summary(renders, n_linked=len(scene.linked))
            coverage_accum.append(cov)

        label_counts[int(label)] = label_counts.get(int(label), 0) + 1

        if out_dir is not None:
            path = os.path.join(out_dir, f"scene_{idx:06d}.npz")
            # NPZ does not natively hold strings; store small metadata
            # via numpy 0-d arrays of object dtype.
            np.savez(path,
                     label=int(label),
                     label_kind=np.array(default_label),
                     operating_point=np.array(gen.knobs.get("_name", "custom")),
                     **payload)
            files.append(path)
        else:
            files.append(None)

        if verbose and (idx + 1) % max(1, n_scenes // 10) == 0:
            dt = time.time() - t0
            print(f"  [{idx + 1:6d}/{n_scenes}]  elapsed = {dt:.1f}s  "
                  f"({(idx + 1) / dt:.1f} scenes/s)")

    summary = {
        "n_scenes": n_scenes,
        "dataset": dataset,
        "operating_point": gen.knobs.get("_name", "custom"),
        "label_kind": default_label,
        "label_distribution": label_counts,
        "augmenter_pipeline": [type(a).__name__ for a in pipeline.augmenters],
        "image_size": image_size,
        "elapsed_seconds": time.time() - t0,
    }
    if coverage_accum:
        cov_fracs = [c["coverage_frac"] for c in coverage_accum]
        summary["coverage_frac_mean"] = float(np.mean(cov_fracs))
        summary["coverage_frac_min"]  = float(np.min(cov_fracs))

    if out_dir is not None and write_manifest:
        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(summary, f, indent=2)
        if verbose:
            print(f"  wrote manifest → {manifest_path}")

    summary["files"] = files
    return summary


# ===========================================================================
# CLI
# ===========================================================================

def _build_pipeline_from_args(args) -> List[Augmenter]:
    """Compose the augmenter pipeline from CLI flags. Order matters:

        random subsample → occlusion → image-FOV crop → noise

    so noise is applied LAST (to whatever pixels are still observable).
    """
    pipeline = []
    if args.subsample is not None and args.subsample < 1.0:
        pipeline.append(RandomSubsampleAugmenter(args.subsample))
    if args.occlude is not None and args.occlude > 0.0:
        pipeline.append(CenterOcclusionAugmenter(args.occlude))
    if args.fov_bbox is not None:
        bbox = tuple(float(v) for v in args.fov_bbox.split(","))
        if len(bbox) != 4:
            raise SystemExit("--fov-bbox needs 4 comma-separated floats")
        pipeline.append(LimitedFOVAugmenter(bbox))
    if args.noise_sigma is not None and args.noise_sigma > 0.0:
        pipeline.append(GaussianNoiseAugmenter(args.noise_sigma))
    if args.salt_pepper is not None and args.salt_pepper > 0.0:
        pipeline.append(SaltPepperNoiseAugmenter(args.salt_pepper))
    return pipeline


def main():
    ap = argparse.ArgumentParser(
        description="Generate correspondence_bench data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset", choices=["static", "video", "multiview"],
                     required=True)
    ap.add_argument("--operating-point", required=True,
                     help="One of easy/basic/hard/extreme/adversarial; or for "
                          "video, also slow_only/fast_only/mixed_hz/multiscale_hz")
    ap.add_argument("--n", type=int, default=100, help="Number of scenes")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--label-kind", default=None)
    ap.add_argument("--label-K", type=int, default=4)

    # Augmenter knobs (independent of operating point)
    ap.add_argument("--noise-sigma", type=float, default=None,
                     help="Gaussian noise σ as fraction of 255 (e.g. 0.05)")
    ap.add_argument("--salt-pepper", type=float, default=None,
                     help="Per-pixel salt+pepper corruption probability")
    ap.add_argument("--subsample", type=float, default=None,
                     help="Fraction of pixels to KEEP (e.g. 0.4 = drop 60%%)")
    ap.add_argument("--occlude", type=float, default=None,
                     help="Centered occlusion fraction (e.g. 0.3 = 30%%×30%% mask)")
    ap.add_argument("--fov-bbox", default=None,
                     help="Image-space FOV bbox, x0,y0,x1,y1 in [0,1] (e.g. '0,0,0.5,1')")

    args = ap.parse_args()
    pipeline = _build_pipeline_from_args(args)
    print(f"Augmenter pipeline: {[type(a).__name__ for a in pipeline]}")

    summary = generate(
        dataset=args.dataset,
        operating_point=args.operating_point,
        n_scenes=args.n,
        image_size=args.image_size,
        augmenters=pipeline,
        label_kind=args.label_kind,
        label_K=args.label_K,
        out_dir=args.out,
        base_seed=args.seed,
        verbose=True,
    )
    print()
    print("Summary:")
    for k, v in summary.items():
        if k == "files": continue
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
