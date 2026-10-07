#!/usr/bin/env python3
"""Detect circular objects (droplets) in a TIFF image, let the user pick one
as the size reference, and write an annotated image plus a table of labels.

Usage:
    python droplets.py image.tif [--min-radius 10] [--max-radius 80]
                       [--tolerance 0.2] [--label N] [--outdir output]
"""
import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import tifffile
from skimage import color, draw, feature, filters, transform

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_image(path):
    """Step 1: read a TIFF and return a 2-D float grayscale array in [0, 1]."""
    img = tifffile.imread(path)
    img = np.squeeze(img)
    if img.ndim == 3:
        if img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)  # channels-first -> channels-last
        if img.shape[-1] in (3, 4):
            img = color.rgb2gray(img[..., :3])
        else:
            img = img[0]  # multi-page stack: use first frame
    img = img.astype(float)
    rng = np.ptp(img)
    return (img - img.min()) / rng if rng else np.zeros_like(img)


def detect_circles(gray, min_r, max_r, max_circles=200):
    """Step 2: find circular objects with a Hough transform.

    Returns a list of (cx, cy, r) sorted top-to-bottom, left-to-right.
    """
    smooth = filters.gaussian(gray, sigma=2)
    edges = feature.canny(smooth, sigma=2)
    radii = np.arange(min_r, max_r + 1)
    hspaces = transform.hough_circle(edges, radii)
    _, cx, cy, r = transform.hough_circle_peaks(
        hspaces, radii,
        min_xdistance=min_r, min_ydistance=min_r,
        threshold=0.5 * hspaces.max(), total_num_peaks=max_circles,
    )
    circles = sorted(zip(cx, cy, r), key=lambda c: (c[1] // max(min_r, 1), c[0]))
    return [(int(x), int(y), int(rad)) for x, y, rad in circles]


def build_label_mask(shape, circles):
    """Label image: each circle's pixels = its label (1..N); 0 = background."""
    mask = np.zeros(shape, dtype=np.int32)
    for i, (cx, cy, r) in enumerate(circles, start=1):
        rr, cc = draw.disk((cy, cx), r, shape=shape)
        mask[rr, cc] = i
    return mask


def annotate(gray, circles, labels, path, title):
    """Draw a mask outline around each circle with its label inside."""
    fig, ax = plt.subplots(figsize=(8, 8 * gray.shape[0] / gray.shape[1]))
    ax.imshow(gray, cmap="gray")
    for label, (cx, cy, r) in zip(labels, circles):
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, color="lime", lw=1.5))
        ax.text(cx, cy, str(label), color="yellow", ha="center", va="center",
                fontsize=9, fontweight="bold")
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def choose_label(n, preset=None):
    """Step 3: ask the user which label to use as the reference."""
    if preset is not None:
        if not 1 <= preset <= n:
            sys.exit(f"--label must be between 1 and {n}")
        return preset
    while True:
        try:
            choice = int(input(f"Select a label (1-{n}) to use as the reference size: "))
        except ValueError:
            print("Please enter a whole number.")
            continue
        if 1 <= choice <= n:
            return choice
        print(f"Label must be between 1 and {n}.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="path to a TIFF image")
    p.add_argument("--min-radius", type=int, default=10)
    p.add_argument("--max-radius", type=int, default=80)
    p.add_argument("--tolerance", type=float, default=0.2,
                   help="relative radius tolerance around the selected circle (default 0.2)")
    p.add_argument("--label", type=int, help="skip the prompt and use this label")
    p.add_argument("--outdir", default="output")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    gray = read_image(args.image)                                   # 1
    circles = detect_circles(gray, args.min_radius, args.max_radius)  # 2
    if not circles:
        sys.exit("No circular objects found; try adjusting --min-radius/--max-radius.")
    all_labels = list(range(1, len(circles) + 1))
    overview = os.path.join(args.outdir, "all_labeled.png")
    annotate(gray, circles, all_labels, overview, f"{len(circles)} circles detected")
    print(f"Detected {len(circles)} circular objects. Overview saved to {overview}")

    chosen = choose_label(len(circles), args.label)                  # 3
    ref_r = circles[chosen - 1][2]
    lo, hi = ref_r * (1 - args.tolerance), ref_r * (1 + args.tolerance)

    # 4 & 5: annotated image and table are built from the same selection.
    keep = [(i, c) for i, c in enumerate(circles, start=1) if lo <= c[2] <= hi]
    labels = [i for i, _ in keep]
    sel = [c for _, c in keep]

    out_img = os.path.join(args.outdir, "selected.png")
    annotate(gray, sel, labels, out_img,
             f"Circles matching label {chosen} (radius {ref_r}px ±{args.tolerance:.0%})")

    table = pd.DataFrame({
        "label": labels,
        "center_x": [c[0] for c in sel],
        "center_y": [c[1] for c in sel],
        "radius_px": [c[2] for c in sel],
        "area_px": [int(np.pi * c[2] ** 2) for c in sel],
    })
    out_csv = os.path.join(args.outdir, "labels.csv")
    table.to_csv(out_csv, index=False)

    print(f"\nReference: label {chosen}, radius {ref_r}px\n")
    print(table.to_string(index=False))
    print(f"\nSaved {out_img} and {out_csv}")


if __name__ == "__main__":
    main()
