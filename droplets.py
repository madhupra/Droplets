#!/usr/bin/env python3
"""Detect circular objects (droplets) in a TIFF image, let the user pick one
as the size reference, and write an annotated image plus a table of labels.

Use from other code:
    from droplets import analyze_droplets
    result = analyze_droplets("image.tif", min_radius=45, max_radius=75)

Use from the command line:
    python droplets.py image.tif [--min-radius 10] [--max-radius 80]
                       [--tolerance 0.1] [--label N] [--outdir output]
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
import tifffile
from skimage import color, draw, feature, filters, transform

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def analyze_droplets(image_path, min_radius=10, max_radius=80, tolerance=0.1,
                     label=None, outdir="output", max_circles=200, verbose=True):
    """Run the full droplet workflow on one TIFF image.

    1. Read the TIFF.
    2. Detect complete circular objects and label them 1..N.
    3. Ask the user for a reference label (skipped if `label` is given).
    4. Save an image outlining every circle whose radius is within `tolerance`
       of the reference, with its number inside.
    5. Build and save a table of those labels.

    Returns a dict with keys: table (DataFrame), circles (all detected
    (cx, cy, r) tuples), label_mask (int image, 0 = background, k = label k for
    the selected circles), reference_label, reference_radius, overview_path,
    selected_path, table_path.
    Raises ValueError if no circles are found or `label` is out of range.
    """

    def read_image(path):
        img = np.squeeze(tifffile.imread(path))
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

    def refine_radius(gray, cx, cy, r, span=12):
        # Hough peaks can land on either edge of the dark ring around a droplet,
        # which makes equally sized droplets look different; snap to the ring's
        # darkest radius, which is consistent.
        h, w = gray.shape
        y0, y1 = max(cy - r - span, 0), min(cy + r + span + 1, h)
        x0, x1 = max(cx - r - span, 0), min(cx + r + span + 1, w)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xx - cx, yy - cy)
        patch = gray[y0:y1, x0:x1]
        ks = np.arange(r - span, r + span + 1)
        prof = [patch[(dist >= k - 0.5) & (dist < k + 0.5)].mean() for k in ks]
        return int(ks[int(np.argmin(prof))])

    def detect_circles(gray):
        # Hough transform; circles cut off by the image border are discarded.
        edges = feature.canny(filters.gaussian(gray, sigma=2), sigma=2)
        radii = np.arange(min_radius, max_radius + 1)
        hspaces = transform.hough_circle(edges, radii)
        _, cx, cy, r = transform.hough_circle_peaks(
            hspaces, radii,
            min_xdistance=min_radius, min_ydistance=min_radius,
            threshold=0.5 * hspaces.max(), total_num_peaks=max_circles,
        )
        h, w = gray.shape
        complete = []
        for x, y, rad in zip(cx, cy, r):
            rad = refine_radius(gray, int(x), int(y), int(rad))
            if x - rad >= 0 and y - rad >= 0 and x + rad < w and y + rad < h:
                complete.append((int(x), int(y), int(rad)))
        # Number top-to-bottom, left-to-right.
        return sorted(complete, key=lambda c: (c[1] // max(min_radius, 1), c[0]))

    def annotate(gray, circles, labels, path, title):
        fig, ax = plt.subplots(figsize=(8, 8 * gray.shape[0] / gray.shape[1]))
        ax.imshow(gray, cmap="gray")
        for lab, (cx, cy, r) in zip(labels, circles):
            ax.add_patch(plt.Circle((cx, cy), r, fill=False, color="lime", lw=1.5))
            ax.text(cx, cy, str(lab), color="yellow", ha="center", va="center",
                    fontsize=9, fontweight="bold")
        ax.set_title(title)
        ax.axis("off")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def choose_label(n):
        if label is not None:
            if not 1 <= label <= n:
                raise ValueError(f"label must be between 1 and {n}")
            return label
        while True:
            try:
                choice = int(input(f"Select a label (1-{n}) to use as the reference size: "))
            except ValueError:
                print("Please enter a whole number.")
                continue
            if 1 <= choice <= n:
                return choice
            print(f"Label must be between 1 and {n}.")

    os.makedirs(outdir, exist_ok=True)

    gray = read_image(image_path)                                        # 1
    circles = detect_circles(gray)                                       # 2
    if not circles:
        raise ValueError("No circular objects found; try adjusting min_radius/max_radius.")
    overview_path = os.path.join(outdir, "all_labeled.png")
    annotate(gray, circles, range(1, len(circles) + 1), overview_path,
             f"{len(circles)} circles detected")
    if verbose:
        print(f"Detected {len(circles)} circular objects. Overview saved to {overview_path}")

    chosen = choose_label(len(circles))                                  # 3
    ref_r = circles[chosen - 1][2]
    lo, hi = ref_r * (1 - tolerance), ref_r * (1 + tolerance)
    keep = [(i, c) for i, c in enumerate(circles, start=1) if lo <= c[2] <= hi]
    labels = [i for i, _ in keep]
    sel = [c for _, c in keep]

    selected_path = os.path.join(outdir, "selected.png")                 # 4
    annotate(gray, sel, labels, selected_path,
             f"Circles matching label {chosen} (radius {ref_r}px ±{tolerance:.0%})")

    label_mask = np.zeros(gray.shape, dtype=np.int32)
    for lab, (cx, cy, r) in zip(labels, sel):
        rr, cc = draw.disk((cy, cx), r, shape=gray.shape)
        label_mask[rr, cc] = lab

    table = pd.DataFrame({                                               # 5
        "label": labels,
        "center_x": [c[0] for c in sel],
        "center_y": [c[1] for c in sel],
        "radius_px": [c[2] for c in sel],
        "area_px": [int(np.pi * c[2] ** 2) for c in sel],
    })
    table_path = os.path.join(outdir, "labels.csv")
    table.to_csv(table_path, index=False)

    if verbose:
        print(f"\nReference: label {chosen}, radius {ref_r}px\n")
        print(table.to_string(index=False))
        print(f"\nSaved {selected_path} and {table_path}")

    return {
        "table": table, "circles": circles, "label_mask": label_mask,
        "reference_label": chosen, "reference_radius": ref_r,
        "overview_path": overview_path, "selected_path": selected_path,
        "table_path": table_path,
    }


def enhance_fitc(image_path, out_path="fitc_enhanced.png", low_pct=50, high_pct=99.8,
                 gamma=0.8, smooth=0.7):
    """Contrast-stretch a single-channel FITC TIFF and save it as a green image.

    The image is lightly smoothed (`smooth` = Gaussian sigma in px, 0 to disable),
    clipped to the [low_pct, high_pct] percentiles (the low value sits at the
    background so it renders black), rescaled to 0-1, gamma-adjusted, and mapped
    to the green channel. Returns the RGB array.
    """
    img = np.squeeze(tifffile.imread(image_path)).astype(float)
    if smooth:
        img = filters.gaussian(img, sigma=smooth, preserve_range=True)
    lo, hi = np.percentile(img, [low_pct, high_pct])
    img = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1) ** gamma
    rgb = np.zeros(img.shape + (3,))
    rgb[..., 1] = img
    plt.imsave(out_path, rgb)
    return rgb


def make_toggle_viewer(brightfield_path, fitc_path, out_html="viewer.html",
                       fitc_kwargs=None,
                       bf_brightness=0.55, save_pngs=True):
    """Write a standalone HTML viewer that switches between three precomputed views:
    brightfield, the contrast-adjusted FITC image (see `enhance_fitc`), and an
    overlay of the two.

    Click a button to switch views. The views are rendered here in Python and the
    page is pure HTML/CSS (radio buttons), with no JavaScript, so it works in
    sandboxed previews. With `save_pngs`, each view is also written next
    to the HTML as <name>_<view>.png (the page itself embeds smaller
    JPEG copies). Both images must have the same shape.
    """
    import base64
    import io

    bf = np.squeeze(tifffile.imread(brightfield_path)).astype(float)
    lo, hi = np.percentile(bf, [0.5, 99.5])
    bf = np.clip((bf - lo) / max(hi - lo, 1e-9), 0, 1)
    tmp = os.path.splitext(out_html)[0] + "_tmp.png"
    fl = enhance_fitc(fitc_path, tmp, **(fitc_kwargs or {}))
    os.remove(tmp)
    if bf.shape != fl.shape[:2]:
        raise ValueError(f"Image shapes differ: {bf.shape} vs {fl.shape[:2]}")
    bf3 = np.dstack([bf] * 3)
    views = [
        ("brightfield", "1 Brightfield", bf3),
        ("fitc", "2 FITC (contrast adjusted)", fl),
        ("overlay", "3 Overlay: brightfield + FITC",
         np.clip(bf3 * bf_brightness + fl, 0, 1)),
    ]

    base = os.path.splitext(out_html)[0]
    srcs = []
    for name, _, rgb in views:
        buf = io.BytesIO()
        plt.imsave(buf, rgb, format="jpg", pil_kwargs={"quality": 90})
        srcs.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
        if save_pngs:
            plt.imsave(f"{base}_{name}.png", rgb)

    # Pure HTML/CSS toggle (radio buttons), so no JavaScript can fail in sandboxed previews.
    default = 2
    radios = "".join(
        f'<input type="radio" name="view" id="r{i}"{" checked" if i == default else ""}>'
        f'<label for="r{i}">{label}</label>' for i, (_, label, _) in enumerate(views))
    imgs = "".join(f'<img class="v" id="v{i}" src="{src}">' for i, src in enumerate(srcs))
    rules = "".join(
        f"#r{i}:checked~#imgs #v{i}{{display:block}}"
        f"#r{i}:checked+label{{background:#2a7;border-color:#2a7}}" for i in range(len(views)))
    html = f"""<!doctype html><meta charset="utf-8"><title>Droplet viewer</title>
<style>
body{{background:#111;color:#eee;font:14px sans-serif;margin:12px}}
input{{position:absolute;opacity:0;pointer-events:none}}
label{{display:inline-block;padding:6px 14px;margin-right:6px;background:#333;border:1px solid #666;cursor:pointer}}
#imgs{{margin-top:10px}}
.v{{display:none;width:min(95vw,90vh)}}
{rules}
</style>
{radios}
<div id="imgs">{imgs}</div>"""
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="path to a TIFF image")
    p.add_argument("--min-radius", type=int, default=10)
    p.add_argument("--max-radius", type=int, default=80)
    p.add_argument("--tolerance", type=float, default=0.1,
                   help="relative radius tolerance around the selected circle (default 0.1)")
    p.add_argument("--label", type=int, help="skip the prompt and use this label")
    p.add_argument("--outdir", default="output")
    a = p.parse_args()
    try:
        analyze_droplets(a.image, a.min_radius, a.max_radius, a.tolerance,
                         a.label, a.outdir)
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
