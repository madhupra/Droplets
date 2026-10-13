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


def enhance_fitc(image_path, out_path="fitc_enhanced.png", low_pct=1, high_pct=99.0,
                 gamma=0.5):
    """Contrast-stretch a single-channel FITC TIFF and save it as a green image.

    Intensities are clipped to the [low_pct, high_pct] percentiles, rescaled to
    0-1, gamma-adjusted, and mapped to the green channel. Returns the RGB array.
    """
    img = np.squeeze(tifffile.imread(image_path)).astype(float)
    lo, hi = np.percentile(img, [low_pct, high_pct])
    img = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1) ** gamma
    rgb = np.zeros(img.shape + (3,))
    rgb[..., 1] = img
    plt.imsave(out_path, rgb)
    return rgb


def make_toggle_viewer(brightfield_path, fitc_path, out_html="viewer.html",
                       fitc_kwargs=None, overlay_floor_pct=97):
    """Write a standalone HTML viewer that overlays the FITC image on the
    brightfield image.

    Buttons (or the keys 1-4 and space) switch between Brightfield, the original
    FITC image, an overlay using the original FITC, and an overlay using a copy
    of the FITC with everything below the `overlay_floor_pct` percentile removed
    (signal only). Sliders set the FITC gain and brightfield brightness. Both images must
    cover the same field of view and have the same shape.
    """
    import base64
    import io

    def to_b64(rgb):
        buf = io.BytesIO()
        plt.imsave(buf, rgb, format="png")
        return base64.b64encode(buf.getvalue()).decode()

    bf = np.squeeze(tifffile.imread(brightfield_path)).astype(float)
    lo, hi = np.percentile(bf, [0.5, 99.5])
    bf = np.clip((bf - lo) / max(hi - lo, 1e-9), 0, 1)
    tmp = os.path.splitext(out_html)[0] + "_fitc.png"
    fl = enhance_fitc(fitc_path, tmp, **(fitc_kwargs or {}))
    kw = dict(fitc_kwargs or {})
    kw.update(low_pct=overlay_floor_pct)  # background-subtracted copy for the overlay
    fl_ov = enhance_fitc(fitc_path, tmp, **kw)
    os.remove(tmp)
    if bf.shape != fl.shape[:2]:
        raise ValueError(f"Image shapes differ: {bf.shape} vs {fl.shape[:2]}")
    bf_b64 = to_b64(np.dstack([bf] * 3))
    fl_b64 = to_b64(fl)
    fl_ov_b64 = to_b64(fl_ov)

    html = """<!doctype html><meta charset="utf-8"><title>Droplet viewer</title>
<style>
body{background:#111;color:#eee;font:14px sans-serif;margin:12px}
button{padding:6px 14px;margin-right:6px;background:#333;color:#eee;border:1px solid #666;cursor:pointer}
button.on{background:#2a7;border-color:#2a7}
canvas{width:min(95vw,90vh);display:block;margin-top:10px}
</style>
<div><button id="b1">1 Brightfield</button><button id="b2">2 FITC (original)</button>
<button id="b3">3 Overlay: original FITC</button><button id="b4">4 Overlay: FITC signal only</button></div>
<div style="margin-top:8px">FITC gain <input id="op" type="range" min="0" max="1" step="0.05" value="1">
&nbsp; Brightfield brightness <input id="bk" type="range" min="0.1" max="1" step="0.05" value="0.55">
&nbsp; (keys 1-4; space = toggle brightfield/FITC)</div>
<canvas id="c"></canvas>
<script>
const load=src=>new Promise(r=>{const i=new Image();i.onload=()=>r(i);i.src=src;});
const c=document.getElementById('c'),ctx=c.getContext('2d'),op=document.getElementById('op'),bk=document.getElementById('bk'),NB=4;
let B,F,G,mode=4;
function pix(img){c.width=img.width;c.height=img.height;ctx.drawImage(img,0,0);
 return ctx.getImageData(0,0,img.width,img.height);}
function show(m){mode=m;
 for(let i=1;i<=NB;i++)document.getElementById('b'+i).classList.toggle('on',i==m);
 if(!B)return;
 const out=ctx.createImageData(B.width,B.height),o=out.data,b=B.data,f=F.data,h=G.data,g=+op.value,k=+bk.value;
 for(let i=0;i<o.length;i+=4){
  if(m==1){o[i]=b[i];o[i+1]=b[i+1];o[i+2]=b[i+2];}
  else if(m==2){o[i]=f[i];o[i+1]=f[i+1];o[i+2]=f[i+2];}
  else{const s=m==3?f:h;o[i]=b[i]*k;o[i+1]=Math.min(255,b[i+1]*k+s[i+1]*g);o[i+2]=b[i+2]*k;}
  o[i+3]=255;}
 ctx.putImageData(out,0,0);}
Promise.all([load("data:image/png;base64,__BF__"),load("data:image/png;base64,__FL__"),load("data:image/png;base64,__OV__")]).then(([bi,fi,oi])=>{
 B=pix(bi);F=pix(fi);G=pix(oi);show(4);});
for(let i=1;i<=NB;i++)document.getElementById('b'+i).onclick=()=>show(i);
op.oninput=bk.oninput=()=>show(mode<3?4:mode);
document.onkeydown=e=>{if(e.key>='1'&&e.key<='4')show(+e.key);
 if(e.key==' '){e.preventDefault();show(mode==1?2:1);}};
</script>""".replace("__BF__", bf_b64).replace("__FL__", fl_b64).replace("__OV__", fl_ov_b64)
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
