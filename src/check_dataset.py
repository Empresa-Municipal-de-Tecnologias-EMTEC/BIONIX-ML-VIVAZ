#!/usr/bin/env python3
"""Inspect dataset: BMP dims, box coords, coverage stats."""
import os, struct, glob, sys

BASE = "DATASET"
stats = {"ok": 0, "no_box": 0, "bad_box": 0, "total": 0}
w_ratios = []
h_ratios = []
box_pcts = []  # box area / image area

splits = ["train", "val"]
for split in splits:
    root = os.path.join(BASE, split)
    if not os.path.isdir(root):
        continue
    for cls in sorted(os.listdir(root))[:5]:  # first 5 classes
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        bmps = sorted(glob.glob(os.path.join(cls_dir, "*.bmp")))[:10]
        for bmp_path in bmps:
            stats["total"] += 1
            try:
                d = open(bmp_path, "rb").read()
                img_w = struct.unpack_from("<i", d, 18)[0]
                img_h = abs(struct.unpack_from("<i", d, 22)[0])
            except Exception as e:
                print(f"  BAD_BMP {bmp_path}: {e}")
                continue

            box_path = bmp_path[:-4] + ".box"
            if not os.path.exists(box_path):
                stats["no_box"] += 1
                print(f"  NO_BOX {bmp_path} ({img_w}x{img_h})")
                continue

            try:
                line = open(box_path).read().strip().split("\n")[0]
                parts = line.split()
                x0, y0, x1, y1 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            except Exception as e:
                stats["bad_box"] += 1
                print(f"  BAD_BOX {bmp_path}: '{open(box_path).read().strip()}' -> {e}")
                continue

            # Determine if coords are normalized or pixel
            if x0 <= 1.5 and y0 <= 1.5 and x1 <= 1.5 and y1 <= 1.5:
                # normalized
                px0, py0, px1, py1 = x0*img_w, y0*img_h, x1*img_w, y1*img_h
                fmt = "norm"
            else:
                px0, py0, px1, py1 = x0, y0, x1, y1
                fmt = "pixel"

            bw = px1 - px0
            bh = py1 - py0
            stats["ok"] += 1

            wr = bw / img_w if img_w > 0 else 0
            hr = bh / img_h if img_h > 0 else 0
            area_pct = (bw * bh) / (img_w * img_h) if img_w * img_h > 0 else 0
            w_ratios.append(wr)
            h_ratios.append(hr)
            box_pcts.append(area_pct)

            ok = "OK" if 0.05 < wr < 1.05 and 0.05 < hr < 1.05 else "WARN"
            print(f"  [{ok}] {split}/{cls}/{os.path.basename(bmp_path)} img={img_w}x{img_h} box=[{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}] fmt={fmt} bw={bw:.0f}xbh={bh:.0f} wr={wr:.2f} hr={hr:.2f}")

print()
print(f"=== STATS: total={stats['total']} ok={stats['ok']} no_box={stats['no_box']} bad_box={stats['bad_box']} ===")
if w_ratios:
    print(f"  w_ratio: min={min(w_ratios):.3f} max={max(w_ratios):.3f} avg={sum(w_ratios)/len(w_ratios):.3f}")
    print(f"  h_ratio: min={min(h_ratios):.3f} max={max(h_ratios):.3f} avg={sum(h_ratios)/len(h_ratios):.3f}")
    print(f"  area%:   min={min(box_pcts)*100:.1f}% max={max(box_pcts)*100:.1f}% avg={sum(box_pcts)/len(box_pcts)*100:.1f}%")
    # boxes that cover > 90% of image
    huge = sum(1 for p in box_pcts if p > 0.9)
    print(f"  boxes covering >90% image area: {huge}/{len(box_pcts)}")
