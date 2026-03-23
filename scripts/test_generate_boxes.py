#!/usr/bin/env python3
"""\nTeste rápido: gera arquivos .box para as primeiras imagens BMP encontradas
Usa parsing mínimo do header BMP para obter largura/altura e calcula crop central igual a heurística Mojo.
"""
from pathlib import Path
import struct

ROOT = Path('src/DATASET')
COUNT = 5

def read_bmp_size(path: Path):
    with open(path, 'rb') as f:
        data = f.read(30)
    if len(data) < 30:
        return None, None
    # width at offset 18 (int32 little), height at 22
    width = struct.unpack_from('<i', data, 18)[0]
    height = struct.unpack_from('<i', data, 22)[0]
    if width < 0:
        width = -width
    if height < 0:
        height = -height
    return width, height


def center_crop_bbox(w, h, frac=0.7):
    short = w if w < h else h
    crop_size = int(short * frac)
    if crop_size <= 0:
        return 0,0,w,h
    cx = w // 2
    cy = h // 2
    half = crop_size // 2
    x0 = cx - half
    y0 = cy - half
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    x1 = x0 + crop_size
    y1 = y0 + crop_size
    if x1 > w:
        x1 = w
        x0 = max(0, w - crop_size)
    if y1 > h:
        y1 = h
        y0 = max(0, h - crop_size)
    return x0,y0,x1,y1


def main():
    n = 0
    for p in ROOT.rglob('*.bmp'):
        if n >= COUNT:
            break
        w,h = read_bmp_size(p)
        if w is None:
            continue
        bbox = center_crop_bbox(w,h)
        box_path = p.with_suffix('.box')
        box_path.write_text(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")
        print(f"Wrote {box_path} -> {bbox}")
        n += 1

if __name__ == '__main__':
    main()
