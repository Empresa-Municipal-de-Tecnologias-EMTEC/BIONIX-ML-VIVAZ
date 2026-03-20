#!/usr/bin/env python3
"""Recursively convert images to BMP inside a dataset directory.

Usage examples:
  python convert_images_to_bmp.py                # uses src/DATASET
  python convert_images_to_bmp.py --dataset DATASET
  python convert_images_to_bmp.py --dataset src/DATASET --overwrite

Options:
  --dataset PATH        Dataset root (default: src/DATASET)
  --overwrite           Overwrite existing .bmp files
  --remove-originals    Remove original files after successful conversion
  --dry-run             Show what would be done without writing files
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

try:
    from PIL import Image
except Exception:
    print("Pillow is required. Install with: pip install pillow", file=sys.stderr)
    sys.exit(1)


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.gif', '.bmp', '.heic', '.heif'}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def convert_file(path: Path, overwrite: bool, dry_run: bool) -> bool:
    target = path.with_suffix('.bmp')
    if path.suffix.lower() == '.bmp':
        return False
    if target.exists() and not overwrite:
        return False

    if dry_run:
        print(f"[DRY] Convert: {path} -> {target}")
        return True

    try:
        with Image.open(path) as im:
            # Convert to RGB to ensure compatibility with BMP
            if im.mode not in ('RGB', 'L'):
                im = im.convert('RGB')
            im.save(target, format='BMP')
        print(f"Converted: {path} -> {target}")
        return True
    except Exception as e:
        print(f"Failed: {path} ({e})", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description='Convert images to BMP recursively')
    parser.add_argument('--dataset', '-d', default='src/DATASET', help='Dataset root dir (default: src/DATASET)')
    parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite existing BMP files')
    parser.add_argument('--dry-run', action='store_true', help='Do not write files; just show actions')

    args = parser.parse_args()

    base = Path(args.dataset)
    if not base.is_absolute():
        base = Path(__file__).parent.joinpath(base).resolve()

    if not base.exists():
        print(f"Dataset directory not found: {base}", file=sys.stderr)
        return 2

    converted = 0
    skipped = 0
    failed = 0

    for root, dirs, files in os.walk(base):
        root_path = Path(root)
        for name in files:
            p = root_path.joinpath(name)
            if not is_image_file(p):
                skipped += 1
                continue
            success = convert_file(p, overwrite=args.overwrite, dry_run=args.dry_run)
            if success:
                converted += 1
                if not args.dry_run:
                    try:
                        p.unlink()
                    except Exception as e:
                        print(f"Warning: could not remove original {p}: {e}", file=sys.stderr)
            else:
                failed += 1

    print(f"Done. Converted: {converted}, Failed: {failed}, Skipped(non-images): {skipped}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
