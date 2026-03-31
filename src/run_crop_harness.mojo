import bionix_ml.dados as dados_pkg
import bionix_ml.graficos as graficos_pkg
import retina.retina_anchor_generator as ag
import os

# Focused harness: load a BMP, generate anchors for `input_size`,
# crop/resize the image around selected anchors and print diagnostics.

fn main() raises:
    var image_path: String = ""  # optionally set a path here
    var input_size: Int = 320
    var anchors_to_check = [6096, 6130]

    # Try to auto-find a .bmp in likely dataset directories
    var candidates = ["DATASET", "./DATASET", "../DATASET", "./", "../"]
    if len(image_path) == 0:
        for d in candidates:
            try:
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        if f.endswith('.bmp'):
                            image_path = os.path.join(d, f)
                            break
                if len(image_path) > 0:
                    break
            except _:
                pass

    if len(image_path) == 0:
        # fallback: try an example BMP from the examples tree
        image_path = "../../BIONIX-ML-EXEMPLOS/src/e000008_reconhecimento_facial/dataset/treino/0/0.bmp"
        print("[HARN] no image_path found in candidates; falling back to", image_path)

    print("[HARN] using image:", image_path)

    var info = dados_pkg.carregar_bmp_rgb(image_path)
    if info.width <= 0 or info.height <= 0:
        print("[HARN] failed to load BMP or empty image")
        return

    # Ensure flat buffer is present for fast access
    try:
        dados_pkg.bmp.ensure_flat(info)
    except _:
        pass

    print("[HARN] bmp dims -> w", info.width, "h", info.height, "channels", info.channels, "flat_len", len(info.flat_pixels))

    # Generate anchors for the network input size
    var anchors = ag.gerar_anchors(input_size)
    print("[HARN] gerar_anchors -> count", len(anchors), "using input_size", input_size)

    for a_idx in anchors_to_check:
        if a_idx < 0 or a_idx >= len(anchors):
            print("[HARN] a_idx out of range:", a_idx)
            continue
        var a = anchors[a_idx].copy()
        if len(a) < 4:
            print("[HARN] malformed anchor at", a_idx)
            continue
        var cx = a[0]; var cy = a[1]; var aw = a[2]; var ah = a[3]
        print("[HARN] anchor", a_idx, "cx,cy,w,h ->", cx, cy, aw, ah)

        var x0 = Int(cx - aw / 2.0)
        var y0 = Int(cy - ah / 2.0)
        var x1 = Int(cx + aw / 2.0)
        var y1 = Int(cy + ah / 2.0)

        if x0 < 0: x0 = 0
        if y0 < 0: y0 = 0
        if x1 >= info.width: x1 = info.width - 1
        if y1 >= info.height: y1 = info.height - 1

        print("[HARN] crop coords -> x0", x0, "y0", y0, "x1", x1, "y1", y1)

        # Run crop+resize from flat buffer (resized patch dims chosen small to inspect)
        var out_h = 32
        var out_w = 32
        try:
            var patch = graficos_pkg.bmp.crop_and_resize_from_flat(info.flat_pixels.copy(), info.width, info.height, info.channels, x0, y0, x1, y1, out_h, out_w)
            print("[HARN] patch dims -> h", len(patch), "w", (len(patch[0]) if len(patch) > 0 else 0))
            # print center pixel of patch
            if len(patch) > 0 and len(patch[0]) > 0 and len(patch[0][0]) >= 3:
                var mid_y = len(patch) // 2
                var mid_x = len(patch[0]) // 2
                var mid_px = patch[mid_y][mid_x]
                print("[HARN] patch center pixel (r,g,b):", mid_px[0], mid_px[1], mid_px[2])
        except _:
            print("[HARN] crop_and_resize_from_flat raised an exception for anchor", a_idx)

        # Also inspect the raw flat pixel at the anchor center (image coordinates)
        var img_x = Int(cx)
        var img_y = Int(cy)
        if img_x < 0: img_x = 0
        if img_y < 0: img_y = 0
        if img_x >= info.width: img_x = info.width - 1
        if img_y >= info.height: img_y = info.height - 1
        try:
            var center_vals = graficos_pkg.bmp.get_flat_pixel_float(info.flat_pixels, info.width, info.height, info.channels, img_y, img_x)
            var cv_strs = List[String]()
            for cv in center_vals:
                cv_strs.append(String(cv))
            print("[HARN] center flat pixel at img_x", img_x, "img_y", img_y, "vals=", String(", ").join(cv_strs.copy()))
        except _:
            print("[HARN] failed to read center flat pixel")

    print("[HARN] done")
