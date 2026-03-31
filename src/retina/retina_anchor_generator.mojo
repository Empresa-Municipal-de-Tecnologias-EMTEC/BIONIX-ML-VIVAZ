import bionix_ml.nucleo.Tensor as tensor_defs
import math


# small helper to detect NaN/Inf or absurd values
fn _num_sane(var v: Float32) -> Bool:
    try:
        if v != v:
            return False
        if v < -1e12 or v > 1e12:
            return False
        return True
    except _:
        return False

fn gerar_anchors(input_size: Int = 640, strides: List[Int] = [8,16,32,64],
                 scales: List[List[Float32]] = List[List[Float32]](), ratios: List[Float32] = [0.5, 1.0, 2.0]) -> List[List[Float32]]:
    # Gera anchors como [cx, cy, w, h] em pixels para cada nível.
    var out: List[List[Float32]] = List[List[Float32]]()
    var scales_local: List[List[Float32]] = scales.copy()
    if len(scales_local) == 0:
        var default_scales: List[List[Float32]] = List[List[Float32]]()
        for s in strides:
            var lvl: List[Float32] = List[Float32]()
            lvl.append(Float32(s) * 4.0)
            default_scales.append(lvl^)
        scales_local = default_scales^

    for idx in range(len(strides)):
        var stride = strides[idx]
        var lvl_scales = scales_local[idx].copy()
        var feat_size = input_size // stride
        for i in range(feat_size):
            var cy = Float32((i + 0.5) * stride)
            for j in range(feat_size):
                var cx = Float32((j + 0.5) * stride)
                for sc in lvl_scales:
                    for r in ratios:
                        var w = sc * Float32(math.sqrt(r))
                        var h = sc / Float32(math.sqrt(r))
                        var a: List[Float32] = List[Float32]()
                        a.append(cx); a.append(cy); a.append(w); a.append(h)
                        # sanity-check anchor components before appending
                        var ok = True
                        for vv in a:
                            if not _num_sane(vv):
                                ok = False
                                break
                        if not ok:
                            try:
                                print("[DBG] gerar_anchors: skipping corrupted anchor at stride", stride, "cx,cy,w,h=", cx, cy, w, h)
                            except _:
                                print("[DBG] gerar_anchors: skipping corrupted anchor (unable to format values)")
                            continue
                        out.append(a^)
    scales_local = scales_local^
    return out^
