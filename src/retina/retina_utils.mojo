import math

# Utilities for Retina: IoU and normalized center/size differences

fn calcular_iou(var boxA: List[Float32], var boxB: List[Float32]) -> Float32:
    try:
        var ax0 = boxA[0]; var ay0 = boxA[1]; var ax1 = boxA[2]; var ay1 = boxA[3]
        var bx0 = boxB[0]; var by0 = boxB[1]; var bx1 = boxB[2]; var by1 = boxB[3]
        var inter_x0 = ax0
        if bx0 > inter_x0:
            inter_x0 = bx0
        var inter_y0 = ay0
        if by0 > inter_y0:
            inter_y0 = by0
        var inter_x1 = ax1
        if bx1 < inter_x1:
            inter_x1 = bx1
        var inter_y1 = ay1
        if by1 < inter_y1:
            inter_y1 = by1
        var iw = inter_x1 - inter_x0
        var ih = inter_y1 - inter_y0
        if iw <= 0.0 or ih <= 0.0:
            return 0.0
        var inter_area = iw * ih
        var wA = ax1 - ax0
        if wA < 0.0: wA = 0.0
        var hA = ay1 - ay0
        if hA < 0.0: hA = 0.0
        var areaA = wA * hA
        var wB = bx1 - bx0
        if wB < 0.0: wB = 0.0
        var hB = by1 - by0
        if hB < 0.0: hB = 0.0
        var areaB = wB * hB
        var union = areaA + areaB - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union
    except _:
        return 0.0


# Returns (center_distance_normalized, width_rel_error, height_rel_error)
fn calcular_distancias_normalizadas(var pred: List[Float32], var gt: List[Float32], var img_w: Float32, var img_h: Float32) -> (Float32, Float32, Float32):
    try:
        var px0 = pred[0]; var py0 = pred[1]; var px1 = pred[2]; var py1 = pred[3]
        var gx0 = gt[0]; var gy0 = gt[1]; var gx1 = gt[2]; var gy1 = gt[3]
        var pcx = (px0 + px1) * 0.5
        var pcy = (py0 + py1) * 0.5
        var gcx = (gx0 + gx1) * 0.5
        var gcy = (gy0 + gy1) * 0.5
        var dx = pcx - gcx
        var dy = pcy - gcy
        var diag = math.sqrt(Float64(img_w * img_w + img_h * img_h))
        var center_norm = Float32(math.sqrt(Float64(dx * dx + dy * dy)) / (diag + 1e-12))
        var pw = px1 - px0
        if pw < 1e-6: pw = 1e-6
        var ph = py1 - py0
        if ph < 1e-6: ph = 1e-6
        var gw = gx1 - gx0
        if gw < 1e-6: gw = 1e-6
        var gh = gy1 - gy0
        if gh < 1e-6: gh = 1e-6
        var w_rel = Float32(abs(Float64(pw / gw - 1.0)))
        var h_rel = Float32(abs(Float64(ph / gh - 1.0)))
        return (center_norm, w_rel, h_rel)
    except _:
        return (1.0, 1.0, 1.0)
