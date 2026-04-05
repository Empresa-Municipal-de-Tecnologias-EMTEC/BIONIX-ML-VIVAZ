import bionix_ml.nucleo.Tensor as tensor_defs
import math


# Container to hold assigner results and allow transferring inner lists
struct AssignResult(Movable):
    var labels: List[Int]
    var targets: List[List[Float32]]

    fn __init__(out self, var labels_in: List[Int] = List[Int](), var targets_in: List[List[Float32]] = List[List[Float32]]()):
        self.labels = labels_in^
        self.targets = targets_in^

fn calcular_iou_xywh(box1: List[Float32], box2: List[Float32]) -> Float32:
    # boxes: [cx, cy, w, h]
    try:
        # fast NaN/Inf guard
        for v in box1:
            if v != v:
                return 0.0
        for v in box2:
            if v != v:
                return 0.0
    except _:
        return 0.0
    var x1_min = box1[0] - box1[2] / 2.0
    var y1_min = box1[1] - box1[3] / 2.0
    var x1_max = box1[0] + box1[2] / 2.0
    var y1_max = box1[1] + box1[3] / 2.0

    var x2_min = box2[0] - box2[2] / 2.0
    var y2_min = box2[1] - box2[3] / 2.0
    var x2_max = box2[0] + box2[2] / 2.0
    var y2_max = box2[1] + box2[3] / 2.0

    var inter_xmin = max(x1_min, x2_min)
    var inter_ymin = max(y1_min, y2_min)
    var inter_xmax = min(x1_max, x2_max)
    var inter_ymax = min(y1_max, y2_max)

    var inter_w = inter_xmax - inter_xmin
    var inter_h = inter_ymax - inter_ymin
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    var inter_area = inter_w * inter_h
    var area1 = (x1_max - x1_min) * (y1_max - y1_min)
    var area2 = (x2_max - x2_min) * (y2_max - y2_min)
    return inter_area / (area1 + area2 - inter_area)

fn assignar_anchors(anchors: List[List[Float32]], gt_boxes: List[List[Int]],
                    iou_pos: Float32 = 0.5, iou_neg: Float32 = 0.4) -> AssignResult:
    # gt_boxes are [x0,y0,x1,y1] em pixels (ints)
    var N = len(anchors)
    # quick validation: count invalid anchors
    var invalid_count: Int = 0
    for i in range(N):
        try:
            var a = anchors[i].copy()
            for v in a:
                if v != v:
                    invalid_count = invalid_count + 1
                    break
        except _:
            invalid_count = invalid_count + 1
    if invalid_count > 0:
        try:
            print("[DBG] assignar_anchors: detected", invalid_count, "anchors with NaN; they will be treated as background")
        except _:
            pass
    var labels: List[Int] = List[Int]()
    var targets: List[List[Float32]] = List[List[Float32]]()
    for i in range(N):
        labels.append(0) # background default
        targets.append(List[Float32]())

    if len(gt_boxes) == 0:
        # return an AssignResult transferring the inner lists
        return AssignResult(labels^, targets^)

    for a_idx in range(N):
        var best_iou: Float32 = 0.0
        var best_gt_idx = -1
        for g_idx in range(len(gt_boxes)):
            var gt = gt_boxes[g_idx].copy()
            var gt_cx = Float32((gt[0] + gt[2]) / 2.0)
            var gt_cy = Float32((gt[1] + gt[3]) / 2.0)
            var gt_w = Float32(gt[2] - gt[0])
            var gt_h = Float32(gt[3] - gt[1])
            var a = anchors[a_idx].copy()
            # sanitize anchor components if needed (avoid NaN propagation)
            try:
                for k in range(len(a)):
                    if a[k] != a[k]:
                        a[k] = 0.0
            except _:
                pass
            var iou = calcular_iou_xywh(a, List[Float32](gt_cx, gt_cy, gt_w, gt_h))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
        if best_iou >= iou_pos and best_gt_idx >= 0:
            labels[a_idx] = 1
            # copy GT entry to avoid implicit copy of caller-owned list
            var gt = gt_boxes[best_gt_idx].copy()
            var gt_cx = Float32((gt[0] + gt[2]) / 2.0)
            var gt_cy = Float32((gt[1] + gt[3]) / 2.0)
            var gt_w = Float32(gt[2] - gt[0])
            var gt_h = Float32(gt[3] - gt[1])
            # targets as deltas: tx = (gx - ax)/aw, ty = (gy - ay)/ah, tw = log(gw/aw), th = log(gh/ah)
            var a_local = anchors[a_idx].copy()
            var ax = a_local[0]; var ay = a_local[1]; var aw = a_local[2]; var ah = a_local[3]
            # ignore anchors with near-zero size to avoid extreme targets / divisions
            try:
                if aw < 1e-3 or ah < 1e-3:
                    # leave label as background (0) and continue
                    continue
            except _:
                pass
            var pre_tx = (gt_cx - ax) / aw
            var pre_ty = (gt_cy - ay) / ah
            var tx = pre_tx
            var ty = pre_ty
            # compute scale deltas and clamp targets to safe range to avoid extreme updates
            var pre_tw = Float32(math.log(gt_w / aw + 1e-6))
            var pre_th = Float32(math.log(gt_h / ah + 1e-6))
            var tw = pre_tw
            var th = pre_th
            # clamp translations and scales to decoder's expected ranges
            if tx > 3.0: tx = 3.0
            if tx < -3.0: tx = -3.0
            if ty > 3.0: ty = 3.0
            if ty < -3.0: ty = -3.0
            if tw > 4.0: tw = 4.0
            if tw < -4.0: tw = -4.0
            if th > 4.0: th = 4.0
            if th < -4.0: th = -4.0
            # selective debug prints for extreme pre-clamp targets or tiny anchors
            try:
                if (abs(pre_tx) > 3.0) or (abs(pre_ty) > 3.0) or (abs(pre_tw) > 2.0) or (abs(pre_th) > 2.0) or (aw < 2.0) or (ah < 2.0):
                    try:
                        print("[DBG-ASSIGN] anchor_idx", a_idx, "aw", aw, "ah", ah, "pre_tx", pre_tx, "pre_ty", pre_ty, "pre_tw", pre_tw, "pre_th", pre_th, "post_tx", tx, "post_ty", ty, "post_tw", tw, "post_th", th)
                    except _:
                        pass
            except _:
                pass
            var t: List[Float32] = List[Float32]()
            t.append(tx); t.append(ty); t.append(tw); t.append(th)
            targets[a_idx] = t^
        elif best_iou < iou_neg:
            labels[a_idx] = 0
        else:
            labels[a_idx] = -1 # ignore

    # return result struct transferring ownership of lists
    # Ensure every GT has at least one positive anchor: force-assign the best anchor per GT
    try:
        for g_idx in range(len(gt_boxes)):
            var best_a = -1
            var best_a_iou: Float32 = 0.0
            var gt = gt_boxes[g_idx].copy()
            var gt_cx = Float32((gt[0] + gt[2]) / 2.0)
            var gt_cy = Float32((gt[1] + gt[3]) / 2.0)
            var gt_w = Float32(gt[2] - gt[0])
            var gt_h = Float32(gt[3] - gt[1])
            for a_idx2 in range(N):
                try:
                    var a2 = anchors[a_idx2].copy()
                    var iou2 = calcular_iou_xywh(a2, List[Float32](gt_cx, gt_cy, gt_w, gt_h))
                    if iou2 > best_a_iou:
                        best_a_iou = iou2
                        best_a = a_idx2
                except _:
                    pass
            if best_a >= 0:
                # compute targets for this anchor (same as above)
                try:
                    var a_local = anchors[best_a].copy()
                    var ax = a_local[0]; var ay = a_local[1]; var aw = a_local[2]; var ah = a_local[3]
                    if aw >= 1e-3 and ah >= 1e-3:
                        var tx = (gt_cx - ax) / aw
                        var ty = (gt_cy - ay) / ah
                        var tw = Float32(math.log(gt_w / aw + 1e-6))
                        var th = Float32(math.log(gt_h / ah + 1e-6))
                        if tx > 3.0: tx = 3.0
                        if tx < -3.0: tx = -3.0
                        if ty > 3.0: ty = 3.0
                        if ty < -3.0: ty = -3.0
                        if tw > 4.0: tw = 4.0
                        if tw < -4.0: tw = -4.0
                        if th > 4.0: th = 4.0
                        if th < -4.0: th = -4.0
                        var t: List[Float32] = List[Float32]()
                        t.append(tx); t.append(ty); t.append(tw); t.append(th)
                        labels[best_a] = 1
                        targets[best_a] = t^
                except _:
                    pass
    except _:
        pass

    return AssignResult(labels^, targets^)
