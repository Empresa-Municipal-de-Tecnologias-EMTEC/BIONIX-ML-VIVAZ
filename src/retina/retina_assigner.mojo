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
            var tx = (gt_cx - ax) / aw
            var ty = (gt_cy - ay) / ah
            var tw = Float32(math.log(gt_w / aw + 1e-6))
            var th = Float32(math.log(gt_h / ah + 1e-6))
            var t: List[Float32] = List[Float32]()
            t.append(tx); t.append(ty); t.append(tw); t.append(th)
            targets[a_idx] = t^
        elif best_iou < iou_neg:
            labels[a_idx] = 0
        else:
            labels[a_idx] = -1 # ignore

    # return result struct transferring ownership of lists
    return AssignResult(labels^, targets^)
