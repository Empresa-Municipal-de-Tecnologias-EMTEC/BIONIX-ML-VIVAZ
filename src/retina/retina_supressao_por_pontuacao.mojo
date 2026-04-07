import bionix_ml.nucleo.Tensor as tensor_defs

fn intersecao_sobre_uniao(boxA: List[Float32], boxB: List[Float32]) -> Float32:
    # boxes in [x0,y0,x1,y1]
    var xA = max(boxA[0], boxB[0])
    var yA = max(boxA[1], boxB[1])
    var xB = min(boxA[2], boxB[2])
    var yB = min(boxA[3], boxB[3])
    var interW = xB - xA
    var interH = yB - yA
    if interW <= 0.0 or interH <= 0.0:
        return 0.0
    var interArea = interW * interH
    var boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    var boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea)

fn supressao_por_pontuacao(boxes: List[List[Float32]], scores: List[Float32], iou_thresh: Float32 = 0.5) -> List[Int]:
    var idxs: List[Int] = List[Int]()
    for i in range(len(boxes)):
        idxs.append(i)
    # sort by scores desc
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            if scores[idxs[j]] > scores[idxs[i]]:
                var tmp = idxs[i]
                idxs[i] = idxs[j]
                idxs[j] = tmp

    var keep: List[Int] = List[Int]()
    for i in range(len(idxs)):
        var keep_i = True
        var idx_i = idxs[i]
        for k in keep:
            var iou = intersecao_sobre_uniao(boxes[idx_i].copy(), boxes[k].copy())
            if iou > iou_thresh:
                keep_i = False
                break
        if keep_i:
            keep.append(idx_i)
    # transferencia de propriedade de keep para o chamador para evitar cópia implícita
    return keep^
