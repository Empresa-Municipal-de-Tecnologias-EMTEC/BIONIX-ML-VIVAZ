import bionix_ml.nucleo.Tensor as tensor_defs
import math

#Container para manter os resultados do assigner e permitir a transferência das listas internas
struct ResultadoAssociado(Movable):
    var labels: List[Int]
    var targets: List[List[Float32]]

    fn __init__(out self, var labels_in: List[Int] = List[Int](), var targets_in: List[List[Float32]] = List[List[Float32]]()):
        self.labels = labels_in^
        self.targets = targets_in^

#Interseção sobre União (IoU) para boxes no formato [cx, cy, w, h]
#A interseção sobre a união retorna um valor entre 0 e 1 indicando o grau de sobreposição entre as duas caixas, onde 0 significa sem sobreposição e 1 significa sobreposição perfeita. Esta função é usada para determinar quais âncoras estão próximas o suficiente dos GT boxes para serem consideradas positivas durante a associação.
#A entrada é composta por duas caixas, cada uma representada por uma lista de 4 floats: [cx, cy, w, h], onde cx e cy são as coordenadas do centro da caixa, e w e h são a largura e altura da caixa, respectivamente. A função calcula as coordenadas dos cantos das caixas, determina a área de interseção e a área de união, e retorna a razão entre a área de interseção e a área de união como o valor de IoU. A função também inclui verificações para lidar com casos em que as caixas não se sobrepõem ou quando os valores de entrada são inválidos (como NaN ou Inf), retornando 0.0 nesses casos para evitar propagação de erros.
fn calcular_iou_xywh(box1: List[Float32], box2: List[Float32]) -> Float32:
    # boxes: [cx, cy, w, h]
    try:
        # proteção contra NaN/Inf
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

fn associar_ancoras(ancoras: List[List[Float32]], caixas_anotadas: List[List[Int]],
                    iou_pos: Float32 = 0.5, iou_neg: Float32 = 0.4, var reg_weights: List[Float32] = List[Float32]()) -> ResultadoAssociado:
    # caixas_anotadas são [x0,y0,x1,y1] em pixels (ints)
    var N = len(ancoras)
    # quick validation: count invalid anchors
    var invalid_count: Int = 0
    for i in range(N):
        try:
            var a = ancoras[i].copy()
            for v in a:
                if v != v:
                    invalid_count = invalid_count + 1
                    break
        except _:
            invalid_count = invalid_count + 1
    if invalid_count > 0:
        try:
            print("[DBG] associar_ancoras: detectado", invalid_count, "ancoras NaN; elas serão tratadas como background")
        except _:
            pass
    var labels: List[Int] = List[Int]()
    var targets: List[List[Float32]] = List[List[Float32]]()
    for i in range(N):
        labels.append(0) # background default
        targets.append(List[Float32]())

    if len(caixas_anotadas) == 0:
        # retorna um ResultadoAssociado transferindo as listas internas, sem cópia adicional
        return ResultadoAssociado(labels^, targets^)

    for a_idx in range(N):
        var best_iou: Float32 = 0.0
        var best_gt_idx = -1
        for g_idx in range(len(caixas_anotadas)):
            var gt = caixas_anotadas[g_idx].copy()
            var gt_cx = Float32((gt[0] + gt[2]) / 2.0)
            var gt_cy = Float32((gt[1] + gt[3]) / 2.0)
            var gt_w = Float32(gt[2] - gt[0])
            var gt_h = Float32(gt[3] - gt[1])
            var a = ancoras[a_idx].copy()
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
            # copia explícita da caixa GT para evitar cópia implícita de lista de propriedade do chamador
            var gt = caixas_anotadas[best_gt_idx].copy()
            var gt_cx = Float32((gt[0] + gt[2]) / 2.0)
            var gt_cy = Float32((gt[1] + gt[3]) / 2.0)
            var gt_w = Float32(gt[2] - gt[0])
            var gt_h = Float32(gt[3] - gt[1])
            # alvos como deltas: tx = (gx - ax)/aw, ty = (gy - ay)/ah, tw = log(gw/aw), th = log(gh/ah)
            var a_local = ancoras[a_idx].copy()
            var ax = a_local[0]; var ay = a_local[1]; var aw = a_local[2]; var ah = a_local[3]
            #ignora ancoras com tamanho próximo de zero para evitar alvos extremos / divisões por zero
            try:
                if aw < 1e-3 or ah < 1e-3:
                    # deixa o rótulo como background (0) e
                    continue
            except _:
                pass
            var pre_tx = (gt_cx - ax) / aw
            var pre_ty = (gt_cy - ay) / ah
            var tx = pre_tx
            var ty = pre_ty
            # computa os deltas de escala e limita os alvos para uma faixa segura para evitar atualizações extremas que podem desestabilizar o treinamento, especialmente no início quando as previsões podem ser muito ruins. Esses limites são baseados na experiência prática e podem ser ajustados conforme necessário. O cálculo dos deltas de escala é feito usando a função logarítmica para transformar a razão entre o tamanho do GT box e o tamanho da ancora em um valor que pode ser positivo ou negativo, dependendo se o GT box é maior ou menor que a ancora. Esses deltas de escala ajudam o modelo a aprender a ajustar as ancoras para melhor se encaixar nos GT boxes durante o treinamento.
            var pre_tw = Float32(math.log(gt_w / aw + 1e-6))
            var pre_th = Float32(math.log(gt_h / ah + 1e-6))
            var tw = pre_tw
            var th = pre_th
            # limita os alvos de regressão para evitar que âncoras com baixa sobreposição gerem alvos extremos que podem desestabilizar o treinamento, especialmente no início quando as previsões podem ser muito ruins. Esses limites são baseados na experiência prática e podem ser ajustados conforme necessário.
            if tx > 3.0: tx = 3.0
            if tx < -3.0: tx = -3.0
            if ty > 3.0: ty = 3.0
            if ty < -3.0: ty = -3.0
            if tw > 4.0: tw = 4.0
            if tw < -4.0: tw = -4.0
            if th > 4.0: th = 4.0
            if th < -4.0: th = -4.0
            # impressões seletivas de depuração para alvos pré-limite extremos ou âncoras minúsculas
            try:
                if (abs(pre_tx) > 3.0) or (abs(pre_ty) > 3.0) or (abs(pre_tw) > 2.0) or (abs(pre_th) > 2.0) or (aw < 2.0) or (ah < 2.0):
                    try:
                        print("[DBG-ASSOCIACAO] anchor_idx", a_idx, "aw", aw, "ah", ah, "pre_tx", pre_tx, "pre_ty", pre_ty, "pre_tw", pre_tw, "pre_th", pre_th, "post_tx", tx, "post_ty", ty, "post_tw", tw, "post_th", th)
                    except _:
                        pass
            except _:
                pass
            var t: List[Float32] = List[Float32]()
            t.append(tx); t.append(ty); t.append(tw); t.append(th)
            # apply regression weights if provided (e.g., [wx,wy,ww,wh])
            try:
                if len(reg_weights) >= 4:
                    for jj in range(4):
                        t[jj] = t[jj] * reg_weights[jj]
            except _:
                pass
            targets[a_idx] = t^
        elif best_iou < iou_neg:
            labels[a_idx] = 0
        else:
            labels[a_idx] = -1 # ignora

    # retorna um resultado struct transferindo a propriedade das listas
    # Garante que cada GT tenha pelo menos uma ancora positiva: força a associação da melhor ancora por GT
    try:
        for g_idx in range(len(caixas_anotadas)):
            var best_a = -1
            var best_a_iou: Float32 = 0.0
            var gt = caixas_anotadas[g_idx].copy()
            var gt_cx = Float32((gt[0] + gt[2]) / 2.0)
            var gt_cy = Float32((gt[1] + gt[3]) / 2.0)
            var gt_w = Float32(gt[2] - gt[0])
            var gt_h = Float32(gt[3] - gt[1])
            for a_idx2 in range(N):
                try:
                    var a2 = ancoras[a_idx2].copy()
                    var iou2 = calcular_iou_xywh(a2, List[Float32](gt_cx, gt_cy, gt_w, gt_h))
                    if iou2 > best_a_iou:
                        best_a_iou = iou2
                        best_a = a_idx2
                except _:
                    pass
            if best_a >= 0:
                # Computa os alvos para esta ancora (mesmo que acima)
                try:
                    var a_local = ancoras[best_a].copy()
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
                        try:
                            if len(reg_weights) >= 4:
                                for jj in range(4):
                                    t[jj] = t[jj] * reg_weights[jj]
                        except _:
                            pass
                        labels[best_a] = 1
                        targets[best_a] = t^
                except _:
                    pass
    except _:
        pass

    return ResultadoAssociado(labels^, targets^)
