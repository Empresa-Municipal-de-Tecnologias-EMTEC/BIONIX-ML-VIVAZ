import retina.retina_anchor_generator as anchor_gen
import retina.retina_assigner as assigner
import retina.retina_nms as nms_pkg
import bionix_ml.computacao.dispatcher_tensor as dispatcher
import bionix_ml.nucleo.Tensor as tensor_defs
import bionix_ml.dados as dados_pkg
import os
import bionix_ml.dados.bmp as bmpmod
import math

fn inferir_retina(model_dir: String = "MODELO", input_size: Int = 640, max_per_image: Int = 16) -> List[List[Int]]:
    # Pipeline de inferência mínima: carrega pesos, gera anchors, produz predições e aplica NMS.
    var anchors = anchor_gen.gerar_anchors(input_size)
    # tenta carregar pesos binários: regressor e classificador (se existirem)
    var cls_scores: List[Float32] = List[Float32]()
    var reg_deltas: List[List[Float32]] = List[List[Float32]]()
    # fallback: retorna vazio se não houver pesos
    var peso_path = os.path.join(model_dir, "retina_pesos_reg.bin")
    if not os.path.exists(peso_path):
        print("[INFER] Nenhum peso de retina encontrado em", peso_path)
        return List[List[Int]]()

    # carregar pesos como placeholder: aqui assumimos que o modelo produz zeros
    for a in anchors:
        cls_scores.append(0.01)
        var d = List[Float32]()
        d.append(0.0); d.append(0.0); d.append(0.0); d.append(0.0)
        reg_deltas.append(d)

    # decodificar deltas para boxes [x0,y0,x1,y1]
    var boxes: List[List[Float32]] = List[List[Float32]]()
    for i in range(len(anchors)):
        var a = anchors[i]
        var dx = reg_deltas[i][0]; var dy = reg_deltas[i][1]; var dw = reg_deltas[i][2]; var dh = reg_deltas[i][3]
        var cx = a[0] + dx * a[2]
        var cy = a[1] + dy * a[3]
        var w = a[2] * Float32(math.exp(Float64(dw)))
        var h = a[3] * Float32(math.exp(Float64(dh)))
        var x0 = cx - w/2.0
        var y0 = cy - h/2.0
        var x1 = cx + w/2.0
        var y1 = cy + h/2.0
        var outb = List[Float32]()
        outb.append(x0); outb.append(y0); outb.append(x1); outb.append(y1)
        boxes.append(outb)

    var keep = nms_pkg.non_max_suppression(boxes, cls_scores, 0.5)
    var kept_boxes = List[List[Int]]()
    for k in keep:
        if len(kept_boxes) >= max_per_image:
            break
        var b = boxes[k]
        var ib = List[Int]()
        ib.append(Int(b[0])); ib.append(Int(b[1])); ib.append(Int(b[2])); ib.append(Int(b[3]))
        kept_boxes.append(ib)

    return kept_boxes
