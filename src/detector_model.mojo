import bionix_ml.camadas.cnn as cnn_pkg
import bionix_ml.computacao.dispatcher_tensor as dispatcher
import bionix_ml.nucleo.Tensor as tensor_defs
import bionix_ml.dados as dados_pkg
import bionix_ml.graficos as graficos_pkg
import math
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import bionix_ml.camadas.cnn.cnn as cnn_impl
import bionix_ml.autograd.tipos_mlp as tipos_mlp
import bionix_ml.perdas.bce as bce_perdas
import bionix_ml.computacao.sessao as sessao_driver
import bionix_ml.computacao.storage_sessao as storage_sessao
import os
import adaptadores.detectar_face as detect_pkg
import bionix_ml.dados.bmp as bmpmod
import bionix_ml.uteis as uteis
import io_modelo

import bionix_ml.uteis.arquivo as arquivo_io
import os

fn _clamp01(x: Float32) -> Float32:
    var y = x
    if y < 0.0:
        y = 0.0
    if y > 1.0:
        y = 1.0
    return y
fn calcular_iou_bbox(pred: List[Int], alvo: List[Int]) -> Float32:
    # pred e alvo: [x0, y0, x1, y1] em pixels
    if len(pred) < 4 or len(alvo) < 4:
        return 0.0
    var xA = max(pred[0], alvo[0])
    var yA = max(pred[1], alvo[1])
    var xB = min(pred[2], alvo[2])
    var yB = min(pred[3], alvo[3])
    var interW = xB - xA + 1
    var interH = yB - yA + 1
    if interW <= 0 or interH <= 0:
        return 0.0
    var interArea = Float32(interW * interH)
    var boxAArea = Float32((pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1))
    var boxBArea = Float32((alvo[2] - alvo[0] + 1) * (alvo[3] - alvo[1] + 1))
    return interArea / (boxAArea + boxBArea - interArea)

fn treinar_detector_bbox_com_saida(
    mut bloco: cnn_pkg.BlocoCNN,
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    var taxa_aprendizado: Float32 = 0.01,
    var epocas: Int = 100,
    var imprimir_cada: Int = 10,
    var model_dir: String = "",
    var dataset_root: String = "",
    var tolerancia: Float32 = 0.1,
    var early_stop: Bool = False,
) raises -> (Float32, List[Int], String):
    # Igual a treinar_detector_bbox, mas retorna também a última predição e imagem de validação
    debug_assert(len(alvos.formato) == 2 and alvos.formato[1] == 4, "alvos deve ser [N,4]")
    debug_assert(entradas.formato[0] == alvos.formato[0], "N de entradas e alvos deve bater")

    var loss_final: Float32 = 0.0
    var ultima_pred: List[Int] = List[Int]()
    var ultima_img: String = ""

    var head_initialized = False

    for epoca in range(epocas):
        var feats = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)
        try:
            feats = cnn_pkg.extrair_features(bloco, entradas)
        except _:
            print("Erro ao extrair features; pulando epoca", epoca)
            continue

        if not head_initialized:
            var feat_dim = feats.formato[1]
            var shape_w = List[Int]()
            shape_w.append(feat_dim); shape_w.append(4)
            bloco.peso_saida = tensor_defs.Tensor(shape_w^, bloco.tipo_computacao)
            var shape_b = List[Int]()
            shape_b.append(1); shape_b.append(4)
            bloco.bias_saida = tensor_defs.Tensor(shape_b^, bloco.tipo_computacao)
            for i in range(len(bloco.peso_saida.dados)):
                bloco.peso_saida.dados[i] = 0.001
            for j in range(len(bloco.bias_saida.dados)):
                bloco.bias_saida.dados[j] = 0.0
            head_initialized = True

        var preds = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)
        try:
            preds = dispatcher.multiplicar_matrizes(feats, bloco.peso_saida)
            preds = dispatcher.adicionar_bias_coluna(preds, bloco.bias_saida)
        except _:
            print("Erro ao calcular preds; pulando epoca", epoca)
            continue

        var loss = dispatcher.erro_quadratico_medio_escalar(preds, alvos)
        var grad_pred = dispatcher.gradiente_mse(preds, alvos)

        var ft = dispatcher.transpor(feats)
        var grad_w = dispatcher.multiplicar_matrizes(ft, grad_pred)

        var cols = grad_pred.formato[1]
        var rows = grad_pred.formato[0]
        var grad_b_vals = List[Float32]()
        for j in range(cols):
            grad_b_vals.append(0.0)
        for i in range(rows):
            for j in range(cols):
                grad_b_vals[j] = grad_b_vals[j] + grad_pred.dados[i * cols + j]

        for i in range(len(bloco.peso_saida.dados)):
            bloco.peso_saida.dados[i] = bloco.peso_saida.dados[i] - taxa_aprendizado * grad_w.dados[i]
        for j in range(cols):
            bloco.bias_saida.dados[j] = bloco.bias_saida.dados[j] - taxa_aprendizado * grad_b_vals[j]

        loss_final = loss

        if imprimir_cada > 0 and (epoca % imprimir_cada == 0 or epoca == epocas - 1):
            print("Epoca", epoca, "| L1-like MSE loss:", loss_final)
            # Calcular IoU e acurácia de bounding box para uma amostra de validação
            try:
                if len(dataset_root) > 0:
                    var found = False
                    var candidate_path = ""
                    var orig_info = bmpmod.zero_bmp()
                    var gt_box = List[Int]()
                    try:
                        for ident in os.listdir(dataset_root):
                            var p_ident = os.path.join(dataset_root, ident)
                            if not os.path.isdir(p_ident):
                                continue
                            for f in os.listdir(p_ident):
                                if not f.endswith(".bmp"):
                                    continue
                                var candidate = os.path.join(p_ident, f)
                                var txt_path = candidate.replace('.bmp', '.txt')
                                var gt = List[Int]()
                                try:
                                    var linhas = dados_pkg.carregar_txt_linhas(txt_path)
                                    if len(linhas) > 0:
                                        var parts = linhas[0].replace("\t", " ").replace(",", " ").split(" ")
                                        var campos = List[String]()
                                        for p in parts:
                                            var ps = p.strip()
                                            if len(ps) != 0:
                                                campos.append(String(ps))
                                        if len(campos) >= 4:
                                            for i in range(4):
                                                try:
                                                    gt.append(Int(Float32(campos[i])))
                                                except _:
                                                    gt.append(0)
                                except _:
                                    gt = List[Int]()
                                if len(gt) >= 4:
                                    gt_box = gt.copy()
                                    orig_info = dados_pkg.carregar_bmp_rgb(candidate)^ 
                                    candidate_path = candidate
                                    found = True
                                    break
                            if found:
                                break
                    except _:
                        found = False
                    if found and orig_info.width > 0 and orig_info.height > 0 and len(gt_box) >= 4 and len(ultima_pred) >= 4:
                        var iou = calcular_iou_bbox(ultima_pred, gt_box)
                        print("[MÉTRICA] IoU da predição na validação:", iou)
                        # Acurácia: considera acerto se IoU > 0.5
                        var acuracia = 1.0 if iou > 0.5 else 0.0
                        print("[MÉTRICA] Acurácia bbox (IoU>0.5):", acuracia)
            except _:
                pass

        # Exportar pesos em formato binário ao final de cada época

        var export_dir = "detector_modelo"
        if not os.path.exists(export_dir):
            os.mkdir(export_dir)
        var export_path = os.path.join(export_dir, "detector_pesos.bin")
        try:
            var ok = False
            ok = dados_pkg.gravar_arquivo_binario(export_path, bloco.peso_saida.dados_bytes_bin())
            if not ok:
                print("[ERRO] Falha ao exportar pesos binários na época", epoca, "em", export_path)
        except _:
            print("[ERRO] Falha ao exportar pesos binários na época", epoca, "em", export_path)

        # Após cada época, tenta obter uma predição e imagem de validação
        try:
            if len(dataset_root) > 0:
                var found = False
                var candidate_path = ""
                var orig_info = bmpmod.zero_bmp()
                try:
                    for ident in os.listdir(dataset_root):
                        var p_ident = os.path.join(dataset_root, ident)
                        if not os.path.isdir(p_ident):
                            continue
                        for f in os.listdir(p_ident):
                            if not f.endswith(".bmp"):
                                continue
                            var candidate = os.path.join(p_ident, f)
                            try:
                                var res = detect_pkg.detect_and_align_bbox(candidate)
                                var info = res[0].copy()
                                if info.width > 0 and info.height > 0:
                                    orig_info = info^
                                    candidate_path = candidate
                                    found = True
                                    break
                            except _:
                                continue
                        if found:
                            break
                except _:
                    found = False

                if found and orig_info.width > 0 and orig_info.height > 0:
                    var W = orig_info.width
                    var H = orig_info.height
                    # resize grayscale orig_info para bloco.altura x bloco.largura
                    var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(orig_info.grayscale.copy()^, bloco.altura, bloco.largura)
                    var flat = List[Float32]()
                    for ry in range(len(resized)):
                        for rx in range(len(resized[0])):
                            flat.append(resized[ry][rx])
                    var in_shape = List[Int]()
                    in_shape.append(1); in_shape.append(bloco.altura * bloco.largura)
                    var input_tensor = tensor_defs.Tensor(in_shape^, bloco.tipo_computacao)
                    for i in range(len(flat)):
                        input_tensor.dados[i] = flat[i]

                    var feats_single = cnn_pkg.extrair_features(bloco, input_tensor)
                    var preds_single = dispatcher.multiplicar_matrizes(feats_single, bloco.peso_saida)
                    preds_single = dispatcher.adicionar_bias_coluna(preds_single, bloco.bias_saida)
                    if len(preds_single.dados) >= 4:
                        var cx0 = _clamp01(preds_single.dados[0])
                        var cy0 = _clamp01(preds_single.dados[1])
                        var cx1 = _clamp01(preds_single.dados[2])
                        var cy1 = _clamp01(preds_single.dados[3])
                        var px0 = Int(cx0 * Float32(W))
                        var py0 = Int(cy0 * Float32(H))
                        var px1 = Int(cx1 * Float32(W))
                        var py1 = Int(cy1 * Float32(H))
                        if px0 < 0: px0 = 0
                        if py0 < 0: py0 = 0
                        if px1 >= W: px1 = W - 1
                        if py1 >= H: py1 = H - 1
                        ultima_pred = List[Int]()
                        ultima_pred.append(px0); ultima_pred.append(py0); ultima_pred.append(px1); ultima_pred.append(py1)
                        ultima_img = candidate_path
        except _:
            pass

    # save head se solicitado
        try:
            if len(model_dir) > 0:
                var meta = "feat_shape=" + String(bloco.peso_saida.formato[0]) + "," + String(bloco.peso_saida.formato[1]) + "\n"
                meta = meta + "weights="
                for i in range(len(bloco.peso_saida.dados)):
                    if i != 0:
                        meta = meta + ","
                    meta = meta + String(Float64(bloco.peso_saida.dados[i]))
                meta = meta + "\n"
                meta = meta + "bias="
                for j in range(len(bloco.bias_saida.dados)):
                    if j != 0:
                        meta = meta + ","
                    meta = meta + String(Float64(bloco.bias_saida.dados[j]))
                _ = io_modelo.save_metadata(os.path.join(model_dir, "bbox_head.txt"), meta)
        except _:
            pass

    return (loss_final, ultima_pred, ultima_img)


# Wrapper utilities for detector model training using BCEWithLogits

fn criar_bloco_detector(
    var altura: Int,
    var largura: Int,
    var num_filtros: Int,
    var kernel_h: Int,
    var kernel_w: Int,
    contexto: contexto_defs.AdaptadorDeContexto = contexto_defs.AdaptadorDeContexto(),
    var ativacao_id: Int = tipos_mlp.ativacao_saida_padrao_id(1),
    var perda_id: Int = tipos_mlp.perda_padrao_id(1),
    var tipo: String = "cpu",
) -> cnn_pkg.BlocoCNN:
    return cnn_pkg.BlocoCNN(altura, largura, num_filtros, kernel_h, kernel_w, contexto.copy(), ativacao_id, perda_id, tipo)


fn _sigmoid_scalar(var x: Float32) -> Float32:
    # stable sigmoid
    if x >= 0.0:
        var z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        var z = math.exp(x)
        return z / (1.0 + z)


fn treinar_detector_bce(
    mut bloco: cnn_pkg.BlocoCNN,
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    var taxa_aprendizado: Float32 = 0.01,
    var epocas: Int = 100,
    var imprimir_cada: Int = 10,
    var dataset_root: String = "",
 ) raises -> Float32:
    # entradas: [N, H*W], alvos: [N,1]
    debug_assert(len(alvos.formato) == 2 and alvos.formato[1] == 1, "alvos deve ser [N,1]")
    debug_assert(entradas.formato[0] == alvos.formato[0], "N de entradas e alvos deve bater")

    var loss_final: Float32 = 0.0
    for epoca in range(epocas):
        var feats = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)
        try:
            feats = cnn_pkg.extrair_features(bloco, entradas)
        except _:
            print("Erro ao extrair features; pulando epoca", epoca)
            continue
        var z = dispatcher.multiplicar_matrizes(feats, bloco.peso_saida)
        z = dispatcher.adicionar_bias_coluna(z, bloco.bias_saida)

        # use framework BCEWithLogits implementation (host fallback; can be optimized later)
        var loss = bce_perdas.bce_with_logits(z, alvos)
        var grad_z = bce_perdas.grad_bce_with_logits(z, alvos)

        loss_final = loss
        var ft = dispatcher.transpor(feats)
        var grad_w = dispatcher.multiplicar_matrizes(ft, grad_z)
        var grad_b = dispatcher.soma_total(grad_z)

        for i in range(len(bloco.peso_saida.dados)):
            bloco.peso_saida.dados[i] = bloco.peso_saida.dados[i] - taxa_aprendizado * grad_w.dados[i]
        bloco.bias_saida.dados[0] = bloco.bias_saida.dados[0] - taxa_aprendizado * grad_b

        if imprimir_cada > 0 and (epoca % imprimir_cada == 0 or epoca == epocas - 1):
            print("Epoca", epoca, "| BCE loss:", loss_final)

        # Exportar pesos em formato binário ao final de cada época
        var export_dir = "detector_modelo"
        if not os.path.exists(export_dir):
            os.mkdir(export_dir)
        var export_path = os.path.join(export_dir, "detector_pesos.bin")
        # Exportar pesos em formato binário usando dados_bytes(), padrão do framework
        var ok = dados_pkg.gravar_arquivo_binario(export_path, bloco.peso_saida.dados_bytes_bin())
        if not ok:
            print("[ERRO] Falha ao exportar pesos binários na época", epoca, "em", export_path)
            # Visual validation: save a color BMP with bbox overlay for the first sample
            try:
                var out_dir = os.path.join("..", "validacao")
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)
                var feat_dim = entradas.formato[1]
                var h = bloco.altura
                var w = bloco.largura
                if feat_dim == h * w and len(entradas.dados) >= feat_dim:
                    # Find a representative original image and draw detected bbox on it
                    var found = False
                    var orig_info = bmpmod.zero_bmp()
                    var bbox = List[Int]()
                    var candidate_path = ""
                    if len(dataset_root) > 0:
                        try:
                            for ident in os.listdir(dataset_root):
                                var p_ident = os.path.join(dataset_root, ident)
                                if not os.path.isdir(p_ident):
                                    continue
                                for f in os.listdir(p_ident):
                                    if not f.endswith(".bmp"):
                                        continue
                                    var candidate = os.path.join(p_ident, f)
                                    try:
                                        var res = detect_pkg.detect_and_align_bbox(candidate)
                                        var info = res[0].copy()
                                        var bb = res[1].copy()
                                        if info.width > 0 and info.height > 0:
                                            orig_info = info^
                                            bbox = bb^
                                            candidate_path = candidate
                                            found = True
                                            break
                                    except _:
                                        continue
                                if found:
                                    break
                        except _:
                            found = False

                    if found and orig_info.width > 0 and orig_info.height > 0 and len(bbox) >= 4:
                        # build RGB flat list from BMPInfo pixels
                        var H = orig_info.height
                        var W = orig_info.width
                        var img_rgb = List[Int](capacity=W * H * 3)
                        if len(orig_info.pixels) > 0:
                            for ry in range(H):
                                for rx in range(W):
                                    var px = orig_info.pixels[ry][rx]
                                    var r_f = px[0] * 255.0
                                    var g_f = px[1] * 255.0
                                    var b_f = px[2] * 255.0
                                    var r = Int(r_f)
                                    var g = Int(g_f)
                                    var b = Int(b_f)
                                    if r < 0: r = 0
                                    if g < 0: g = 0
                                    if b < 0: b = 0
                                    if r > 255: r = 255
                                    if g > 255: g = 255
                                    if b > 255: b = 255
                                    img_rgb.append(r); img_rgb.append(g); img_rgb.append(b)
                        else:
                            # fallback to a grayscale visualization from entrada sample
                            var img_gray = List[Int](capacity=w * h)
                            for i in range(w * h):
                                var v: Float32 = 0.0
                                try:
                                    v = entradas.dados[i]
                                except _:
                                    v = 0.0
                                var iv = Int(v * 255.0)
                                if iv < 0:
                                    iv = 0
                                if iv > 255:
                                    iv = 255
                                img_gray.append(iv)
                            for p in img_gray:
                                img_rgb.append(p); img_rgb.append(p); img_rgb.append(p)
                            W = w; H = h

                        # parse .txt box if present (baseline) and draw it in green;
                        # adapter bbox will be drawn in red
                        var file_box = List[Int]()
                        try:
                            # accept either .txt or legacy .box files
                            var txt_path = candidate_path.replace('.bmp', '.txt')
                            if not os.path.exists(txt_path):
                                var txt_path_box = candidate_path.replace('.bmp', '.box')
                                if os.path.exists(txt_path_box):
                                    txt_path = txt_path_box
                            if os.path.exists(txt_path):
                                var linhas = dados_pkg.carregar_txt_linhas(txt_path)
                                if len(linhas) > 0:
                                    var parts = linhas[0].replace("\t", " ").replace(",", " ").split(" ")
                                    var campos = List[String]()
                                    for p in parts:
                                        var ps = p.strip()
                                        if len(ps) != 0:
                                            campos.append(String(ps))
                                    if len(campos) >= 4:
                                        for i in range(4):
                                            try:
                                                var v = Float32(uteis.parse_float_ascii(String(campos[i])))
                                                file_box.append(Int(v))
                                            except _:
                                                file_box.append(0)
                        except _:
                            file_box = List[Int]()

                        var thickness = 2
                        var x0 = bbox[0]
                        var y0 = bbox[1]
                        var x1 = bbox[2]
                        var y1 = bbox[3]
                        if x0 < 0: x0 = 0
                        if y0 < 0: y0 = 0
                        if x1 >= W: x1 = W - 1
                        if y1 >= H: y1 = H - 1
                        for t in range(thickness):
                            for x in range(x0 + t, x1 - t + 1):
                                var ti_top = ((y0 + t) * W + x) * 3
                                img_rgb[ti_top + 0] = 255
                                img_rgb[ti_top + 1] = 0
                                img_rgb[ti_top + 2] = 0
                                var ti_bot = ((y1 - t) * W + x) * 3
                                img_rgb[ti_bot + 0] = 255
                                img_rgb[ti_bot + 1] = 0
                                img_rgb[ti_bot + 2] = 0
                            for y in range(y0 + t, y1 - t + 1):
                                var ti_l = (y * W + (x0 + t)) * 3
                                img_rgb[ti_l + 0] = 255
                                img_rgb[ti_l + 1] = 0
                                img_rgb[ti_l + 2] = 0
                                var ti_r = (y * W + (x1 - t)) * 3
                                img_rgb[ti_r + 0] = 255
                                img_rgb[ti_r + 1] = 0
                                img_rgb[ti_r + 2] = 0

                        # draw file_box if available (green)
                        if len(file_box) >= 4:
                            var fx0 = file_box[0]
                            var fy0 = file_box[1]
                            var fx1 = file_box[2]
                            var fy1 = file_box[3]
                            if fx0 < 0: fx0 = 0
                            if fy0 < 0: fy0 = 0
                            if fx1 >= W: fx1 = W - 1
                            if fy1 >= H: fy1 = H - 1
                            for t in range(thickness):
                                for x in range(fx0 + t, fx1 - t + 1):
                                    var ti_topf = ((fy0 + t) * W + x) * 3
                                    img_rgb[ti_topf + 0] = 0
                                    img_rgb[ti_topf + 1] = 255
                                    img_rgb[ti_topf + 2] = 0
                                    var ti_botf = ((fy1 - t) * W + x) * 3
                                    img_rgb[ti_botf + 0] = 0
                                    img_rgb[ti_botf + 1] = 255
                                    img_rgb[ti_botf + 2] = 0
                                for y in range(fy0 + t, fy1 - t + 1):
                                    var ti_lf = (y * W + (fx0 + t)) * 3
                                    img_rgb[ti_lf + 0] = 0
                                    img_rgb[ti_lf + 1] = 255
                                    img_rgb[ti_lf + 2] = 0
                                    var ti_rf = (y * W + (fx1 - t)) * 3
                                    img_rgb[ti_rf + 0] = 0
                                    img_rgb[ti_rf + 1] = 255
                                    img_rgb[ti_rf + 2] = 0

                        var bmp_bytes = graficos_pkg.bmp.gerar_bmp_24bits_de_rgb(img_rgb, W, H)
                        _ = dados_pkg.gravar_arquivo_binario(os.path.join(out_dir, "validacao.bmp"), bmp_bytes^)
                    else:
                        # fallback: write the input-sample crop visualization
                        var img_gray = List[Int](capacity=w * h)
                        for i in range(w * h):
                            var v: Float32 = 0.0
                            try:
                                v = entradas.dados[i]
                            except _:
                                v = 0.0
                            var iv = Int(v * 255.0)
                            if iv < 0:
                                iv = 0
                            if iv > 255:
                                iv = 255
                            img_gray.append(iv)
                        var img_rgb = List[Int](capacity=w * h * 3)
                        for p in img_gray:
                            img_rgb.append(p); img_rgb.append(p); img_rgb.append(p)
                        var bmp_bytes = graficos_pkg.bmp.gerar_bmp_24bits_de_rgb(img_rgb, w, h)
                        _ = dados_pkg.gravar_arquivo_binario(os.path.join(out_dir, "validacao.bmp"), bmp_bytes^)
            except _:
                pass

    return loss_final


fn treinar_detector_bbox(
    mut bloco: cnn_pkg.BlocoCNN,
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    var taxa_aprendizado: Float32 = 0.01,
    var epocas: Int = 100,
    var imprimir_cada: Int = 10,
    var model_dir: String = "",
    var dataset_root: String = "",
    var tolerancia: Float32 = 0.1,
    var early_stop: Bool = False,
) raises -> Float32:
    # Expect alvos shape [N,4]
    debug_assert(len(alvos.formato) == 2 and alvos.formato[1] == 4, "alvos deve ser [N,4]")
    debug_assert(entradas.formato[0] == alvos.formato[0], "N de entradas e alvos deve bater")

    var loss_final: Float32 = 0.0

    # Build a small linear head for bbox: weights [feat_dim,4], bias [1,4]
    var head_w: tensor_defs.Tensor = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)
    var head_b: tensor_defs.Tensor = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)
    var head_initialized = False

    for epoca in range(epocas):
        var feats = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)
        try:
            feats = cnn_pkg.extrair_features(bloco, entradas)
        except _:
            print("Erro ao extrair features; pulando epoca", epoca)
            continue

        if not head_initialized:
            var feat_dim = feats.formato[1]
            var shape_w = List[Int]()
            shape_w.append(feat_dim); shape_w.append(4)
            head_w = tensor_defs.Tensor(shape_w^, bloco.tipo_computacao)
            var shape_b = List[Int]()
            shape_b.append(1); shape_b.append(4)
            head_b = tensor_defs.Tensor(shape_b^, bloco.tipo_computacao)
            # small init
            for i in range(len(head_w.dados)):
                head_w.dados[i] = 0.001
            for j in range(len(head_b.dados)):
                head_b.dados[j] = 0.0
            head_initialized = True

        var preds = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)
        try:
            preds = dispatcher.multiplicar_matrizes(feats, head_w)
            preds = dispatcher.adicionar_bias_coluna(preds, head_b)
        except _:
            print("Erro ao calcular preds; pulando epoca", epoca)
            continue

        var loss = dispatcher.erro_quadratico_medio_escalar(preds, alvos)
        var grad_pred = dispatcher.gradiente_mse(preds, alvos)

        # weight gradient
        var ft = dispatcher.transpor(feats)
        var grad_w = dispatcher.multiplicar_matrizes(ft, grad_pred)

        # bias gradient: sum grad_pred across batch for each column
        var cols = grad_pred.formato[1]
        var rows = grad_pred.formato[0]
        var grad_b_vals = List[Float32]()
        for j in range(cols):
            grad_b_vals.append(0.0)
        for i in range(rows):
            for j in range(cols):
                grad_b_vals[j] = grad_b_vals[j] + grad_pred.dados[i * cols + j]

        # apply gradients
        for i in range(len(head_w.dados)):
            head_w.dados[i] = head_w.dados[i] - taxa_aprendizado * grad_w.dados[i]
        # bias dims: head_b.dados length == cols
        for j in range(cols):
            head_b.dados[j] = head_b.dados[j] - taxa_aprendizado * grad_b_vals[j]

        loss_final = loss

        if imprimir_cada > 0 and (epoca % imprimir_cada == 0 or epoca == epocas - 1):
            print("Epoca", epoca, "| L1-like MSE loss:", loss_final)
            # Debug: print first sample target (normalized) and raw preds before clamp
            try:
                if entradas.formato[0] > 0 and alvos.formato[0] > 0 and len(preds.dados) >= 4:
                    var sample_target = List[Float32]()
                    for k in range(4):
                        sample_target.append(alvos.dados[0 * 4 + k])
                    var sample_pred = List[Float32]()
                    for k in range(4):
                        sample_pred.append(preds.dados[k])
                    print("[DEBUG] alvo_normalizado:", sample_target, "preds_raw:", sample_pred)
            except _:
                pass

        # Exportar pesos em formato binário ao final de cada época

        var export_dir = "detector_modelo"
        if not os.path.exists(export_dir):
            os.mkdir(export_dir)
        var export_path = os.path.join(export_dir, "detector_pesos.bin")
        # Exportar pesos em formato binário usando dados_bytes(), padrão do framework
        var ok = dados_pkg.gravar_arquivo_binario(export_path, head_w.dados_bytes_bin())
        if not ok:
            print("[ERRO] Falha ao exportar pesos binários na época", epoca, "em", export_path)

            # Visual validation: save two images for comparison
            try:
                if len(dataset_root) > 0:
                    var out_dir = os.path.join("..", "validacao")
                    if not os.path.isdir(out_dir):
                        os.makedirs(out_dir)

                    # find a candidate image with .bmp and .txt
                    var found = False
                    var candidate_path = ""
                    var orig_info = bmpmod.zero_bmp()
                    var file_box = List[Int]()
                    try:
                        for ident in os.listdir(dataset_root):
                            var p_ident = os.path.join(dataset_root, ident)
                            if not os.path.isdir(p_ident):
                                continue
                            for f in os.listdir(p_ident):
                                if not f.endswith(".bmp"):
                                    continue
                                var candidate = os.path.join(p_ident, f)
                                try:
                                    var res = detect_pkg.detect_and_align_bbox(candidate)
                                    var info = res[0].copy()
                                    if info.width > 0 and info.height > 0:
                                        orig_info = info^
                                        candidate_path = candidate
                                        found = True
                                        break
                                except _:
                                    continue
                            if found:
                                break
                    except _:
                        found = False

                    if found and orig_info.width > 0 and orig_info.height > 0:
                        # parse .txt box if present (baseline)
                        try:
                            # accept either .txt or legacy .box files
                            var txt_path = candidate_path.replace('.bmp', '.txt')
                            if not os.path.exists(txt_path):
                                var txt_path_box = candidate_path.replace('.bmp', '.box')
                                if os.path.exists(txt_path_box):
                                    txt_path = txt_path_box
                            if os.path.exists(txt_path):
                                var linhas = dados_pkg.carregar_txt_linhas(txt_path)
                                if len(linhas) > 0:
                                    var parts = linhas[0].replace("\t", " ").replace(",", " ").split(" ")
                                    var campos = List[String]()
                                    for p in parts:
                                        var ps = p.strip()
                                        if len(ps) != 0:
                                            campos.append(String(ps))
                                    if len(campos) >= 4:
                                        for i in range(4):
                                            try:
                                                var v = Float32(uteis.parse_float_ascii(String(campos[i])))
                                                file_box.append(Int(v))
                                            except _:
                                                file_box.append(0)
                        except _:
                            file_box = List[Int]()

                        # Build RGB flat list and also grayscale input for model prediction
                        var W = orig_info.width
                        var H = orig_info.height
                        var img_rgb = List[Int](capacity=W * H * 3)
                        var img_gray = List[List[Float32]]()
                        if len(orig_info.pixels) > 0:
                            for ry in range(H):
                                var row = List[Float32]()
                                for rx in range(W):
                                    var px = orig_info.pixels[ry][rx]
                                    var r_f = px[0] * 255.0
                                    var g_f = px[1] * 255.0
                                    var b_f = px[2] * 255.0
                                    var r = Int(r_f)
                                    var g = Int(g_f)
                                    var b = Int(b_f)
                                    if r < 0: r = 0
                                    if g < 0: g = 0
                                    if b < 0: b = 0
                                    if r > 255: r = 255
                                    if g > 255: g = 255
                                    if b > 255: b = 255
                                    img_rgb.append(r); img_rgb.append(g); img_rgb.append(b)
                                    # grayscale normalized 0..1
                                    var gray = (px[0] + px[1] + px[2]) / 3.0
                                    row.append(gray)
                                img_gray.append(row)

                        # Create baseline image (file_box) and detected image (adapter bbox) separately
                        # baseline image: draw file_box in green
                        if len(img_rgb) == 0:
                            # fallback: nothing to draw
                            pass
                        else:
                            # Copy rgb list for two images
                            var rgb_box = img_rgb.copy()
                            var rgb_detect = img_rgb.copy()

                            # draw file_box if available (green) on rgb_box
                            if len(file_box) >= 4:
                                var fx0 = file_box[0]
                                var fy0 = file_box[1]
                                var fx1 = file_box[2]
                                var fy1 = file_box[3]
                                if fx0 < 0: fx0 = 0
                                if fy0 < 0: fy0 = 0
                                if fx1 >= W: fx1 = W - 1
                                if fy1 >= H: fy1 = H - 1
                                var thickness = 2
                                for t in range(thickness):
                                    for x in range(fx0 + t, fx1 - t + 1):
                                        var ti_topf = ((fy0 + t) * W + x) * 3
                                        rgb_box[ti_topf + 0] = 0
                                        rgb_box[ti_topf + 1] = 255
                                        rgb_box[ti_topf + 2] = 0
                                        var ti_botf = ((fy1 - t) * W + x) * 3
                                        rgb_box[ti_botf + 0] = 0
                                        rgb_box[ti_botf + 1] = 255
                                        rgb_box[ti_botf + 2] = 0
                                    for y in range(fy0 + t, fy1 - t + 1):
                                        var ti_lf = (y * W + (fx0 + t)) * 3
                                        rgb_box[ti_lf + 0] = 0
                                        rgb_box[ti_lf + 1] = 255
                                        rgb_box[ti_lf + 2] = 0
                                        var ti_rf = (y * W + (fx1 - t)) * 3
                                        rgb_box[ti_rf + 0] = 0
                                        rgb_box[ti_rf + 1] = 255
                                        rgb_box[ti_rf + 2] = 0

                            # adapter detected bbox: use adapter result (red) on rgb_detect
                            var det_bbox = List[Int]()
                            try:
                                var res2 = detect_pkg.detect_and_align_bbox(candidate_path)
                                det_bbox = res2[1].copy()
                            except _:
                                det_bbox = List[Int]()

                            if len(det_bbox) >= 4:
                                var x0 = det_bbox[0]
                                var y0 = det_bbox[1]
                                var x1 = det_bbox[2]
                                var y1 = det_bbox[3]
                                if x0 < 0: x0 = 0
                                if y0 < 0: y0 = 0
                                if x1 >= W: x1 = W - 1
                                if y1 >= H: y1 = H - 1
                                var thickness = 2
                                for t in range(thickness):
                                    for x in range(x0 + t, x1 - t + 1):
                                        var ti_top = ((y0 + t) * W + x) * 3
                                        rgb_detect[ti_top + 0] = 255
                                        rgb_detect[ti_top + 1] = 0
                                        rgb_detect[ti_top + 2] = 0
                                        var ti_bot = ((y1 - t) * W + x) * 3
                                        rgb_detect[ti_bot + 0] = 255
                                        rgb_detect[ti_bot + 1] = 0
                                        rgb_detect[ti_bot + 2] = 0
                                    for y in range(y0 + t, y1 - t + 1):
                                        var ti_l = (y * W + (x0 + t)) * 3
                                        rgb_detect[ti_l + 0] = 255
                                        rgb_detect[ti_l + 1] = 0
                                        rgb_detect[ti_l + 2] = 0
                                        var ti_r = (y * W + (x1 - t)) * 3
                                        rgb_detect[ti_r + 0] = 255
                                        rgb_detect[ti_r + 1] = 0
                                        rgb_detect[ti_r + 2] = 0

                            # model prediction: run the head on the candidate image converted to input
                            var model_pred_box = List[Int]()
                            try:
                                # resize grayscale orig_info to bloco.altura x bloco.largura
                                var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(orig_info.grayscale.copy()^, bloco.altura, bloco.largura)
                                var flat = List[Float32]()
                                for ry in range(len(resized)):
                                    for rx in range(len(resized[0])):
                                        flat.append(resized[ry][rx])
                                var in_shape = List[Int]()
                                in_shape.append(1); in_shape.append(bloco.altura * bloco.largura)
                                var input_tensor = tensor_defs.Tensor(in_shape^, bloco.tipo_computacao)
                                for i in range(len(flat)):
                                    input_tensor.dados[i] = flat[i]

                                var feats_single = cnn_pkg.extrair_features(bloco, input_tensor)
                                var preds_single = dispatcher.multiplicar_matrizes(feats_single, bloco.peso_saida)
                                preds_single = dispatcher.adicionar_bias_coluna(preds_single, bloco.bias_saida)
                                # read first row and clamp outputs to [0,1]
                                if len(preds_single.dados) >= 4:
                                    var cx0 = _clamp01(preds_single.dados[0])
                                    var cy0 = _clamp01(preds_single.dados[1])
                                    var cx1 = _clamp01(preds_single.dados[2])
                                    var cy1 = _clamp01(preds_single.dados[3])
                                    var px0 = Int(cx0 * Float32(W))
                                    var py0 = Int(cy0 * Float32(H))
                                    var px1 = Int(cx1 * Float32(W))
                                    var py1 = Int(cy1 * Float32(H))
                                    if px0 < 0: px0 = 0
                                    if py0 < 0: py0 = 0
                                    if px1 >= W: px1 = W - 1
                                    if py1 >= H: py1 = H - 1
                                    model_pred_box.append(px0); model_pred_box.append(py0); model_pred_box.append(px1); model_pred_box.append(py1)
                                    # draw model prediction on rgb_detect (blue)
                                    var thickness = 2
                                    for t in range(thickness):
                                        for x in range(px0 + t, px1 - t + 1):
                                            var ti_topm = ((py0 + t) * W + x) * 3
                                            rgb_detect[ti_topm + 0] = 0
                                            rgb_detect[ti_topm + 1] = 0
                                            rgb_detect[ti_topm + 2] = 255
                                            var ti_botm = ((py1 - t) * W + x) * 3
                                            rgb_detect[ti_botm + 0] = 0
                                            rgb_detect[ti_botm + 1] = 0
                                            rgb_detect[ti_botm + 2] = 255
                                        for y in range(py0 + t, py1 - t + 1):
                                            var ti_lm = (y * W + (px0 + t)) * 3
                                            rgb_detect[ti_lm + 0] = 0
                                            rgb_detect[ti_lm + 1] = 0
                                            rgb_detect[ti_lm + 2] = 255
                                            var ti_rm = (y * W + (px1 - t)) * 3
                                            rgb_detect[ti_rm + 0] = 0
                                            rgb_detect[ti_rm + 1] = 0
                                            rgb_detect[ti_rm + 2] = 255
                            except _:
                                pass

                            # write the two images
                            var bmp_box = graficos_pkg.bmp.gerar_bmp_24bits_de_rgb(rgb_box, W, H)
                            var bmp_detect = graficos_pkg.bmp.gerar_bmp_24bits_de_rgb(rgb_detect, W, H)
                            _ = dados_pkg.gravar_arquivo_binario(os.path.join(out_dir, "validacao_box.bmp"), bmp_box^)
                            _ = dados_pkg.gravar_arquivo_binario(os.path.join(out_dir, "validacao_detected.bmp"), bmp_detect^)

                            # tolerance check between file_box center and model_pred_box center
                            if len(file_box) >= 4 and len(model_pred_box) >= 4:
                                var cfx = (Float32(file_box[0]) + Float32(file_box[2])) / 2.0
                                var cfy = (Float32(file_box[1]) + Float32(file_box[3])) / 2.0
                                var cmx = (Float32(model_pred_box[0]) + Float32(model_pred_box[2])) / 2.0
                                var cmy = (Float32(model_pred_box[1]) + Float32(model_pred_box[3])) / 2.0
                                var dx = abs(cfx - cmx) / Float32(W)
                                var dy = abs(cfy - cmy) / Float32(H)
                                if dx <= tolerancia and dy <= tolerancia:
                                    print("Predição dentro da tolerância (dx=", dx, "dy=", dy, ")")
                                    if early_stop:
                                        print("Early stop: predição aceitável; terminando treinamento.")
                                        return loss_final
                    # end found
            except _:
                pass

    # save head if requested
    try:
        if len(model_dir) > 0:
            var meta = "feat_shape=" + String(head_w.formato[0]) + "," + String(head_w.formato[1]) + "\n"
            meta = meta + "weights="
            for i in range(len(head_w.dados)):
                if i != 0:
                    meta = meta + ","
                meta = meta + String(Float64(head_w.dados[i]))
            meta = meta + "\n"
            meta = meta + "bias="
            for j in range(len(head_b.dados)):
                if j != 0:
                    meta = meta + ","
                meta = meta + String(Float64(head_b.dados[j]))
            _ = io_modelo.save_metadata(os.path.join(model_dir, "bbox_head.txt"), meta)
    except _:
        pass

    return loss_final


fn salvar_checkpoint(mut bloco: cnn_pkg.BlocoCNN, var model_dir: String) -> Bool:
    try:
        var driver = sessao_driver.driver_sessao_disco(model_dir)
        var storage = storage_sessao.criar_storage_sessao(driver)
        cnn_impl._salvar_bloco_em_storage(bloco, storage)
        return True
    except _:
        return False


fn carregar_checkpoint(mut bloco: cnn_pkg.BlocoCNN, var model_dir: String) -> Bool:
    try:
        if not os.path.isdir(model_dir):
            return False
        var driver = sessao_driver.driver_sessao_disco(model_dir)
        var storage = storage_sessao.criar_storage_sessao(driver)
        cnn_impl._carregar_bloco_de_storage(bloco, storage)
        return True
    except _:
        return False
