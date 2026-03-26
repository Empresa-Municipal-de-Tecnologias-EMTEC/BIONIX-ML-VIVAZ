import config as cfg
import os
import io_modelo
import detector_model as model_pkg
import bionix_ml.dados as dados_pkg
import bionix_ml.dados.arquivo as dados_arquivo
import bionix_ml.graficos as graficos_pkg
import bionix_ml.dados.bmp as bmpmod
import adaptadores.detectar_face as detect_pkg
import bionix_ml.camadas.cnn as cnn_pkg
import bionix_ml.nucleo.Tensor as tensor_defs_local
import bionix_ml.computacao.dispatcher_tensor as dispatcher
import bionix_ml.uteis.arquivo as arquivo_io


fn main() -> None:
    print("Inferência do detector — valida 10 imagens e salva em validacao_inferencia")

    # determine dataset path
    var dataset_path = cfg.DATASET_ROOT + "/treino"
    if not os.path.isdir(dataset_path):
        dataset_path = cfg.DATASET_ROOT + "/train"

    # try to read model metadata to get input size
    var altura: Int = 100
    var largura: Int = 64
    try:
        var meta = io_modelo.load_metadata(os.path.join(cfg.MODEL_DIR, "metadata_detector.txt"))
        if len(meta) > 0:
            for L in meta.split("\n"):
                if L.startswith("input_h:"):
                    try:
                        altura = Int(L.replace("input_h:", "").strip())
                    except _:
                        pass
                elif L.startswith("input_w:"):
                    try:
                        largura = Int(L.replace("input_w:", "").strip())
                    except _:
                        pass
    except _:
        pass

    # create block and load checkpoint if available
    import bionix_ml.computacao.adaptadores.contexto as contexto_defs
    var ctx = contexto_defs.criar_contexto_padrao("cpu")
    var bloco = model_pkg.criar_bloco_detector(altura, largura, 6, 3, 3, ctx)
    var carregado = model_pkg.carregar_checkpoint(bloco, cfg.MODEL_DIR)
    if carregado:
        print("Checkpoint carregado de", cfg.MODEL_DIR)
    else:
        print("Nenhum checkpoint encontrado — a inferência continuará mas pode falhar")

    # load bbox head binary metadata (prefer binary weights saved by framework)
    var w_vals = List[Float32]()
    var b_vals = List[Float32]()
    try:
        var meta = io_modelo.load_metadata(os.path.join(cfg.MODEL_DIR, "metadata_detector.txt"))
        var weights_bin = ""
        var bias_bin = ""
        if len(meta) > 0:
            for L in meta.split("\n"):
                if L.startswith("weights_bin="):
                    weights_bin = String(L.replace("weights_bin=", "").strip())
                elif L.startswith("bias_bin="):
                    bias_bin = String(L.replace("bias_bin=", "").strip())
        if len(weights_bin) > 0:
            try:
                var raw_w = dados_arquivo.ler_arquivo_binario(os.path.join(cfg.MODEL_DIR, weights_bin))
                # try to load into the bloco.peso_saida if present
                if len(raw_w) > 0 and bloco.peso_saida.formato and len(bloco.peso_saida.formato) > 0:
                    bloco.peso_saida.carregar_dados_bytes_bin(raw_w)
            except _:
                pass
        if len(bias_bin) > 0:
            try:
                var raw_b = dados_arquivo.ler_arquivo_binario(os.path.join(cfg.MODEL_DIR, bias_bin))
                if len(raw_b) > 0 and bloco.bias_saida.formato and len(bloco.bias_saida.formato) > 0:
                    bloco.bias_saida.carregar_dados_bytes_bin(raw_b)
            except _:
                pass
        # Fallback: if binary files not present or failed, try legacy bbox_head.txt (text weights)
        try:
            if not bloco.peso_saida.formato or len(bloco.peso_saida.dados) == 0 or bloco.peso_saida.dados[0] == 0.0:
                var bbox_txt = io_modelo.load_metadata(os.path.join(cfg.MODEL_DIR, "bbox_head.txt"))
                if len(bbox_txt) > 0:
                    var w_line = ""
                    var b_line = ""
                    for LL in bbox_txt.split("\n"):
                        if LL.startswith("weights="):
                            w_line = String(LL.replace("weights=", ""))
                        elif LL.startswith("bias="):
                            b_line = String(LL.replace("bias=", ""))
                    if len(w_line) > 0:
                        var toksw = w_line.split(",")
                        for t in toksw:
                            try:
                                w_vals.append(Float32(Float64(String(t))))
                            except _:
                                w_vals.append(0.0)
                    if len(b_line) > 0:
                        var toksb = b_line.split(",")
                        for t in toksb:
                            try:
                                b_vals.append(Float32(Float64(String(t))))
                            except _:
                                b_vals.append(0.0)
                    if bloco.peso_saida.formato and len(w_vals) >= len(bloco.peso_saida.dados):
                        for i in range(len(bloco.peso_saida.dados)):
                            bloco.peso_saida.dados[i] = w_vals[i]
                    if bloco.bias_saida.formato and len(b_vals) >= len(bloco.bias_saida.dados):
                        for j in range(len(bloco.bias_saida.dados)):
                            bloco.bias_saida.dados[j] = b_vals[j]
        except _:
            pass
    except _:
        pass

    # collect first 10 bmp files from dataset
    var candidates = List[String]()
    try:
        for ident in os.listdir(dataset_path):
            var p_ident = os.path.join(dataset_path, ident)
            if not os.path.isdir(p_ident):
                continue
            for f in os.listdir(p_ident):
                if f.lower().endswith('.bmp'):
                    candidates.append(os.path.join(p_ident, f))
                if len(candidates) >= 10:
                    break
            if len(candidates) >= 10:
                break
    except _:
        pass

    if len(candidates) == 0:
        print("Nenhuma imagem .bmp encontrada em", dataset_path)
        return

    # prepare output dir
    var out_dir = "validacao_inferencia"
    try:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
    except _:
        pass

    var idx = 0
    for path in candidates:
        if idx >= 10:
            break
        idx = idx + 1
        var info = bmpmod.zero_bmp()
        try:
            info = dados_pkg.carregar_bmp_rgb(path)^
        except _:
            print("Falha ao carregar imagem", path)
            continue

        # build grayscale resized input same as training
        var img_gray = List[List[Float32]]()
        try:
            img_gray = dados_pkg.carregar_bmp_grayscale_matriz(path)
        except _:
            img_gray = List[List[Float32]]()

        if len(img_gray) == 0:
            print("Falha ao gerar matriz grayscale para", path)
            continue

        var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(img_gray.copy()^, altura, largura)
        var features = altura * largura
        var shape_x = List[Int]()
        shape_x.append(1); shape_x.append(features)
        var x_s = tensor_defs_local.Tensor(shape_x^, bloco.tipo_computacao)
        for i in range(features):
            x_s.dados[i] = resized[i // largura][i % largura]

        # extract features and apply head if present
        var pred_box = List[Float32]()
        fn _clamp01(x: Float32) -> Float32:
            var y = x
            if y < 0.0:
                y = 0.0
            if y > 1.0:
                y = 1.0
            return y
        try:
            var feats = cnn_pkg.extrair_features(bloco, x_s)
            if len(w_vals) > 0 and len(b_vals) > 0:
                var feat_dim = feats.formato[1]
                var shape_hw = List[Int]()
                shape_hw.append(feat_dim); shape_hw.append(4)
                var head_w = tensor_defs_local.Tensor(shape_hw^, bloco.tipo_computacao)
                var shape_hb = List[Int]()
                shape_hb.append(1); shape_hb.append(4)
                var head_b = tensor_defs_local.Tensor(shape_hb^, bloco.tipo_computacao)
                for i in range(min(len(head_w.dados), len(w_vals))):
                    head_w.dados[i] = w_vals[i]
                for j in range(min(len(head_b.dados), len(b_vals))):
                    head_b.dados[j] = b_vals[j]
                var pred = dispatcher.multiplicar_matrizes(feats, head_w)
                pred = dispatcher.adicionar_bias_coluna(pred, head_b)
                # collect normalized prediction (clamped)
                for v in pred.dados:
                    pred_box.append(_clamp01(v))
        except _:
            pass


        # if no pred, try adapter detection as fallback
        var box_pixels = List[Int]()
        try:
            if len(pred_box) >= 4:
                var px0 = Int(pred_box[0] * Float32(info.width - 1))
                var py0 = Int(pred_box[1] * Float32(info.height - 1))
                var px1 = Int(pred_box[2] * Float32(info.width - 1))
                var py1 = Int(pred_box[3] * Float32(info.height - 1))
                if px0 < 0: px0 = 0
                if py0 < 0: py0 = 0
                if px1 >= info.width: px1 = info.width - 1
                if py1 >= info.height: py1 = info.height - 1
                box_pixels.append(px0); box_pixels.append(py0); box_pixels.append(px1); box_pixels.append(py1)
                # Calcular IoU e acurácia se houver ground truth
                var txt_path = path.replace('.bmp', '.txt')
                var gt_box = List[Int]()
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
                                    gt_box.append(Int(Float32(campos[i])))
                                except _:
                                    gt_box.append(0)
                except _:
                    gt_box = List[Int]()
                if len(gt_box) >= 4:
                    # Função IoU igual à do detector_model.mojo
                    fn calcular_iou_bbox(pred: List[Int], alvo: List[Int]) -> Float32:
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
                    var iou = calcular_iou_bbox(box_pixels, gt_box)
                    print("[MÉTRICA] IoU da predição:", iou)
                    var acuracia = 1.0 if iou > 0.5 else 0.0
                    print("[MÉTRICA] Acurácia bbox (IoU>0.5):", acuracia)
            else:
                if cfg.USE_DETECTOR_HEURISTIC:
                    print("O modelo não produziu predição válida para", path, "- usando heurística de fallback do adaptador")
                    var res = detect_pkg.detect_and_align_bbox(path)
                    var info2 = res[0].copy()
                    var bb = res[1].copy()
                    if len(bb) >= 4:
                        box_pixels.append(bb[0]); box_pixels.append(bb[1]); box_pixels.append(bb[2]); box_pixels.append(bb[3])
                else:
                    # Heuristic disabled by config: do not produce fallback bbox
                    pass
        except _:
            pass

        # build RGB flat list and draw predicted box (blue) if we have coordinates
        if len(box_pixels) == 4:
            print("[INFERÊNCIA] Caixa prevista: (", box_pixels[0], box_pixels[1], box_pixels[2], box_pixels[3], ") para imagem:", path)
        var W = info.width
        var H = info.height
        var img_rgb = List[Int](capacity=W * H * 3)
        if len(info.pixels) > 0:
            for ry in range(H):
                for rx in range(W):
                    var px = info.pixels[ry][rx]
                    var r = Int(px[0] * 255.0)
                    var g = Int(px[1] * 255.0)
                    var b = Int(px[2] * 255.0)
                    if r < 0: r = 0
                    if g < 0: g = 0
                    if b < 0: b = 0
                    if r > 255: r = 255
                    if g > 255: g = 255
                    if b > 255: b = 255
                    img_rgb.append(r); img_rgb.append(g); img_rgb.append(b)
        else:
            for i in range(W * H):
                img_rgb.append(0); img_rgb.append(0); img_rgb.append(0)

        if len(box_pixels) >= 4:
            var px0 = box_pixels[0]
            var py0 = box_pixels[1]
            var px1 = box_pixels[2]
            var py1 = box_pixels[3]
            var thickness = 2
            for t in range(thickness):
                for x in range(px0 + t, px1 - t + 1):
                    var ti_top = ((py0 + t) * W + x) * 3
                    img_rgb[ti_top + 0] = 0
                    img_rgb[ti_top + 1] = 0
                    img_rgb[ti_top + 2] = 255
                for x in range(px0 + t, px1 - t + 1):
                    var ti_bot = ((py1 - t) * W + x) * 3
                    img_rgb[ti_bot + 0] = 0
                    img_rgb[ti_bot + 1] = 0
                    img_rgb[ti_bot + 2] = 255
                for y in range(py0 + t, py1 - t + 1):
                    var ti_l = (y * W + (px0 + t)) * 3
                    img_rgb[ti_l + 0] = 0
                    img_rgb[ti_l + 1] = 0
                    img_rgb[ti_l + 2] = 255
                for y in range(py0 + t, py1 - t + 1):
                    var ti_r = (y * W + (px1 - t)) * 3
                    img_rgb[ti_r + 0] = 0
                    img_rgb[ti_r + 1] = 0
                    img_rgb[ti_r + 2] = 255

        var bmp_bytes = graficos_pkg.bmp.gerar_bmp_24bits_de_rgb(img_rgb, W, H)
        var base = os.path.basename(path)
        var out_name = "inf_" + String(idx) + "_" + base
        _ = dados_pkg.gravar_arquivo_binario(os.path.join(out_dir, out_name), bmp_bytes^)
        print("Salvo:", os.path.join(out_dir, out_name))

    print("Inferência concluída. Imagens salvas em", out_dir)
