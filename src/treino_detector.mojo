# Ponto de entrada de treino (Mojo) - produção POC
# Uso: mojo -I ../../BIONIX-ML src/treino.mojo


import config as cfg
import io_modelo as io
import gerar_retangulos_face as gerador_boxes
import detector_dataset as dataset_pkg
import detector_model as model_pkg
import bionix_ml.computacao as computacao_pkg
import os
import adaptadores.detectar_face as detect_pkg
import bionix_ml.dados.bmp as bmpmod
import bionix_ml.nucleo.Tensor as tensor_defs_local
import bionix_ml.dados as dados_pkg
import bionix_ml.graficos as graficos_pkg
import bionix_ml.computacao.dispatcher_tensor as dispatcher
import bionix_ml.camadas.cnn as cnn_pkg

fn main() raises -> None:
    print("Treino do detector — pipeline integrado (CPU/CUDA plugável)")

    # gerar pseudo-labels (.box) se configurado
    if cfg.GERAR_RETANGULOS_FACE:
        print("Gerador de retângulos ativado — processando dataset...")
        gerador_boxes.main()
        print("Processamento de .box finalizado")

    # backend selection: prefer CUDA if available, fallback to CPU
    var tipo = computacao_pkg.backend_nome_de_id(computacao_pkg.backend_cpu_id())

    var altura = 100
    var largura = 64
    print("Carregando dataset detector (full-image bbox loader preferred)...")
    var dataset_path = cfg.DATASET_ROOT + "/treino"
    if not os.path.isdir(dataset_path):
        dataset_path = cfg.DATASET_ROOT + "/train"
    # Try bbox dataset loader first (full-image -> bbox targets)
    var res_bbox = dataset_pkg.carregar_dataset_detector_bbox(dataset_path, altura, largura, tipo, 5)
    var x_det = res_bbox[0].copy()
    var y_det = res_bbox[1].copy()
    if len(x_det.formato) == 0 or x_det.formato[0] == 0:
        print("Falha ao construir dataset detector. Verifique arquivos e .txt de bboxes.")
        return

    print("Amostras detector:", x_det.formato[0], "| Features:", x_det.formato[1])

    # criar contexto com escolha de backend (device_kind)
    import bionix_ml.computacao.adaptadores.contexto as contexto_defs
    var device_kind = tipo
    var ctx = contexto_defs.criar_contexto_padrao(device_kind)

    var bloco = model_pkg.criar_bloco_detector(altura, largura, 6, 3, 3, ctx)
    # ajustar tipo de computacao conforme detecção de backend
    bloco.tipo_computacao = tipo
    # tentar carregar checkpoint se existir
    var carregado = model_pkg.carregar_checkpoint(bloco, cfg.MODEL_DIR)
    if carregado:
        print("Checkpoint carregado de", cfg.MODEL_DIR)
    else:
        print("Nenhum checkpoint encontrado — treinando do zero")
    # Decide training mode by target shape: if targets have 4 columns, run bbox regression
    try:
        if len(y_det.formato) == 2 and y_det.formato[1] == 4:
            print("Treinando detector (bbox regression)")
            # Pass dataset path so trainer can save per-epoch comparison images.
            # Enable early_stop with a small tolerance on center distance.
            var loss, pred_coords, img_path = model_pkg.treinar_detector_bbox_com_saida(bloco, x_det, y_det, 0.01, cfg.EPOCHS, 1, cfg.MODEL_DIR, dataset_path, 0.05, True)
            pred_coords = pred_coords.copy()
            print("Treino bbox finalizado. Loss final:", loss)
            # Imprime as coordenadas da caixa prevista para a última imagem de validação
            if len(pred_coords) >= 4 and len(img_path) > 0:
                print("[TREINO] Caixa prevista: (", pred_coords[0], pred_coords[1], pred_coords[2], pred_coords[3], ") para imagem:", img_path)
        else:
            print("Treinando detector com BCEWithLogits (PoC)")
            var loss = model_pkg.treinar_detector_bce(bloco, x_det, y_det, 0.01, cfg.EPOCHS, 10, dataset_path)
            print("Treino finalizado. Loss final:", loss)
    except _:
        print("Erro durante o treino do detector.")

    # salvar metadados simples (formatado como texto)
    var model_dir = cfg.MODEL_DIR
    var meta_path = model_dir + "/metadata_detector.txt"
    io._ensure_parent_dir(meta_path)
    var meta_str = "model: vivaz-detector-poc\n"
    meta_str = meta_str + "input_h: " + String(altura) + "\n"
    meta_str = meta_str + "input_w: " + String(largura) + "\n"
    _ = io.save_metadata(meta_path, meta_str)

    # salvar checkpoint final
    var ok = model_pkg.salvar_checkpoint(bloco, cfg.MODEL_DIR)
    if ok:
        print("Checkpoint salvo em", cfg.MODEL_DIR)
    else:
        print("Falha ao salvar checkpoint em", cfg.MODEL_DIR)

    # If bbox head metadata exists, load it and run a quick inference to draw predicted bbox
    try:
        var bbox_meta = io.load_metadata(os.path.join(cfg.MODEL_DIR, "bbox_head.txt"))
        if len(bbox_meta) > 0:
            print("Encontrado bbox_head.txt, gerando imagem de validação com previsão do modelo...")
            # parse simple meta format
            var lines = bbox_meta.split("\n")
            var weights_line = ""
            var bias_line = ""
            var feat_shape_line = ""
            for L in lines:
                if L.startswith("weights="):
                    weights_line = L.replace("weights=", "")
                elif L.startswith("bias="):
                    bias_line = L.replace("bias=", "")
                elif L.startswith("feat_shape="):
                    feat_shape_line = L.replace("feat_shape=", "")

            var w_vals = List[Float32]()
            if len(weights_line) > 0:
                for p in weights_line.split(","):
                    try:
                        w_vals.append(Float32(Float64(p)))
                    except _:
                        pass
            var b_vals = List[Float32]()
            if len(bias_line) > 0:
                for p in bias_line.split(","):
                    try:
                        b_vals.append(Float32(Float64(p)))
                    except _:
                        pass

            # pick a candidate image to visualize
            var found = False
            var cand = ""
            try:
                for ident in os.listdir(dataset_path):
                    var p_ident = os.path.join(dataset_path, ident)
                    if not os.path.isdir(p_ident):
                        continue
                    for f in os.listdir(p_ident):
                        if f.endswith('.bmp'):
                            cand = os.path.join(p_ident, f)
                            found = True
                            break
                    if found:
                        break
            except _:
                found = False

            if found:
                var info = bmpmod.zero_bmp()
                try:
                    var res = detect_pkg.detect_and_align_bbox(cand)
                    var tmp = res[0].copy()
                    if tmp.width > 0:
                        info = tmp^
                except _:
                    pass

                if info.width > 0:
                    # build grayscale resized input as used in dataset loader
                    var img_gray = List[List[Float32]]()
                    try:
                        img_gray = dados_pkg.carregar_bmp_grayscale_matriz(cand)
                    except _:
                        img_gray = List[List[Float32]]()
                    if len(img_gray) > 0:
                        var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(img_gray.copy()^, altura, largura)
                        var features = altura * largura
                        var shape_x = List[Int]()
                        shape_x.append(1); shape_x.append(features)
                        var x_s = tensor_defs_local.Tensor(shape_x^, tipo)
                        for i in range(features):
                            x_s.dados[i] = resized[i // largura][i % largura]
                        # extract features and compute pred
                        var feats = cnn_pkg.extrair_features(bloco, x_s)
                        # reconstruct head tensors
                        if len(w_vals) > 0 and len(b_vals) > 0:
                            var feat_dim = feats.formato[1]
                            var shape_hw = List[Int]()
                            shape_hw.append(feat_dim); shape_hw.append(4)
                            var head_w = tensor_defs_local.Tensor(shape_hw^, tipo)
                            var shape_hb = List[Int]()
                            shape_hb.append(1); shape_hb.append(4)
                            var head_b = tensor_defs_local.Tensor(shape_hb^, tipo)
                            for i in range(min(len(head_w.dados), len(w_vals))):
                                head_w.dados[i] = w_vals[i]
                            for j in range(min(len(head_b.dados), len(b_vals))):
                                head_b.dados[j] = b_vals[j]
                            var pred = dispatcher.multiplicar_matrizes(feats, head_w)
                            pred = dispatcher.adicionar_bias_coluna(pred, head_b)
                            # pred is [1,4] normalized coords
                            var px0 = Int(pred.dados[0] * Float32(info.width - 1))
                            var py0 = Int(pred.dados[1] * Float32(info.height - 1))
                            var px1 = Int(pred.dados[2] * Float32(info.width - 1))
                            var py1 = Int(pred.dados[3] * Float32(info.height - 1))
                            if px0 < 0: px0 = 0
                            if py0 < 0: py0 = 0
                            if px1 >= info.width: px1 = info.width - 1
                            if py1 >= info.height: py1 = info.height - 1
                            print("[TREINO] Caixa prevista: (", px0, py0, px1, py1, ") para imagem:", cand)
                            # build RGB and draw predicted box in blue
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
                            _ = dados_pkg.gravar_arquivo_binario(os.path.join("validacao", "validacao.bmp"), bmp_bytes^)
    except _:
        pass

    print("Treino detector concluído.")
