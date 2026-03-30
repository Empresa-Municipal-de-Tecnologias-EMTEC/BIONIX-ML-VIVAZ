import bionix_ml.camadas.cnn as cnn_pkg
import model_detector as model_pkg
import bionix_ml.dados as dados_pkg
import bionix_ml.dados.arquivo as arquivo_pkg
import retina.retina_model as model_utils
import detector_dataset as dataset_pkg
import bionix_ml.graficos as graficos_pkg
import bionix_ml.uteis as uteis
import bionix_ml.nucleo.Tensor as tensor_defs
import retina.retina_anchor_generator as anchor_gen
import retina.retina_assigner as assigner
import retina.retina_utils as retina_utils
import os
import math


# Simple field-splitting helper
fn _split_fields(line: String) -> List[String]:
    var fields = List[String]()
    var cur = String("")
    for ch in line:
        if ch == '\t' or ch == ',' or ch == ' ':
            if len(cur) > 0:
                fields.append(cur)
                cur = String("")
        else:
            cur = cur + String(ch)
    if len(cur) > 0:
        fields.append(cur)
    return fields


# Minimal retina trainer that uses RGB patches and the retina_model helpers.
fn treinar_retina_minimal(mut detector: model_utils.RetinaFace, var dataset_dir: String, var altura: Int = 640, var largura: Int = 640,
                          var patch_size: Int = 64, var epocas: Int = 5, var taxa_aprendizado: Float32 = 0.0001,
                          var batch_size: Int = 4) raises -> String:

    # scheduler (ReduceLROnPlateau-like)
    var lr_atual = taxa_aprendizado
    var best_loss: Float32 = 1e30
    var scheduler_wait: Int = 0
    var scheduler_patience: Int = 2
    var scheduler_factor: Float32 = 0.5
    var min_lr: Float32 = 1e-7
    var min_delta: Float32 = 1e-6
    # IoU-based early stopping params
    var best_iou: Float32 = 0.0
    var iou_patience: Int = 5
    var iou_patience_count: Int = 0
    var iou_min_delta: Float32 = 1e-4
    var iou_target: Float32 = 0.75
    var iou_consec_required: Int = 3
    var iou_consec_count: Int = 0

    # try to load existing heads if available via detector
    var export_dir = os.path.join("MODELO", "retina_modelo")
    try:
        if not os.path.exists(export_dir):
            os.mkdir(export_dir)
    except _:
        pass

    # allocate placeholder heads (use detector's backend)
    var head_peso_cls = tensor_defs.Tensor(List[Int](), detector.bloco_cnn.tipo_computacao)
    var head_bias_cls = tensor_defs.Tensor(List[Int](), detector.bloco_cnn.tipo_computacao)
    var head_initialized = False

    # Try loading any existing saved regression/classification heads
    try:
        model_pkg.carregar_checkpoint(bloco, export_dir)
    except _:
        pass
    try:
        # attempt to read canonical classification head files
        var raw_w = arquivo_pkg.ler_arquivo_binario(os.path.join(export_dir, "peso_cls.bin"))
        var raw_b = arquivo_pkg.ler_arquivo_binario(os.path.join(export_dir, "bias_cls.bin"))
        if len(raw_w) > 0:
            try:
                var D = 0
                try:
                    D = detector.bloco_cnn.peso_saida.formato[0]
                except _:
                    D = 0
                if D > 0:
                    var shape_w = List[Int]()
                    shape_w.append(D); shape_w.append(4)
                        head_peso_cls = tensor_defs.Tensor(shape_w^, detector.bloco_cnn.tipo_computacao)
                    head_peso_cls.carregar_dados_bytes_bin(raw_w)
                    var shape_b = List[Int]()
                    shape_b.append(1); shape_b.append(4)
                        head_bias_cls = tensor_defs.Tensor(shape_b^, detector.bloco_cnn.tipo_computacao)
                    if len(raw_b) > 0:
                        head_bias_cls.carregar_dados_bytes_bin(raw_b)
                    head_initialized = True
            except _:
                head_initialized = False
    except _:
        pass

    # anchors (fixed for image size)
    var anchors = anchor_gen.gerar_anchors(largura)

    # build class lists from dataset_dir (expect structure: dataset_dir/train/<class>/*.bmp)
    var train_root = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_root):
        train_root = dataset_dir
    var class_names = List[String]()
    var class_images = List[List[String]]()
    try:
        for cls in os.listdir(train_root):
            var pcls = os.path.join(train_root, cls)
            if not os.path.isdir(pcls):
                continue
            class_names.append(cls)
            var imgs = List[String]()
            try:
                for f in os.listdir(pcls):
                    if f.endswith('.bmp'):
                        imgs.append(os.path.join(pcls, f))
            except _:
                pass
            class_images.append(imgs)
    except _:
        pass
    var class_ptrs = List[Int]()
    for _ in range(len(class_names)):
        class_ptrs.append(0)

    # `detector` is provided as parameter (RetinaFace); ensure its export dir is set
    try:
        detector.diretorio_modelo = export_dir
    except _:
        pass
    # reuse any loaded heads in detector or the local placeholders
    detector.cabeca_classificacao_peso = head_peso_cls
    detector.cabeca_classificacao_bias = head_bias_cls

    for ep in range(epocas):
        var soma_loss: Float32 = 0.0
        var count_pos: Int = 0
        var iou_sum: Float32 = 0.0
        var count_iou: Int = 0

        # build a mini-batch list: one image per class this epoch (safer for memory)
        var batch_paths = List[String]()
        for c in range(len(class_names)):
            var imgs = class_images[c]
            if len(imgs) == 0:
                continue
            var ptr = class_ptrs[c]
            if ptr >= len(imgs):
                ptr = 0
            batch_paths.append(imgs[ptr])
            class_ptrs[c] = ptr + 1

        var E = len(batch_paths)
        if E == 0:
            print("Nenhuma imagem para treinar; verifique DATASET")
            return "Falha: sem imagens"
        for bstart in range(0, E, batch_size):
            var bend = bstart + batch_size
            if bend > E:
                bend = E
            for i in range(bstart, bend):
                var path = batch_paths[i]
                var bmp = dados_pkg.carregar_bmp_rgb(path)
                if bmp.width == 0:
                    continue
                var img_matrix = List[List[List[Float32]]]()
                try:
                    img_matrix = graficos_pkg.redimensionar_matriz_rgb_nearest(bmp.pixels.copy()^, altura, largura)^
                except _:
                    continue

                # load ground-truth box if exists
                var tx0: Float32 = 0.0; var ty0: Float32 = 0.0; var tx1: Float32 = 0.0; var ty1: Float32 = 0.0
                var parsed: Bool = False
                try:
                    var box_path = path.replace('.bmp', '.box')
                    var lines = dados_pkg.carregar_txt_linhas(box_path)
                    if len(lines) > 0:
                        var parts = _split_fields(lines[0])
                        if len(parts) >= 4:
                            tx0 = Float32(uteis.parse_float_ascii(String(parts[0])))
                            ty0 = Float32(uteis.parse_float_ascii(String(parts[1])))
                            tx1 = Float32(uteis.parse_float_ascii(String(parts[2])))
                            ty1 = Float32(uteis.parse_float_ascii(String(parts[3])))
                            parsed = True
                except _:
                    pass

                var gt_x0: Int; var gt_y0: Int; var gt_x1: Int; var gt_y1: Int
                if parsed:
                    if tx0 > 1.5 or ty0 > 1.5 or tx1 > 1.5 or ty1 > 1.5:
                        gt_x0 = Int(tx0); gt_y0 = Int(ty0); gt_x1 = Int(tx1); gt_y1 = Int(ty1)
                    else:
                        gt_x0 = Int(tx0 * Float32(max(1, largura - 1)))
                        gt_y0 = Int(ty0 * Float32(max(1, altura - 1)))
                        gt_x1 = Int(tx1 * Float32(max(1, largura - 1)))
                        gt_y1 = Int(ty1 * Float32(max(1, altura - 1)))
                else:
                    gt_x0 = 0; gt_y0 = 0; gt_x1 = 0; gt_y1 = 0

                var gt_box = List[Int]()
                gt_box.append(gt_x0); gt_box.append(gt_y0); gt_box.append(gt_x1); gt_box.append(gt_y1)

                var gt_list = List[List[Int]]()
                if len(gt_box) == 4:
                    gt_list.append(gt_box.copy())

                var (labels, targets) = assigner.assignar_anchors(anchors, gt_list^)

                # for each positive anchor update small regression head and cls head
                for a_idx in range(len(anchors)):
                    if labels[a_idx] != 1:
                        continue
                    var a = anchors[a_idx].copy()
                    var ax = Int(a[0] - a[2] / 2.0); var ay = Int(a[1] - a[3] / 2.0)
                    var aw = Int(a[2]); var ah = Int(a[3])
                    if ax < 0: ax = 0
                    if ay < 0: ay = 0
                    if ax + aw > largura: aw = max(1, largura - ax)
                    if ay + ah > altura: ah = max(1, altura - ay)
                    var crop_rgb = graficos_pkg.crop_and_resize_rgb(img_matrix, ax, ay, ax + aw - 1, ay + ah - 1, patch_size, patch_size)

                    var in_shape = List[Int](); in_shape.append(1); in_shape.append(patch_size * patch_size * 3)
                    var tensor_in = tensor_defs.Tensor(in_shape^, detector.bloco_cnn.tipo_computacao)
                    for yy in range(patch_size):
                        for xx in range(patch_size):
                            var pix = crop_rgb[yy][xx]
                            var base = (yy * patch_size + xx) * 3
                            tensor_in.dados[base + 0] = pix[0]
                            tensor_in.dados[base + 1] = pix[1]
                            tensor_in.dados[base + 2] = pix[2]

                    var feats = cnn_pkg.extrair_features(detector.bloco_cnn, tensor_in)
                    var D = 0
                    try:
                        D = feats.formato[1]
                    except _:
                        D = 0

                    # initialize heads from features on first positive
                    if not head_initialized and D > 0:
                        var shape_w = List[Int](); shape_w.append(D); shape_w.append(4)
                        detector.bloco_cnn.peso_saida = tensor_defs.Tensor(shape_w^, detector.bloco_cnn.tipo_computacao)
                        var shape_b = List[Int](); shape_b.append(1); shape_b.append(4)
                        detector.bloco_cnn.bias_saida = tensor_defs.Tensor(shape_b^, detector.bloco_cnn.tipo_computacao)
                        for k in range(len(detector.bloco_cnn.peso_saida.dados)):
                            detector.bloco_cnn.peso_saida.dados[k] = 0.001
                        for k in range(len(detector.bloco_cnn.bias_saida.dados)):
                            detector.bloco_cnn.bias_saida.dados[k] = 0.0
                        var shape_cw = List[Int](); shape_cw.append(D); shape_cw.append(1)
                        head_peso_cls = tensor_defs.Tensor(shape_cw^, detector.bloco_cnn.tipo_computacao)
                        var shape_cb = List[Int](); shape_cb.append(1); shape_cb.append(1)
                        head_bias_cls = tensor_defs.Tensor(shape_cb^, detector.bloco_cnn.tipo_computacao)
                        for k in range(len(head_peso_cls.dados)):
                            head_peso_cls.dados[k] = 0.001
                        head_bias_cls.dados[0] = 0.0
                        head_initialized = True

                    # compute prediction and simple L1 update
                        var pred = List[Float32]()
                        for j in range(4):
                            var s: Float32 = 0.0
                            for d in range(D):
                                s = s + feats.dados[d] * detector.bloco_cnn.peso_saida.dados[d * 4 + j]
                            s = s + detector.bloco_cnn.bias_saida.dados[j]
                            pred.append(s)

                    var tgt = targets[a_idx].copy()
                    if len(tgt) < 4:
                        continue
                    # compute predicted box in image coordinates (same decoding as inference)
                    try:
                        var dx = pred[0]; var dy = pred[1]; var dw = pred[2]; var dh = pred[3]
                        var cx = a[0] + dx * a[2]
                        var cy = a[1] + dy * a[3]
                        var w = a[2] * Float32(math.exp(Float64(dw)))
                        var h = a[3] * Float32(math.exp(Float64(dh)))
                        var px0 = cx - w/2.0; var py0 = cy - h/2.0; var px1 = cx + w/2.0; var py1 = cy + h/2.0
                        var pred_box = List[Float32](); pred_box.append(px0); pred_box.append(py0); pred_box.append(px1); pred_box.append(py1)
                        var gt_box = List[Float32](); gt_box.append(Float32(gt_x0)); gt_box.append(Float32(gt_y0)); gt_box.append(Float32(gt_x1)); gt_box.append(Float32(gt_y1))
                        var iou_val = retina_utils.calcular_iou(pred_box, gt_box)
                        iou_sum = iou_sum + iou_val
                        count_iou = count_iou + 1
                    except _:
                        pass
                        for j in range(4):
                            var err = pred[j] - tgt[j]
                        # clamped gradient step
                        if err > 100.0: err = 100.0
                        if err < -100.0: err = -100.0
                        for d in range(D):
                            var grad_w = feats.dados[d] * err
                            if grad_w > 100.0: grad_w = 100.0
                            if grad_w < -100.0: grad_w = -100.0
                            detector.bloco_cnn.peso_saida.dados[d * 4 + j] = detector.bloco_cnn.peso_saida.dados[d * 4 + j] - lr_atual * grad_w
                        detector.bloco_cnn.bias_saida.dados[j] = detector.bloco_cnn.bias_saida.dados[j] - lr_atual * err
                        soma_loss = soma_loss + abs(err)
                    count_pos = count_pos + 1

        var avg_loss: Float32 = 0.0
        if count_pos > 0:
            avg_loss = soma_loss / Float32(max(1, count_pos * 4))
        print("Epoca", ep, "Avg L1 loss (pos anchors):", avg_loss, "Positives:", count_pos, "LR:", lr_atual)

        # save epoch state
        try:
            print("DBG: about to salvar_estado_modelo for epoch", ep)
            var meta_lines = List[String]()
            meta_lines.append("epoch:" + String(ep))
            meta_lines.append("avg_loss:" + String(avg_loss))
            meta_lines.append("positives:" + String(count_pos))
            meta_lines.append("lr:" + String(lr_atual))
            meta_lines.append("best_loss:" + String(best_loss))
            meta_lines.append("scheduler_wait:" + String(scheduler_wait))
            # delegate full workspace save to RetinaFace wrapper
            detector.cabeca_classificacao_peso = head_peso_cls
            detector.cabeca_classificacao_bias = head_bias_cls
            detector.treinamento_epoca = ep
            detector.treinamento_lr = lr_atual
            _ = detector.salvar_workspace(export_dir)
            print("DBG: returned from salvar_estado_modelo for epoch", ep)
        except _:
            pass

        # End-of-epoch: save one sample inference (pred vs GT) for up to first 10 classes
        try:
            try:
                var samples_dir = os.path.join(export_dir, "epoch_samples")
                if not os.path.exists(samples_dir):
                    os.mkdir(samples_dir)
            except _:
                pass

            var max_sample_classes = 10
            var limit = max_sample_classes if max_sample_classes < len(class_names) else len(class_names)
            for c in range(limit):
                try:
                    print("DBG: epoch_samples: processing class", c, "name", class_names[c])
                    var imgs = class_images[c]
                    if len(imgs) == 0:
                        continue
                    # pick first available image for the class
                    var img_path = imgs[0]
                    var bmp = dados_pkg.carregar_bmp_rgb(img_path, largura, altura)
                    if bmp.width == 0:
                        continue

                    # load GT box for this sample
                    var s_tx0: Float32 = 0.0; var s_ty0: Float32 = 0.0; var s_tx1: Float32 = 0.0; var s_ty1: Float32 = 0.0
                    var s_parsed: Bool = False
                    try:
                        var box_path = img_path.replace('.bmp', '.box')
                        var lines = dados_pkg.carregar_txt_linhas(box_path)
                        if len(lines) > 0:
                            var parts = _split_fields(lines[0])
                            if len(parts) >= 4:
                                s_tx0 = Float32(uteis.parse_float_ascii(String(parts[0])))
                                s_ty0 = Float32(uteis.parse_float_ascii(String(parts[1])))
                                s_tx1 = Float32(uteis.parse_float_ascii(String(parts[2])))
                                s_ty1 = Float32(uteis.parse_float_ascii(String(parts[3])))
                                s_parsed = True
                    except _:
                        pass

                    var s_gt_x0: Int; var s_gt_y0: Int; var s_gt_x1: Int; var s_gt_y1: Int
                    if s_parsed:
                        if s_tx0 > 1.5 or s_ty0 > 1.5 or s_tx1 > 1.5 or s_ty1 > 1.5:
                            s_gt_x0 = Int(s_tx0); s_gt_y0 = Int(s_ty0); s_gt_x1 = Int(s_tx1); s_gt_y1 = Int(s_ty1)
                        else:
                            s_gt_x0 = Int(s_tx0 * Float32(max(1, largura - 1)))
                            s_gt_y0 = Int(s_ty0 * Float32(max(1, altura - 1)))
                            s_gt_x1 = Int(s_tx1 * Float32(max(1, largura - 1)))
                            s_gt_y1 = Int(s_ty1 * Float32(max(1, altura - 1)))
                    else:
                        s_gt_x0 = 0; s_gt_y0 = 0; s_gt_x1 = 0; s_gt_y1 = 0

                    # prepare head bytes
                    var raw_w_bytes = List[Int]()
                    var raw_b_bytes = List[Int]()
                    try:
                        if head_initialized:
                            raw_w_bytes = head_peso_cls.dados_bytes_bin()
                            raw_b_bytes = head_bias_cls.dados_bytes_bin()
                        else:
                            try:
                                raw_w_bytes = arquivo_pkg.ler_arquivo_binario(os.path.join(export_dir, "peso_cls.bin"))
                            except _:
                                raw_w_bytes = List[Int]()
                            try:
                                raw_b_bytes = arquivo_pkg.ler_arquivo_binario(os.path.join(export_dir, "bias_cls.bin"))
                            except _:
                                raw_b_bytes = List[Int]()
                    except _:
                        pass

                    var boxes = List[List[Int]]()
                        try:
                            print("DBG: calling detector.inferir for epoch sample class", class_names[c])
                            # load bytes into detector heads if available
                            try:
                                if len(raw_w_bytes) > 0 and len(detector.cabeca_classificacao_peso.formato) >= 1:
                                    detector.cabeca_classificacao_peso.carregar_dados_bytes_bin(raw_w_bytes)
                                if len(raw_b_bytes) > 0 and len(detector.cabeca_classificacao_bias.formato) >= 1:
                                    detector.cabeca_classificacao_bias.carregar_dados_bytes_bin(raw_b_bytes)
                            except _:
                                pass
                            boxes = detector.inferir(bmp.pixels, largura, 1)
                            print("DBG: returned from detector.inferir for epoch sample class", class_names[c])
                        except _:
                            print("DBG: detector.inferir raised for epoch sample class", class_names[c])
                            boxes = List[List[Int]]()

                    if len(boxes) == 0:
                        print("Epoch sample", ep, "class", class_names[c], "NO_PRED_BOX gt_box", s_gt_x0, s_gt_y0, s_gt_x1, s_gt_y1)
                    else:
                        var pb = boxes[0]
                        print("Epoch sample", ep, "class", class_names[c], "pred_box", pb[0], pb[1], pb[2], pb[3], "gt_box", s_gt_x0, s_gt_y0, s_gt_x1, s_gt_y1)

                    # write sample file
                    try:
                        var sample_path = os.path.join(export_dir, "epoch_samples", "epoch_" + String(ep) + "_class_" + class_names[c] + ".txt")
                        var lines_out = List[String]()
                        if len(boxes) > 0:
                            var pb2 = boxes[0]
                            lines_out.append("pred_box: " + String(pb2[0]) + " " + String(pb2[1]) + " " + String(pb2[2]) + " " + String(pb2[3]))
                        else:
                            lines_out.append("pred_box: none")
                        lines_out.append("gt_box: " + String(s_gt_x0) + " " + String(s_gt_y0) + " " + String(s_gt_x1) + " " + String(s_gt_y1))
                        uteis.gravar_texto_seguro(sample_path, String("\n").join(lines_out.copy()^))
                    except _:
                        pass
                except _:
                    pass
        except _:
            pass

        # Evaluate IoU-based convergence and early stopping
        try:
            var mean_iou: Float32 = 0.0
            if count_iou > 0:
                mean_iou = iou_sum / Float32(count_iou)
            print("Epoca", ep, "Mean IoU (pos anchors):", mean_iou)
            # improvement?
            if mean_iou > best_iou + iou_min_delta:
                best_iou = mean_iou
                iou_patience_count = 0
                iou_consec_count = iou_consec_count + 1
            else:
                iou_patience_count = iou_patience_count + 1
                iou_consec_count = 0

            # Check target consecutive condition
            if best_iou >= iou_target and iou_consec_count >= iou_consec_required:
                print("Early stopping: reached IoU target", best_iou, "at epoch", ep)
                break

            # Patience-based stop
            if iou_patience_count >= iou_patience:
                print("Early stopping: no IoU improvement for", iou_patience, "epochs. Best IoU:", best_iou)
                break
        except _:
            pass

        # validation: run inference on a small validation set (if available)
        try:
            var val_root = os.path.join(dataset_dir, "val")
            if not os.path.exists(val_root):
                # fallback to a small subset of train if no explicit val set exists
                val_root = os.path.join(dataset_dir, "train")

            var printed_images = 0
            var max_images = 50
            if os.path.exists(val_root):
                try:
                    for cls in os.listdir(val_root):
                        var pcls = os.path.join(val_root, cls)
                        if not os.path.isdir(pcls):
                            continue
                        for f in os.listdir(pcls):
                            if printed_images >= max_images:
                                break
                            if not f.endswith('.bmp'):
                                continue
                            var img_path = os.path.join(pcls, f)
                            try:
                                print("DBG: validation: processing image", img_path)
                                var bmp = dados_pkg.carregar_bmp_rgb(img_path, largura, altura)
                                if bmp.width == 0:
                                    continue

                                # ground-truth box (if exists)
                                var gt_x0: Int = 0; var gt_y0: Int = 0; var gt_x1: Int = 0; var gt_y1: Int = 0
                                try:
                                    var box_path = img_path.replace('.bmp', '.box')
                                    var lines = dados_pkg.carregar_txt_linhas(box_path)
                                    if len(lines) > 0:
                                        var parts = _split_fields(lines[0])
                                        if len(parts) >= 4:
                                            var tx0 = Float32(uteis.parse_float_ascii(String(parts[0])))
                                            var ty0 = Float32(uteis.parse_float_ascii(String(parts[1])))
                                            var tx1 = Float32(uteis.parse_float_ascii(String(parts[2])))
                                            var ty1 = Float32(uteis.parse_float_ascii(String(parts[3])))
                                            if tx0 > 1.5 or ty0 > 1.5 or tx1 > 1.5 or ty1 > 1.5:
                                                gt_x0 = Int(tx0); gt_y0 = Int(ty0); gt_x1 = Int(tx1); gt_y1 = Int(ty1)
                                            else:
                                                gt_x0 = Int(tx0 * Float32(max(1, largura - 1)))
                                                gt_y0 = Int(ty0 * Float32(max(1, altura - 1)))
                                                gt_x1 = Int(tx1 * Float32(max(1, largura - 1)))
                                                gt_y1 = Int(ty1 * Float32(max(1, altura - 1)))
                                except _:
                                    pass

                                var raw_w_bytes = head_peso_cls.dados_bytes_bin() if head_initialized else List[Int]()
                                var raw_b_bytes = head_bias_cls.dados_bytes_bin() if head_initialized else List[Int]()
                                var boxes = List[List[Int]]()
                                try:
                                    print("DBG: calling detector.inferir for validation image", img_path)
                                    try:
                                        if len(raw_w_bytes) > 0 and len(detector.cabeca_classificacao_peso.formato) >= 1:
                                            detector.cabeca_classificacao_peso.carregar_dados_bytes_bin(raw_w_bytes)
                                        if len(raw_b_bytes) > 0 and len(detector.cabeca_classificacao_bias.formato) >= 1:
                                            detector.cabeca_classificacao_bias.carregar_dados_bytes_bin(raw_b_bytes)
                                    except _:
                                        pass
                                    boxes = detector.inferir(bmp.pixels, largura, 16)
                                    print("DBG: returned from detector.inferir for validation image", img_path)
                                except _:
                                    print("DBG: detector.inferir raised for validation image", img_path)
                                    boxes = List[List[Int]]()

                                # print detailed per-image diagnostics (only during validation)
                                if len(boxes) == 0:
                                    print("Img", img_path, "pred_boxes", 0)
                                else:
                                    for bi in range(len(boxes)):
                                        var pb = boxes[bi]
                                        var px0 = pb[0]; var py0 = pb[1]; var px1 = pb[2]; var py1 = pb[3]
                                        # compute center distance to GT if available
                                        var center_dist: Float32 = 0.0
                                        var w_ratio: Float32 = 0.0
                                        var h_ratio: Float32 = 0.0
                                        try:
                                            var pcx = (px0 + px1) / 2.0; var pcy = (py0 + py1) / 2.0
                                            var gcx = (gt_x0 + gt_x1) / 2.0; var gcy = (gt_y0 + gt_y1) / 2.0
                                            center_dist = Float32(math.sqrt(Float64((pcx - gcx)*(pcx - gcx) + (pcy - gcy)*(pcy - gcy))))
                                            var pw = max(1, px1 - px0); var ph = max(1, py1 - py0)
                                            var gw = max(1, gt_x1 - gt_x0); var gh = max(1, gt_y1 - gt_y0)
                                            w_ratio = Float32(pw) / Float32(gw) if gw > 0 else 0.0
                                            h_ratio = Float32(ph) / Float32(gh) if gh > 0 else 0.0
                                        except _:
                                            center_dist = 0.0; w_ratio = 0.0; h_ratio = 0.0

                                        print("Img", img_path, "AnchorIdx", bi, "pred_box", px0, py0, px1, py1, "gt_box", gt_x0, gt_y0, gt_x1, gt_y1, "center_dist", center_dist, "w_ratio", w_ratio, "h_ratio", h_ratio)

                                printed_images = printed_images + 1
                            except _:
                                pass
                        if printed_images >= max_images:
                            break
                except _:
                    pass
        except _:
            pass

        # scheduler step
        try:
            if count_pos > 0:
                if avg_loss + min_delta < best_loss:
                    best_loss = avg_loss
                    scheduler_wait = 0
                else:
                    scheduler_wait = scheduler_wait + 1
                if scheduler_wait >= scheduler_patience:
                    lr_atual = lr_atual * scheduler_factor
                    if lr_atual < min_lr:
                        lr_atual = min_lr
                    scheduler_wait = 0
        except _:
            pass

    # final save
    try:
        var final_meta_lines = List[String]()
        final_meta_lines.append("epoch:final")
        final_meta_lines.append("lr:" + String(lr_atual))
        final_meta_lines.append("best_loss:" + String(best_loss))
        final_meta_lines.append("scheduler_wait:" + String(scheduler_wait))
        detector.cabeca_classificacao_peso = head_peso_cls
        detector.cabeca_classificacao_bias = head_bias_cls
        _ = detector.salvar_workspace(export_dir)
    except _:
        pass

    return "Treino finalizado"
