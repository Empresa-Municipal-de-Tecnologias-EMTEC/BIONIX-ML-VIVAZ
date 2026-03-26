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
fn treinar_retina_minimal(mut bloco: cnn_pkg.BlocoCNN, var dataset_dir: String, var altura: Int = 640, var largura: Int = 640,
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

    # try to load existing heads if available
    var export_dir = os.path.join("MODELO", "retina_modelo")
    try:
        if not os.path.exists(export_dir):
            os.mkdir(export_dir)
    except _:
        pass

    # prepare dataset listing
    var class_images = List[List[String]]()
    var class_ptrs = List[Int]()
    var class_names = List[String]()
    try:
        var train_root = os.path.join(dataset_dir, "train")
        if not os.path.exists(train_root):
            train_root = dataset_dir
        for cls in os.listdir(train_root):
            var pcls = os.path.join(train_root, cls)
            if not os.path.isdir(pcls):
                continue
            class_names.append(cls)
            var imgs = List[String]()
            for f in os.listdir(pcls):
                if f.endswith('.bmp'):
                    imgs.append(os.path.join(pcls, f))
            class_images.append(imgs)
            class_ptrs.append(0)
    except _:
        pass

    # allocate placeholder heads
    var head_peso_cls = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)
    var head_bias_cls = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)
    var head_initialized = False

    # Try loading any existing saved regression/classification heads
    try:
        model_utils.carregar_regression_head(bloco, export_dir)
    except _:
        pass
    try:
        var head_bytes = model_utils.carregar_head_bytes(export_dir)
        if len(head_bytes) >= 2:
            var raw_w = head_bytes[0]
            var raw_b = head_bytes[1]
            if len(raw_w) > 0:
                try:
                    var D = 0
                    try:
                        D = bloco.peso_saida.formato[0]
                    except _:
                        D = 0
                    if D > 0:
                        var shape_w = List[Int]()
                        shape_w.append(D); shape_w.append(1)
                        head_peso_cls = tensor_defs.Tensor(shape_w^, bloco.tipo_computacao)
                        head_peso_cls.carregar_dados_bytes_bin(raw_w)
                        var shape_b = List[Int]()
                        shape_b.append(1); shape_b.append(1)
                        head_bias_cls = tensor_defs.Tensor(shape_b^, bloco.tipo_computacao)
                        if len(raw_b) > 0:
                            head_bias_cls.carregar_dados_bytes_bin(raw_b)
                        head_initialized = True
                except _:
                    head_initialized = False
    except _:
        pass

    # anchors (fixed for image size)
    var anchors = anchor_gen.gerar_anchors(largura)

    for ep in range(epocas):
        var soma_loss: Float32 = 0.0
        var count_pos: Int = 0

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
                    var tensor_in = tensor_defs.Tensor(in_shape^, bloco.tipo_computacao)
                    for yy in range(patch_size):
                        for xx in range(patch_size):
                            var pix = crop_rgb[yy][xx]
                            var base = (yy * patch_size + xx) * 3
                            tensor_in.dados[base + 0] = pix[0]
                            tensor_in.dados[base + 1] = pix[1]
                            tensor_in.dados[base + 2] = pix[2]

                    var feats = cnn_pkg.extrair_features(bloco, tensor_in)
                    var D = 0
                    try:
                        D = feats.formato[1]
                    except _:
                        D = 0

                    # initialize heads from features on first positive
                    if not head_initialized and D > 0:
                        var shape_w = List[Int](); shape_w.append(D); shape_w.append(4)
                        bloco.peso_saida = tensor_defs.Tensor(shape_w^, bloco.tipo_computacao)
                        var shape_b = List[Int](); shape_b.append(1); shape_b.append(4)
                        bloco.bias_saida = tensor_defs.Tensor(shape_b^, bloco.tipo_computacao)
                        for k in range(len(bloco.peso_saida.dados)):
                            bloco.peso_saida.dados[k] = 0.001
                        for k in range(len(bloco.bias_saida.dados)):
                            bloco.bias_saida.dados[k] = 0.0
                        var shape_cw = List[Int](); shape_cw.append(D); shape_cw.append(1)
                        head_peso_cls = tensor_defs.Tensor(shape_cw^, bloco.tipo_computacao)
                        var shape_cb = List[Int](); shape_cb.append(1); shape_cb.append(1)
                        head_bias_cls = tensor_defs.Tensor(shape_cb^, bloco.tipo_computacao)
                        for k in range(len(head_peso_cls.dados)):
                            head_peso_cls.dados[k] = 0.001
                        head_bias_cls.dados[0] = 0.0
                        head_initialized = True

                    # compute prediction and simple L1 update
                    var pred = List[Float32]()
                    for j in range(4):
                        var s: Float32 = 0.0
                        for d in range(D):
                            s = s + feats.dados[d] * bloco.peso_saida.dados[d * 4 + j]
                        s = s + bloco.bias_saida.dados[j]
                        pred.append(s)

                    var tgt = targets[a_idx].copy()
                    if len(tgt) < 4:
                        continue
                    for j in range(4):
                        var err = pred[j] - tgt[j]
                        # clamped gradient step
                        if err > 100.0: err = 100.0
                        if err < -100.0: err = -100.0
                        for d in range(D):
                            var grad_w = feats.dados[d] * err
                            if grad_w > 100.0: grad_w = 100.0
                            if grad_w < -100.0: grad_w = -100.0
                            bloco.peso_saida.dados[d * 4 + j] = bloco.peso_saida.dados[d * 4 + j] - lr_atual * grad_w
                        bloco.bias_saida.dados[j] = bloco.bias_saida.dados[j] - lr_atual * err
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
            model_utils.salvar_estado_modelo(bloco, head_peso_cls, head_bias_cls, export_dir, meta_lines)
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
                            var hb = model_utils.carregar_head_bytes(export_dir)
                            if len(hb) >= 2:
                                raw_w_bytes = hb[0]
                                raw_b_bytes = hb[1]
                    except _:
                        pass

                    var boxes = List[List[Int]]()
                    try:
                        print("DBG: calling inferir_com_bloco for epoch sample class", class_names[c])
                        boxes = model_utils.inferir_com_bloco(bloco, raw_w_bytes, raw_b_bytes, bmp.pixels, largura, 1)
                        print("DBG: returned from inferir_com_bloco for epoch sample class", class_names[c])
                    except _:
                        print("DBG: inferir_com_bloco raised for epoch sample class", class_names[c])
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
                                    print("DBG: calling inferir_com_bloco for validation image", img_path)
                                    boxes = model_utils.inferir_com_bloco(bloco, raw_w_bytes, raw_b_bytes, bmp.pixels, largura, 16)
                                    print("DBG: returned from inferir_com_bloco for validation image", img_path)
                                except _:
                                    print("DBG: inferir_com_bloco raised for validation image", img_path)
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
        model_utils.salvar_estado_modelo(bloco, head_peso_cls, head_bias_cls, export_dir, final_meta_lines)
    except _:
        pass

    return "Treino finalizado"
