import bionix_ml.camadas.cnn as cnn_pkg
import bionix_ml.dados as dados_pkg
import detector_dataset as dataset_pkg
import bionix_ml.graficos as graficos_pkg
import bionix_ml.uteis as uteis
import bionix_ml.nucleo.Tensor as tensor_defs
import retina_anchor_generator as anchor_gen
import retina_assigner as assigner
import os
import math

# Treinador mínimo que realiza SGD por âncoras positivas usando crop-based features.
fn treinar_retina_minimal(mut bloco: cnn_pkg.BlocoCNN, var dataset_dir: String, var altura: Int = 640, var largura: Int = 640,
                          var patch_size: Int = 64, var epocas: Int = 5, var taxa_aprendizado: Float32 = 0.0001,
                          var batch_size: Int = 4) raises -> String:
    # Colete caminhos de imagens no dataset; as imagens serão carregadas por época (streaming)
    var img_paths = List[String]()
    # Build per-class image lists from DATASET/train (or dataset_dir if directly organized)
    var class_names = List[String]()
    var class_images = List[List[String]]()
    var labels = List[String]()
    var train_root = String("")
    try:
        train_root = os.path.join(dataset_dir, "train")
        if not os.path.exists(train_root):
            train_root = dataset_dir
        labels = os.listdir(train_root)
    except _:
        return "Falha ao listar dataset"

    for lbl in labels:
        var lbl_path = os.path.join(train_root, lbl)
        if not os.path.isdir(lbl_path):
            continue
        var files = List[String]()
        try:
            files = os.listdir(lbl_path)
        except _:
            continue
        var imgs = List[String]()
        for f in files:
            if f.endswith('.bmp'):
                imgs.append(os.path.join(lbl_path, f))
        if len(imgs) > 0:
            class_names.append(lbl)
            class_images.append(imgs^)

    var num_classes = len(class_names)
    if num_classes == 0:
        return "Dataset vazio"

    # pointers per class to cycle through images each epoch
    var class_ptrs = List[Int]()
    for _ in range(num_classes):
        class_ptrs.append(0)

    # Gera anchors uma vez no tamanho de entrada
    var anchors = anchor_gen.gerar_anchors(altura)

    # Inicializa cabeça se necessário: peso_reg [feat_dim,4], bias_reg [1,4]
    var head_initialized = False
    var head_peso_cls = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)
    var head_bias_cls = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)

    # Ensure model export directory exists and remove old weight/metadata files
    var model_dir = os.path.join("MODELO", "retina_modelo")
    try:
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        else:
            try:
                var old_files = os.listdir(model_dir)
                for of in old_files:
                    if of.startswith("peso_") or of.startswith("bias_") or of.startswith("metadata_") or of.endswith(".bin"):
                        try:
                            os.remove(os.path.join(model_dir, of))
                        except _:
                            pass
            except _:
                pass
    except _:
        pass

    for ep in range(epocas):
        var soma_loss: Float32 = 0.0
        var count_pos: Int = 0
        # build epoch image list: one image per class (advance class_ptrs each epoch)
        var epoch_imgs = List[String]()
        for c in range(num_classes):
            var ptr = class_ptrs[c]
            var imgs = class_images[c]
            if len(imgs) == 0:
                continue
            if ptr >= len(imgs):
                ptr = 0
            epoch_imgs.append(imgs[ptr])
            class_ptrs[c] = ptr + 1

        var E = len(epoch_imgs)
        if E == 0:
            continue

        for bstart in range(0, E, batch_size):
            var bend = bstart + batch_size
            if bend > E:
                bend = E
            for i in range(bstart, bend):
                var path = epoch_imgs[i]
                # carregar imagem e redimensionar na hora
                var img = List[List[Float32]]()
                try:
                    img = dados_pkg.carregar_bmp_grayscale_matriz(path)
                except _:
                    continue
                var img_matrix: List[List[Float32]] = List[List[Float32]]()
                try:
                    var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(img^, altura, largura)
                    img_matrix = resized^
                except _:
                    continue

                # parse bbox (.box preferred, fallback to .txt). .box may contain absolute pixels or normalized coords
                var box_path = path.replace('.bmp', '.box')
                var txt_path = path.replace('.bmp', '.txt')
                var tx0: Float32 = 0.0; var ty0: Float32 = 0.0; var tx1: Float32 = 0.0; var ty1: Float32 = 0.0
                var parsed: Bool = False
                try:
                    var linhas = dados_pkg.carregar_txt_linhas(box_path)
                    if len(linhas) > 0:
                        var parts = linhas[0].replace("\t", " ").replace(",", " ").split(" ")
                        var campos = List[String]()
                        for p in parts:
                            var ps = p.strip()
                            if len(ps) != 0:
                                campos.append(String(ps))
                        if len(campos) >= 4:
                            tx0 = Float32(uteis.parse_float_ascii(String(campos[0])))
                            ty0 = Float32(uteis.parse_float_ascii(String(campos[1])))
                            tx1 = Float32(uteis.parse_float_ascii(String(campos[2])))
                            ty1 = Float32(uteis.parse_float_ascii(String(campos[3])))
                            parsed = True
                except _:
                    pass
                if not parsed:
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
                                tx0 = Float32(uteis.parse_float_ascii(String(campos[0])))
                                ty0 = Float32(uteis.parse_float_ascii(String(campos[1])))
                                tx1 = Float32(uteis.parse_float_ascii(String(campos[2])))
                                ty1 = Float32(uteis.parse_float_ascii(String(campos[3])))
                                parsed = True
                    except _:
                        pass

                var gt_x0: Int; var gt_y0: Int; var gt_x1: Int; var gt_y1: Int
                if parsed:
                    # detect absolute vs normalized: values > 1.5 treated as absolute pixels
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

                # Assign anchors
                var gt_list = List[List[Int]]()
                if len(gt_box) == 4:
                    gt_list.append(gt_box.copy())
                var (labels, targets) = assigner.assignar_anchors(anchors, gt_list^)

                # For each positive anchor, crop image, extract features, compute pred and update head
                for a_idx in range(len(anchors)):
                    if labels[a_idx] != 1:
                        continue
                    if not head_initialized:
                        # build an initial head based on feature extractor output
                        # compute one patch to infer feat_dim
                        var a = anchors[a_idx].copy()
                        var ax = Int(a[0] - a[2] / 2.0); var ay = Int(a[1] - a[3] / 2.0)
                        var aw = Int(a[2]); var ah = Int(a[3])
                        # clamp
                        if ax < 0: ax = 0
                        if ay < 0: ay = 0
                        if ax + aw > largura: aw = max(1, largura - ax)
                        if ay + ah > altura: ah = max(1, altura - ay)
                        var crop = _crop_and_resize(img_matrix, ax, ay, ax + aw - 1, ay + ah - 1, patch_size, patch_size)
                        var in_shape = List[Int]()
                        in_shape.append(1); in_shape.append(patch_size * patch_size)
                        var tensor_in = tensor_defs.Tensor(in_shape^, bloco.tipo_computacao)
                        for yy in range(patch_size):
                            for xx in range(patch_size):
                                tensor_in.dados[yy * patch_size + xx] = crop[yy][xx]
                        var feats = cnn_pkg.extrair_features(bloco, tensor_in)
                        var feat_dim = feats.formato[1]
                        var shape_w = List[Int]()
                        shape_w.append(feat_dim); shape_w.append(4)
                        bloco.peso_saida = tensor_defs.Tensor(shape_w^, bloco.tipo_computacao)
                        var shape_b = List[Int]()
                        shape_b.append(1); shape_b.append(4)
                        bloco.bias_saida = tensor_defs.Tensor(shape_b^, bloco.tipo_computacao)
                        for k in range(len(bloco.peso_saida.dados)):
                            bloco.peso_saida.dados[k] = 0.001
                        for k in range(len(bloco.bias_saida.dados)):
                            bloco.bias_saida.dados[k] = 0.0
                        # classification head init (binary logits)
                        var shape_cw = List[Int]()
                        shape_cw.append(feat_dim); shape_cw.append(1)
                        head_peso_cls = tensor_defs.Tensor(shape_cw^, bloco.tipo_computacao)
                        var shape_cb = List[Int]()
                        shape_cb.append(1); shape_cb.append(1)
                        head_bias_cls = tensor_defs.Tensor(shape_cb^, bloco.tipo_computacao)
                        for k in range(len(head_peso_cls.dados)):
                            head_peso_cls.dados[k] = 0.001
                        head_bias_cls.dados[0] = 0.0
                        head_initialized = True

                    # crop anchor region and resize to patch_size
                    var a = anchors[a_idx].copy()
                    var ax = Int(a[0] - a[2] / 2.0); var ay = Int(a[1] - a[3] / 2.0)
                    var aw = Int(a[2]); var ah = Int(a[3])
                    if ax < 0: ax = 0
                    if ay < 0: ay = 0
                    if ax + aw > largura: aw = max(1, largura - ax)
                    if ay + ah > altura: ah = max(1, altura - ay)
                    var crop = _crop_and_resize(img_matrix, ax, ay, ax + aw - 1, ay + ah - 1, patch_size, patch_size)

                    # build input tensor and extract features
                    var in_shape = List[Int]()
                    in_shape.append(1); in_shape.append(patch_size * patch_size)
                    var tensor_in = tensor_defs.Tensor(in_shape^, bloco.tipo_computacao)
                    for yy in range(patch_size):
                        for xx in range(patch_size):
                            tensor_in.dados[yy * patch_size + xx] = crop[yy][xx]
                    var feats = cnn_pkg.extrair_features(bloco, tensor_in)

                    # sanitize features: replace NaN/inf/extreme values with 0.0
                    var D = feats.formato[1]
                    for fd in range(D):
                        var fv = feats.dados[fd]
                        if fv != fv or fv > 1e8 or fv < -1e8:
                            feats.dados[fd] = 0.0

                    # clamp feature magnitudes to avoid extreme activations
                    for fd in range(D):
                        var fv = feats.dados[fd]
                        if fv > 10.0:
                            feats.dados[fd] = 10.0
                        elif fv < -10.0:
                            feats.dados[fd] = -10.0

                    # compute pred = feats (1xD) * peso_saida (Dx4) + bias (1x4)
                    var pred = List[Float32]()
                    for j in range(4):
                        var s: Float32 = 0.0
                        for d in range(D):
                            s = s + feats.dados[d] * bloco.peso_saida.dados[d * 4 + j]
                        s = s + bloco.bias_saida.dados[j]
                        pred.append(s)

                    # validate pred values (skip anchor if invalid)
                    var invalid_pred: Bool = False
                    for pv in pred:
                        if pv != pv or pv > 1e8 or pv < -1e8:
                            invalid_pred = True
                            break
                    if invalid_pred:
                        continue

                    # decode pred deltas into box coordinates using anchor
                    var a_dec = anchors[a_idx].copy()
                    var a_cx = a_dec[0]; var a_cy = a_dec[1]; var a_w = a_dec[2]; var a_h = a_dec[3]
                    var pred_cx = pred[0] * a_w + a_cx
                    var pred_cy = pred[1] * a_h + a_cy
                    # clamp log-space predictions to avoid overflow
                    var tw_clamped = pred[2]
                    var th_clamped = pred[3]
                    if tw_clamped > 10.0: tw_clamped = 10.0
                    if tw_clamped < -10.0: tw_clamped = -10.0
                    if th_clamped > 10.0: th_clamped = 10.0
                    if th_clamped < -10.0: th_clamped = -10.0
                    var pred_w = Float32(math.exp(Float64(tw_clamped))) * a_w
                    var pred_h = Float32(math.exp(Float64(th_clamped))) * a_h
                    # ensure decoded coords are finite before casting to Int
                    if pred_cx != pred_cx or pred_cy != pred_cy or pred_w != pred_w or pred_h != pred_h:
                        continue
                    if pred_cx > 1e8 or pred_cy > 1e8 or pred_w > 1e8 or pred_h > 1e8:
                        continue
                    var pred_x0 = Int(pred_cx - pred_w / 2.0)
                    var pred_y0 = Int(pred_cy - pred_h / 2.0)
                    var pred_x1 = Int(pred_cx + pred_w / 2.0)
                    var pred_y1 = Int(pred_cy + pred_h / 2.0)

                    # target deltas from assigner (copy)
                    var tgt = targets[a_idx].copy()
                    if len(tgt) < 4:
                        continue
                    # compute ground-truth center and size for metrics
                    var gt_cx = Float32((gt_box[0] + gt_box[2]) / 2.0)
                    var gt_cy = Float32((gt_box[1] + gt_box[3]) / 2.0)
                    var gt_w = Float32(gt_box[2] - gt_box[0])
                    var gt_h = Float32(gt_box[3] - gt_box[1])

                    # instrumentation: center distance and width/height ratios
                    var center_dist = Float32(math.sqrt(Float64((pred_cx - gt_cx) * (pred_cx - gt_cx) + (pred_cy - gt_cy) * (pred_cy - gt_cy))))
                    var width_ratio = pred_w / (gt_w + 1e-6)
                    var height_ratio = pred_h / (gt_h + 1e-6)
                    print("Img", path, "AnchorIdx", a_idx, "pred_box", pred_x0, pred_y0, pred_x1, pred_y1, "gt_box", gt_box[0], gt_box[1], gt_box[2], gt_box[3], "center_dist", center_dist, "w_ratio", width_ratio, "h_ratio", height_ratio)
                    # compute simple L1 loss and gradients
                    for j in range(4):
                        var err = pred[j] - tgt[j]
                        soma_loss = soma_loss + abs(err)
                    count_pos = count_pos + 1

                    # simple classification (sigmoid + BCE) update for this positive anchor
                    var logit: Float32 = 0.0
                    for d in range(D):
                        logit = logit + feats.dados[d] * head_peso_cls.dados[d]
                    logit = logit + head_bias_cls.dados[0]
                    # stable sigmoid: clamp logit to avoid overflow
                    var sig: Float32 = 0.0
                    if logit > 50.0:
                        sig = 1.0
                    elif logit < -50.0:
                        sig = 0.0
                    else:
                        sig = 1.0 / (1.0 + Float32(math.exp(-Float64(logit))))
                    var cls_target: Float32 = 1.0
                    var eps: Float32 = 1e-6
                    soma_loss = soma_loss + (-cls_target * Float32(math.log(Float64(max(eps, sig)))))
                    var cls_err = sig - cls_target
                    for j in range(4):
                        var err = pred[j] - tgt[j]
                        # gradient clipping per-weight
                        if err > 100.0: err = 100.0
                        if err < -100.0: err = -100.0
                        for d in range(D):
                            var grad_w = feats.dados[d] * err
                            # clip gradient magnitude
                            if grad_w > 100.0: grad_w = 100.0
                            if grad_w < -100.0: grad_w = -100.0
                            bloco.peso_saida.dados[d * 4 + j] = bloco.peso_saida.dados[d * 4 + j] - taxa_aprendizado * grad_w
                        bloco.bias_saida.dados[j] = bloco.bias_saida.dados[j] - taxa_aprendizado * err
        var avg_loss: Float32 = 0.0
        if count_pos > 0:
            # guard against NaN soma_loss
            if soma_loss != soma_loss or soma_loss > 1e30 or soma_loss < -1e30:
                avg_loss = Float32(0.0)
            else:
                avg_loss = soma_loss / Float32(count_pos * 4)
        print("Epoca", ep, "Avg L1 loss (pos anchors):", avg_loss, "Positives:", count_pos)

        # save weights and metadata each epoch
        var export_dir = os.path.join("MODELO", "retina_modelo")
        try:
            if not os.path.exists(export_dir):
                os.mkdir(export_dir)
        except _:
            pass
        # binary weight files (overwrite same filenames each epoch)
        try:
            var wb_path = os.path.join(export_dir, "peso_reg.bin")
            var bb_path = os.path.join(export_dir, "bias_reg.bin")
            var ok_w = dados_pkg.gravar_arquivo_binario(wb_path, bloco.peso_saida.dados_bytes_bin())
            var ok_b = dados_pkg.gravar_arquivo_binario(bb_path, bloco.bias_saida.dados_bytes_bin())
            # classifier head (overwrite)
            var cw_path = os.path.join(export_dir, "peso_cls.bin")
            var cb_path = os.path.join(export_dir, "bias_cls.bin")
            var ok_cw = dados_pkg.gravar_arquivo_binario(cw_path, head_peso_cls.dados_bytes_bin())
            var ok_cb = dados_pkg.gravar_arquivo_binario(cb_path, head_bias_cls.dados_bytes_bin())
        except _:
            pass

        # metadata
        try:
            var meta_path = os.path.join(export_dir, "metadata.txt")
            var meta_lines = List[String]()
            meta_lines.append("epoch:" + String(ep))
            meta_lines.append("avg_loss:" + String(avg_loss))
            meta_lines.append("positives:" + String(count_pos))
            var meta_text = String("\n").join(meta_lines^)
            uteis.gravar_texto_seguro(meta_path, meta_text)
        except _:
            pass

    # salvar pesos finais
    var export_dir = os.path.join("MODELO", "retina")
    try:
        if not os.path.exists(export_dir):
            os.mkdir(export_dir)
    except _:
        pass
    var path_w = os.path.join(export_dir, "retina_pesos_reg.bin")
    try:
        var ok = dados_pkg.gravar_arquivo_binario(path_w, bloco.peso_saida.dados_bytes_bin())
    except _:
        ok = False
    return "Treino finalizado"


fn _crop_and_resize(img: List[List[Float32]], x0: Int, y0: Int, x1: Int, y1: Int, out_h: Int, out_w: Int) -> List[List[Float32]]:
    var h = len(img)
    var w = 0
    if h > 0:
        w = len(img[0])
    var xx0 = x0
    var yy0 = y0
    var xx1 = x1
    var yy1 = y1
    if xx0 < 0: xx0 = 0
    if yy0 < 0: yy0 = 0
    if xx1 >= w: xx1 = w - 1
    if yy1 >= h: yy1 = h - 1
    var src_h = yy1 - yy0 + 1
    var src_w = xx1 - xx0 + 1
    if src_h <= 0 or src_w <= 0:
        # return zero patch
        var out0 = List[List[Float32]]()
        for yy in range(out_h):
            var row = List[Float32]()
            for xx in range(out_w):
                row.append(0.0)
            out0.append(row^)
        return out0^
    var patch = List[List[Float32]]()
    for yy in range(yy0, yy1 + 1):
        var row = List[Float32]()
        for xx in range(xx0, xx1 + 1):
            row.append(img[yy][xx])
        patch.append(row^)
    var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(patch^, out_h, out_w)
    return resized^
