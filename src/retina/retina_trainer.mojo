import bionix_ml.camadas.cnn as cnn_pkg
import retina.model_detector as model_pkg
import bionix_ml.dados as dados_pkg
import bionix_ml.dados.arquivo as arquivo_pkg
import retina.retina_model as model_utils
import detector_dataset as dataset_pkg
import bionix_ml.graficos as graficos_pkg
import bionix_ml.uteis as uteis
import bionix_ml.nucleo.Tensor as tensor_defs
import retina.retina_gerador_ancoras as anchor_gen
import retina.retina_assigner as assigner
import retina.retina_utils as retina_utils
import os
import math
import diagnostics.logger as logger


# Sigmoid scalar (stable)
fn _sigmoid_scalar(var x: Float32) -> Float32:
    if x >= 0.0:
        var z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        var z = math.exp(x)
        return z / (1.0 + z)


# Separador de campos: divide string por espaço, tab ou vírgula
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
    # transfer local list to caller to avoid implicit copy
    return fields^


# Sanity-check an anchor vector (x, y, w, h). Returns True if anchor looks reasonable.
fn _anchor_sane(a: List[Float32], var max_width: Int, var max_height: Int) -> Bool:
    try:
        if len(a) < 4:
            return False
        for i in range(4):
            # ensure numeric and not NaN/inf by simple comparisons
            var v = a[i]
            if v != v:
                return False
            # reject extreme integer sentinel values that may indicate corruption
            if v < -1e12 or v > 1e12:
                return False
        # width/height must be positive and not absurdly large
        if a[2] <= 0.0 or a[3] <= 0.0:
            return False
        if a[2] > Float32(max_width * 10) or a[3] > Float32(max_height * 10):
            return False
        return True
    except _:
        return False


# Minimal retina trainer that uses RGB patches and the retina_model helpers.
fn treinar_retina_minimal(mut detector: model_utils.RetinaFace, var dataset_dir: String, var altura: Int = 640, var largura: Int = 640,
                          var patch_size: Int = 64, var epocas: Int = 5, var taxa_aprendizado: Float32 = 0.05,
                          var batch_size_inicio: Int = 8, var batch_size_fim: Int = 128, var samples_per_class: Int = 1, var randomize: Bool = True,
                          var early_stop: Bool = True, var allowed_classes: List[String] = List[String]()) raises -> String:

    print("Iniciando treino: altura=", altura, " largura=", largura, " patch=", patch_size, " epocas=", epocas)

    # scheduler (ReduceLROnPlateau-like)
    var lr_atual = taxa_aprendizado
    var lr_inicial = taxa_aprendizado
    var best_loss: Float32 = 1e30
    var scheduler_wait: Int = 0
    var scheduler_patience: Int = 5
    var scheduler_factor: Float32 = 0.75
    var min_lr: Float32 = 1e-4
    var min_delta: Float32 = 1e-6
    # IoU-based early stopping params
    var best_iou: Float32 = 0.0
    # Composite validation score (mean IoU minus FP penalty) for stricter early-stop
    var best_val_score: Float32 = -1.0
    var val_fp_penalty: Float32 = 0.08
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

    # Carrega checkpoint anterior se existir (pesos treinados em sessões anteriores)
    try:
        _ = model_pkg.carregar_checkpoint(detector.bloco_cnn, export_dir)
    except _:
        pass
    # Verifica se parâmetros persistidos são compatíveis; se não, ignora estado salvo
    try:
        var state_txt = arquivo_pkg.ler_arquivo_texto(os.path.join(export_dir, "retina_state.txt"))
        var saved_input_size = -1
        var saved_patch_size = -1
        for L in state_txt.split("\n"):
            try:
                if L.startswith("input_size:"):
                    saved_input_size = Int(L[11:])
                elif L.startswith("patch_size:"):
                    saved_patch_size = Int(L[11:])
            except _:
                continue
        # Checkpoint antigo (sem campos de params) -> resetar pra evitar pesos ruins
        if saved_input_size == -1 and saved_patch_size == -1:
            print("[WARN] Checkpoint sem metadados de parametros (formato antigo). Ignorando estado anterior.")
            detector.treinamento_epoca = -1
            detector.treinamento_lr = taxa_aprendizado
            head_initialized = False
        elif (saved_input_size > 0 and saved_input_size != largura) or (saved_patch_size > 0 and saved_patch_size != patch_size):
            print("[WARN] Parametros diferentes do checkpoint (input_size=", saved_input_size, " patch_size=", saved_patch_size, "). Ignorando estado anterior.")
            detector.treinamento_epoca = -1
            detector.treinamento_lr = taxa_aprendizado
            head_initialized = False
    except _:
        pass
    # Sanity-check regression bias after loading: if scale channels (dw/dh = indices 2,3)
    # have |bias| > 2.5 the head has diverged (predicts 13x+ scale change regardless of image).
    # This typically happens when training ran for many epochs with the un-normalized LR bug.
    # Reset the whole regression head to known-good initial values in that case.
    try:
        var bias_dw = detector.bloco_cnn.bias_saida.dados[2]
        var bias_dh = detector.bloco_cnn.bias_saida.dados[3]
        var diverged = False
        if bias_dw != bias_dw or bias_dh != bias_dh:
            diverged = True  # NaN
        if bias_dw > 2.5 or bias_dw < -2.5:
            diverged = True
        if bias_dh > 2.5 or bias_dh < -2.5:
            diverged = True
        if diverged:
            print("[WARN] Bias de escala divergiu (bias_dw=", bias_dw, " bias_dh=", bias_dh, "). Reinicializando cabeca de regressao.")
            var D2 = 0
            try:
                D2 = detector.bloco_cnn.peso_saida.formato[0]
            except _:
                D2 = 0
            if D2 > 0:
                var shape_w2 = List[Int](); shape_w2.append(D2); shape_w2.append(4)
                detector.bloco_cnn.peso_saida = tensor_defs.Tensor(shape_w2^, detector.bloco_cnn.tipo_computacao)
                var shape_b2 = List[Int](); shape_b2.append(1); shape_b2.append(4)
                detector.bloco_cnn.bias_saida = tensor_defs.Tensor(shape_b2^, detector.bloco_cnn.tipo_computacao)
                for k2 in range(len(detector.bloco_cnn.peso_saida.dados)):
                    var col2 = k2 % 4
                    if col2 == 2 or col2 == 3:
                        detector.bloco_cnn.peso_saida.dados[k2] = 1e-5
                    else:
                        detector.bloco_cnn.peso_saida.dados[k2] = 0.001
                for k2 in range(len(detector.bloco_cnn.bias_saida.dados)):
                    detector.bloco_cnn.bias_saida.dados[k2] = 0.0
            detector.treinamento_epoca = -1
            detector.treinamento_lr = taxa_aprendizado
            head_initialized = False
    except _:
        pass
    try:
        # Tenta carregar cabeças de classificação salvas
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
                    shape_w.append(D); shape_w.append(1)
                    head_peso_cls = tensor_defs.Tensor(shape_w^, detector.bloco_cnn.tipo_computacao)
                    _ = head_peso_cls.carregar_dados_bytes_bin(raw_w.copy())
                    var shape_b = List[Int]()
                    shape_b.append(1); shape_b.append(1)
                    head_bias_cls = tensor_defs.Tensor(shape_b^, detector.bloco_cnn.tipo_computacao)
                    if len(raw_b) > 0:
                        _ = head_bias_cls.carregar_dados_bytes_bin(raw_b.copy())
                    head_initialized = True
            except _:
                head_initialized = False
    except _:
        pass

    # anchors (fixed for image size)
    var anchors = anchor_gen.gerar_ancoras(largura)
    print("Anchors gerados:", len(anchors))
    # snapshot anchors immediately after generation to detect later corruption
    var anchors_snapshot = List[List[Float32]]()
    for i in range(len(anchors)):
        anchors_snapshot.append(anchors[i].copy()^)
    # lightweight checksum to detect where anchors mutate (sum of first 100 anchors weighted)
    var anchors_checksum: Float32 = Float32(0.0)
    try:
        var limit_ck = 100
        if len(anchors) < limit_ck:
            limit_ck = len(anchors)
        for i in range(limit_ck):
            var a_ck = anchors[i].copy()
            for v in a_ck:
                anchors_checksum = anchors_checksum + v * Float32(i + 1)
    except _:
        anchors_checksum = Float32(-1.0)
    import diagnostics.logger as logger
    # quick scan for corrupted anchors right after generation
    try:
        var bad = 0
        var bad_examples = List[Int]()
        for i in range(len(anchors)):
            try:
                var ar = anchors[i].copy()
                var ok = True
                for v in ar:
                    if v != v:
                        ok = False
                        break
                if not ok:
                    bad = bad + 1
                    if len(bad_examples) < 10:
                        bad_examples.append(i)
            except _:
                bad = bad + 1
                if len(bad_examples) < 10:
                    bad_examples.append(i)
        if bad > 0:
            try:
                print("[DBG] varredura da geração de âncoras: encontrado", bad, "corrupted anchors immediately after generation; sample indices:")
                for idx in bad_examples:
                    try:
                        print(idx)
                    except _:
                        pass
                for idx in bad_examples:
                    try:
                        var aex = anchors[idx].copy()
                        print("[DBG] varredura : idx", idx, "vals:", aex[0], aex[1], aex[2], aex[3])
                    except _:
                        print("[DBG] varredura da geraçao de ancoras: idx", idx, "(unable to format)")
            except _:
                pass
    except _:
        pass

    # build class lists from dataset_dir (expect structure: dataset_dir/train/<class>/*.bmp)
    var train_root = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_root):
        train_root = dataset_dir
    var class_names = List[String]()
    var class_images = List[List[String]]()
    var total_images: Int = 0
    try:
        for cls in os.listdir(train_root):
            var pcls = os.path.join(train_root, cls)
            if not os.path.isdir(pcls):
                continue
            # optionally filter dataset classes by `allowed_classes` param
            if len(allowed_classes) > 0:
                var keep = False
                for ac in allowed_classes:
                    if ac == cls:
                        keep = True
                        break
                if not keep:
                    continue
            class_names.append(cls)
            var imgs = List[String]()
            try:
                for f in os.listdir(pcls):
                    if f.endswith('.bmp'):
                        imgs.append(os.path.join(pcls, f))
            except _:
                pass
            var imgs_len = len(imgs)
            class_images.append(imgs^)
            total_images = total_images + imgs_len
            if total_images > 500:
                break
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
    # reuse any loaded heads in detector or the local placeholders (transfer ownership)
    try:
        detector.cabeca_classificacao_peso = head_peso_cls.copy()
    except _:
        pass
    try:
        detector.cabeca_classificacao_bias = head_bias_cls.copy()
    except _:
        pass

    for ep in range(epocas):
        print("Epoca", ep, "/", epocas - 1, "iniciando...")
        # check anchors checksum at epoch start to detect mutation timing
        # ensure `cur_ck` is always initialized outside the try/except
        var cur_ck: Float32 = Float32(-2.0)
        try:
            cur_ck = Float32(0.0)
            var limit_ck2 = 100
            if len(anchors) < limit_ck2:
                limit_ck2 = len(anchors)
            for i in range(limit_ck2):
                var a_ck2 = anchors[i].copy()
                for v in a_ck2:
                    cur_ck = cur_ck + v * Float32(i + 1)
        except _:
            cur_ck = Float32(-2.0)
        if cur_ck != anchors_checksum:
            print("[DBG-ERR] anchors checksum mismatch at epoch", ep, "gen_ck=", anchors_checksum, "cur_ck=", cur_ck)
            # attempt to locate first differing anchor
            try:
                var first_diff = -1
                var scan_lim = min(len(anchors), 200)
                for ii in range(scan_lim):
                    var orig = anchors_snapshot[ii].copy()
                    var cur = anchors[ii].copy()
                    var diff = False
                    for jj in range(4):
                        if orig[jj] != cur[jj]:
                            diff = True
                            break
                    if diff:
                        first_diff = ii
                        break
                print("[DBG-ERR] first differing anchor idx=", first_diff)
                if first_diff >= 0:
                    try:
                        var osnap = anchors_snapshot[first_diff].copy()
                        var ocurr = anchors[first_diff].copy()
                        print("[DBG-ERR] snapshot=", osnap[0], osnap[1], osnap[2], osnap[3], "current=", ocurr[0], ocurr[1], ocurr[2], ocurr[3])
                    except _:
                        pass
            except _:
                pass
        var soma_loss: Float32 = 0.0
        var count_pos: Int = 0
        var iou_sum: Float32 = 0.0
        var count_iou: Int = 0
        var penalty_count: Int = 0

        # Escalonamento dinâmico do lote: batch_size_inicio→batch_size_fim conforme LR cai
        var batch_size_atual: Int
        var lr_range = lr_inicial - min_lr
        if lr_range > 0.0:
            var t = (lr_inicial - lr_atual) / lr_range
            if t < 0.0: t = 0.0
            if t > 1.0: t = 1.0
            batch_size_atual = batch_size_inicio + Int(Float32(batch_size_fim - batch_size_inicio) * t)
        else:
            batch_size_atual = batch_size_fim
        var samples_per_class_atual = max(1, batch_size_atual // max(1, len(class_names)))

        # build mini-batch list. Default: sample `samples_per_class_atual` images per class.
        # If `randomize` is True, pick random images per class; otherwise use round-robin pointers.
        var batch_paths = List[String]()
        for c in range(len(class_names)):
            var imgs = class_images[c].copy()
            var nimgs = len(imgs)
            if nimgs == 0:
                continue
            # number of samples to draw for this class
            var to_draw = samples_per_class_atual
            if to_draw <= 0:
                to_draw = 1
            if randomize:
                # LCG seeded by epoch, class and a constant offset to avoid seed=0 when ep=c=0
                var seed = (ep * 1664525 + c * 1013904223 + 0xabcdef) & 0x7fffffff
                for s in range(min(to_draw, nimgs)):
                    seed = (1103515245 * seed + 12345) & 0x7fffffff
                    var idx = seed % nimgs
                    batch_paths.append(imgs[idx])
            else:
                var ptr = class_ptrs[c]
                for s in range(min(to_draw, nimgs)):
                    if ptr >= nimgs:
                        ptr = 0
                    batch_paths.append(imgs[ptr])
                    ptr = ptr + 1
                class_ptrs[c] = ptr

        var E = len(batch_paths)
        if E == 0:
            print("Nenhuma imagem para treinar; verifique DATASET")
            return "Falha: sem imagens"
        for bstart in range(0, E, batch_size_inicio):
            var bend = bstart + batch_size_inicio
            if bend > E:
                bend = E
            for i in range(bstart, bend):
                var path = batch_paths[i]
                var bmp = dados_pkg.carregar_bmp_rgb(path)
                if bmp.width == 0:
                    continue
                # Do not pre-resize the full image (high peak allocations).
                # We'll attempt flat-buffer crops first; only if that fails we'll
                # lazily construct a nested `img_matrix` for the fallback path.
                img_matrix = List[List[List[Float32]]]()

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
                        # Pixel coords in original image space → scale to training resolution
                        var sx = Float32(largura) / Float32(max(1, bmp.width))
                        var sy = Float32(altura)  / Float32(max(1, bmp.height))
                        gt_x0 = Int(tx0 * sx); gt_y0 = Int(ty0 * sy)
                        gt_x1 = Int(tx1 * sx); gt_y1 = Int(ty1 * sy)
                    else:
                        # Normalized [0,1] → training resolution
                        gt_x0 = Int(tx0 * Float32(largura))
                        gt_y0 = Int(ty0 * Float32(altura))
                        gt_x1 = Int(tx1 * Float32(largura))
                        gt_y1 = Int(ty1 * Float32(altura))
                else:
                    gt_x0 = 0; gt_y0 = 0; gt_x1 = 0; gt_y1 = 0

                var gt_box = List[Int]()
                gt_box.append(gt_x0); gt_box.append(gt_y0); gt_box.append(gt_x1); gt_box.append(gt_y1)

                var gt_list = List[List[Int]]()
                if len(gt_box) == 4:
                    gt_list.append(gt_box.copy())

                # call assigner with explicit copies to avoid moving local ownership
                var res = assigner.assignar_anchors(anchors.copy(), gt_list.copy())
                # avoid copying large lists here (reduce peak memory)
                var labels = res.labels.copy()
                var targets = res.targets.copy()

                # Quick checksum of anchors immediately after assigner to detect early mutations
                try:
                    var post_assign_ck: Float32 = Float32(0.0)
                    var lim_ck = 100
                    if len(anchors) < lim_ck:
                        lim_ck = len(anchors)
                    for i_ck in range(lim_ck):
                        var a_ckv = anchors[i_ck].copy()
                        for vv in a_ckv:
                            post_assign_ck = post_assign_ck + vv * Float32(i_ck + 1)
                    if post_assign_ck != anchors_checksum:
                        print("[DBG-ERR] anchors checksum changed after assigner: gen_ck=", anchors_checksum, "post_assign_ck=", post_assign_ck)
                        # locate first diff
                        try:
                            var first_diff2 = -1
                            for ii in range(min(len(anchors), 200)):
                                var o2 = anchors_snapshot[ii].copy()
                                var c2 = anchors[ii].copy()
                                var d2 = False
                                for jj in range(4):
                                    if o2[jj] != c2[jj]:
                                        d2 = True; break
                                if d2:
                                    first_diff2 = ii; break
                            print("[DBG-ERR] first differing anchor after assigner idx=", first_diff2)
                        except _:
                            pass
                        return "Falha: anchors mutated after assigner"
                except _:
                    pass

                # Diagnostic: report basic /proc/meminfo lines (available on Linux/WSL)
                try:
                    var memtxt = arquivo_pkg.ler_arquivo_texto("/proc/meminfo")
                    if len(memtxt) > 0:
                        var memlines = memtxt.split("\n")
                except _:
                    pass

                # for each positive anchor update regression+cls heads, and for sampled negative anchors update cls head
                # limit processing of positive anchors per image to avoid peak memory spikes
                var pos_processed: Int = 0
                var _max_pos_per_image: Int = 8
                var neg_step_counter: Int = 0
                var neg_processed_anchor: Int = 0
                var max_neg_per_image: Int = _max_pos_per_image * 3
                var neg_sample_rate: Int = 30

                # Tensor grayscale [1, patch*patch] — tamanho correto para BlocoCNN de patch_size×patch_size
                var in_shape = List[Int](); in_shape.append(1); in_shape.append(patch_size * patch_size)
                
                for a_idx in range(len(anchors)):
                    # Exit early when both positive and negative anchor quotas are satisfied
                    if pos_processed >= _max_pos_per_image and neg_processed_anchor >= max_neg_per_image:
                        break
                    var is_pos_anchor = (labels[a_idx] == 1)
                    var is_neg_anchor = False
                    if is_pos_anchor:
                        pos_processed = pos_processed + 1
                        if pos_processed > _max_pos_per_image:
                            continue  # quota reached; keep iterating for negative samples
                    elif labels[a_idx] == 0:
                        neg_step_counter = neg_step_counter + 1
                        if neg_step_counter % neg_sample_rate != 0 or neg_processed_anchor >= max_neg_per_image:
                            continue
                        neg_processed_anchor = neg_processed_anchor + 1
                        is_neg_anchor = True
                    else:
                        continue
                    
                    # Check anchors checksum immediately before processing this anchor
                    try:
                        var pre_anchor_ck: Float32 = Float32(0.0)
                        var lim_ck2 = 100
                        if len(anchors) < lim_ck2:
                            lim_ck2 = len(anchors)
                        for i_ck2 in range(lim_ck2):
                            var a_ckv2 = anchors[i_ck2].copy()
                            for vv2 in a_ckv2:
                                pre_anchor_ck = pre_anchor_ck + vv2 * Float32(i_ck2 + 1)
                        if pre_anchor_ck != anchors_checksum:
                            logger.error_log(String("anchors_checksum_changed_before_anchor"), String("a_idx=") + String(a_idx) + String(" gen_ck=") + String(anchors_checksum) + String(" pre_anchor_ck=") + String(pre_anchor_ck))
                            # locate first diff near reported idx
                            try:
                                var first_diff3 = -1
                                for ii in range(min(len(anchors), 200)):
                                    var o3 = anchors_snapshot[ii].copy()
                                    var c3 = anchors[ii].copy()
                                    var d3 = False
                                    for jj in range(4):
                                        if o3[jj] != c3[jj]:
                                            d3 = True; break
                                    if d3:
                                        first_diff3 = ii; break
                                logger.error_log(String("first_differing_anchor_pre_anchor"), String(first_diff3))
                                # recent write-trace disabled (instrumentation removed)
                            except _:
                                pass
                            return "Falha: anchors mutated before processing anchor"
                    except _:
                        pass

                    # Create a fresh tensor for each anchor to isolate memory
                    var tensor_in = tensor_defs.Tensor(in_shape.copy(), detector.bloco_cnn.tipo_computacao)

                    var a = anchors[a_idx].copy()
                    # defensive validation: skip anchors with nonsensical sizes or corrupted values
                    if not _anchor_sane(a, largura, altura):
                        continue
                    var ax = Int(a[0] - a[2] / 2.0); var ay = Int(a[1] - a[3] / 2.0)
                    var aw = Int(a[2]); var ah = Int(a[3])
                    if ax < 0: ax = 0
                    if ay < 0: ay = 0
                    if ax + aw > largura: aw = max(1, largura - ax)
                    if ay + ah > altura: ah = max(1, altura - ay)
                    # Anchor is in training resolution → scale crop region to original image space
                    var _ax1 = Int(Float32(ax + aw) * Float32(bmp.width)  / Float32(largura))
                    var _ay1 = Int(Float32(ay + ah) * Float32(bmp.height) / Float32(altura))
                    ax = Int(Float32(ax) * Float32(bmp.width)  / Float32(largura))
                    ay = Int(Float32(ay) * Float32(bmp.height) / Float32(altura))
                    aw = max(1, _ax1 - ax)
                    ah = max(1, _ay1 - ay)

                    # Use local copy of pixels
                    var bmp_flat = bmp.flat_pixels.copy()
                    var bmp_channels = bmp.channels
                    var bmp_w = bmp.width; var bmp_h = bmp.height
                    
                    # Redimensiona região da anchor para patch_size×patch_size e converte para escala de cinza
                    var yy0_local = ay; var xx0_local = ax
                    var src_h_local = ah; var src_w_local = aw
                    if src_h_local <= 0: src_h_local = 1
                    if src_w_local <= 0: src_w_local = 1
                    var flat_len = len(bmp_flat)
                    for yy in range(patch_size):
                        var src_y = (yy * src_h_local) // patch_size
                        if src_y < 0: src_y = 0
                        if src_y >= src_h_local: src_y = src_h_local - 1
                        var img_y = yy0_local + src_y
                        if img_y < 0: img_y = 0
                        if img_y >= bmp_h: img_y = bmp_h - 1
                        for xx in range(patch_size):
                            var src_x = (xx * src_w_local) // patch_size
                            if src_x < 0: src_x = 0
                            if src_x >= src_w_local: src_x = src_w_local - 1
                            var img_x = xx0_local + src_x
                            if img_x < 0: img_x = 0
                            if img_x >= bmp_w: img_x = bmp_w - 1
                            var idx_flat = (img_y * bmp_w + img_x) * bmp_channels
                            var r: Float32 = 0.0; var g: Float32 = 0.0; var b: Float32 = 0.0
                            if bmp_channels >= 3 and idx_flat + 2 < flat_len:
                                r = bmp_flat[idx_flat + 0]; g = bmp_flat[idx_flat + 1]; b = bmp_flat[idx_flat + 2]
                            elif idx_flat < flat_len:
                                r = bmp_flat[idx_flat]; g = r; b = r
                            tensor_in.dados[yy * patch_size + xx] = Float32(0.299) * r + Float32(0.587) * g + Float32(0.114) * b

                    var feats = cnn_pkg.extrair_features(detector.bloco_cnn, tensor_in)
                    var D = 0
                    try:
                        D = feats.formato[1]
                    except _:
                        D = 0

                    # Normalized weight learning rate: divides by squared feature L2-norm so that
                    # the net prediction change per step ≈ lr_atual * err (same magnitude as bias),
                    # preventing the D-feature dot product from overshooting its target each iteration.
                    var feat_norm_sq: Float32 = 0.0
                    for d_ns in range(D):
                        feat_norm_sq = feat_norm_sq + feats.dados[d_ns] * feats.dados[d_ns]
                    var lr_w = lr_atual / max(Float32(1.0), feat_norm_sq)

                    # Ensure regression head has correct shape [D, 4] regardless of whether
                    # the classification head was loaded from a checkpoint. This guards against
                    # the case where head_initialized=True (cls head loaded) but peso_saida still
                    # has its default shape [D, 1] from BlocoCNN.__init__ — which would cause
                    # OOB writes of the form dados[d*4+j] on a 1-column buffer.
                    if D > 0:
                        var reg_shape_ok = False
                        try:
                            reg_shape_ok = (len(detector.bloco_cnn.peso_saida.formato) == 2 and
                                            detector.bloco_cnn.peso_saida.formato[0] == D and
                                            detector.bloco_cnn.peso_saida.formato[1] == 4)
                        except _:
                            reg_shape_ok = False
                        if not reg_shape_ok:
                            var shape_w = List[Int](); shape_w.append(D); shape_w.append(4)
                            detector.bloco_cnn.peso_saida = tensor_defs.Tensor(shape_w^, detector.bloco_cnn.tipo_computacao)
                            var shape_b = List[Int](); shape_b.append(1); shape_b.append(4)
                            detector.bloco_cnn.bias_saida = tensor_defs.Tensor(shape_b^, detector.bloco_cnn.tipo_computacao)
                            # All channels start at zero bias: exp(0)=1× preserves anchor size exactly.
                            # Tiny weights ensure the first prediction = anchor size (neutral start).
                            for k in range(len(detector.bloco_cnn.peso_saida.dados)):
                                var col = k % 4
                                if col == 2 or col == 3:
                                    detector.bloco_cnn.peso_saida.dados[k] = 1e-5
                                else:
                                    detector.bloco_cnn.peso_saida.dados[k] = 0.001
                            for k in range(len(detector.bloco_cnn.bias_saida.dados)):
                                detector.bloco_cnn.bias_saida.dados[k] = 0.0
                        if not head_initialized:
                            var shape_cw = List[Int](); shape_cw.append(D); shape_cw.append(1)
                            head_peso_cls = tensor_defs.Tensor(shape_cw^, detector.bloco_cnn.tipo_computacao)
                            var shape_cb = List[Int](); shape_cb.append(1); shape_cb.append(1)
                            head_bias_cls = tensor_defs.Tensor(shape_cb^, detector.bloco_cnn.tipo_computacao)
                            for k in range(len(head_peso_cls.dados)):
                                head_peso_cls.dados[k] = 0.001
                            head_bias_cls.dados[0] = 0.0
                            head_initialized = True

                    # Negative anchor: update classification head only (target=0), skip regression
                    if is_neg_anchor:
                        if head_initialized and D > 0:
                            var cls_pred_n: Float32 = 0.0
                            for d_n in range(D):
                                cls_pred_n = cls_pred_n + feats.dados[d_n] * head_peso_cls.dados[d_n]
                            cls_pred_n = cls_pred_n + head_bias_cls.dados[0]
                            # BCE-with-logits gradient: sigmoid(pred) - target (target=0)
                            var grad_n: Float32 = _sigmoid_scalar(cls_pred_n)
                            # clamp gradient to avoid extreme updates
                            if grad_n > 5.0: grad_n = 5.0
                            if grad_n < -5.0: grad_n = -5.0
                            for d_n in range(D):
                                var cg_n = feats.dados[d_n] * grad_n
                                if cg_n > 10.0: cg_n = 10.0
                                if cg_n < -10.0: cg_n = -10.0
                                head_peso_cls.dados[d_n] = head_peso_cls.dados[d_n] - lr_w * cg_n
                            head_bias_cls.dados[0] = head_bias_cls.dados[0] - lr_atual * grad_n
                        continue

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
                    var penalty_factor: Float32 = 1.0
                    try:
                        var dx = pred[0]; var dy = pred[1]; var dw = pred[2]; var dh = pred[3]
                        var cx = a[0] + dx * a[2]
                        var cy = a[1] + dy * a[3]
                        if dw > 4.0: dw = 4.0
                        if dw < -4.0: dw = -4.0
                        if dh > 4.0: dh = 4.0
                        if dh < -4.0: dh = -4.0
                        var w = a[2] * Float32(math.exp(Float64(dw)))
                        var h = a[3] * Float32(math.exp(Float64(dh)))
                        var px0 = cx - w/2.0; var py0 = cy - h/2.0; var px1 = cx + w/2.0; var py1 = cy + h/2.0
                        var pred_box = List[Float32](); pred_box.append(px0); pred_box.append(py0); pred_box.append(px1); pred_box.append(py1)
                        var gt_box = List[Float32](); gt_box.append(Float32(gt_x0)); gt_box.append(Float32(gt_y0)); gt_box.append(Float32(gt_x1)); gt_box.append(Float32(gt_y1))
                        var iou_val = retina_utils.calcular_iou(pred_box.copy(), gt_box.copy())
                        iou_sum = iou_sum + iou_val
                        count_iou = count_iou + 1
                        # compute out-of-bounds or low-IoU penalty
                        try:
                            var oob = False
                            if px0 < -100.0 or py0 < -100.0 or px1 > Float32(largura) + 100.0 or py1 > Float32(altura) + 100.0:
                                oob = True
                            if oob or iou_val < Float32(0.1):
                                penalty_factor = 10.0
                                penalty_count = penalty_count + 1
                                # extra monitoring loss term (penalize low IoU)
                                soma_loss = soma_loss + penalty_factor * (1.0 - iou_val)
                        except _:
                            pass
                        # selective debug prints when targets are extreme or predicted box is out-of-bounds
                        try:
                            var should_print = False
                            try:
                                if tgt[0] != tgt[0] or tgt[1] != tgt[1] or tgt[2] != tgt[2] or tgt[3] != tgt[3]:
                                    should_print = True
                            except _:
                                pass
                            try:
                                if abs(tgt[0]) > 3.0 or abs(tgt[1]) > 3.0 or abs(tgt[2]) > 3.0 or abs(tgt[3]) > 3.0:
                                    should_print = True
                            except _:
                                pass
                            try:
                                if px0 < -100.0 or py0 < -100.0 or px1 > Float32(largura) + 100.0 or py1 > Float32(altura) + 100.0:
                                    should_print = True
                            except _:
                                pass
                            if should_print:
                                try:
                                    print("[DBG-TRAIN] img", path, "a_idx", a_idx, "aw", a[2], "ah", a[3], "tgt", tgt[0], tgt[1], tgt[2], tgt[3], "pred", px0, py0, px1, py1, "feat_norm_sq", feat_norm_sq, "lr_w", lr_w)
                                except _:
                                    pass
                        except _:
                            pass
                    except _:
                        pass

                    # Use Smooth L1 (Huber) loss for regression to stabilize gradients
                    var beta: Float32 = 1.0
                    for j in range(4):
                        var err = pred[j] - tgt[j]
                        # clamp extreme errors to avoid NaNs
                        if err > 100.0: err = 100.0
                        if err < -100.0: err = -100.0
                        var abs_err = err if err >= 0.0 else -err
                        var loss_j: Float32 = 0.0
                        var grad_factor: Float32 = 0.0
                        if abs_err < beta:
                            loss_j = 0.5 * err * err / beta
                            grad_factor = err / beta
                        else:
                            loss_j = abs_err - 0.5 * beta
                            grad_factor = Float32(1.0) if err > 0.0 else Float32(-1.0)
                        # apply IoU/OOB penalty to gradient magnitude
                        grad_factor = grad_factor * penalty_factor
                        # update weights using grad_factor (derivative w.r.t. prediction)
                        for d in range(D):
                            var grad_w = feats.dados[d] * grad_factor
                            if grad_w > 100.0: grad_w = 100.0
                            if grad_w < -100.0: grad_w = -100.0
                            detector.bloco_cnn.peso_saida.dados[d * 4 + j] = detector.bloco_cnn.peso_saida.dados[d * 4 + j] - lr_w * grad_w
                        detector.bloco_cnn.bias_saida.dados[j] = detector.bloco_cnn.bias_saida.dados[j] - lr_atual * grad_factor
                        soma_loss = soma_loss + loss_j
                    # Cls head: positive anchor → target=1
                    if head_initialized and D > 0:
                        var cls_pred_p: Float32 = 0.0
                        for d_p in range(D):
                            cls_pred_p = cls_pred_p + feats.dados[d_p] * head_peso_cls.dados[d_p]
                        cls_pred_p = cls_pred_p + head_bias_cls.dados[0]
                        # BCE-with-logits gradient: sigmoid(pred) - target (target=1)
                        var grad_p: Float32 = _sigmoid_scalar(cls_pred_p) - 1.0
                        # amplify classification gradient when regression IoU is very low or OOB
                        grad_p = grad_p * penalty_factor
                        if grad_p > 5.0 * penalty_factor: grad_p = 5.0 * penalty_factor
                        if grad_p < -5.0 * penalty_factor: grad_p = -5.0 * penalty_factor
                        for d_p in range(D):
                            var cg_p = feats.dados[d_p] * grad_p
                            if cg_p > 10.0: cg_p = 10.0
                            if cg_p < -10.0: cg_p = -10.0
                            head_peso_cls.dados[d_p] = head_peso_cls.dados[d_p] - lr_w * cg_p
                        head_bias_cls.dados[0] = head_bias_cls.dados[0] - lr_atual * grad_p
                    count_pos = count_pos + 1

        var avg_loss: Float32 = 0.0
        if count_pos > 0:
            avg_loss = soma_loss / Float32(max(1, count_pos * 4))
        print("Epoca", ep, "Avg L1 loss (pos anchors):", avg_loss, "Positives:", count_pos, "LR:", lr_atual)

        # save epoch state
        try:
            var meta_lines = List[String]()
            meta_lines.append("epoch:" + String(ep))
            meta_lines.append("avg_loss:" + String(avg_loss))
            meta_lines.append("positives:" + String(count_pos))
            meta_lines.append("lr:" + String(lr_atual))
            meta_lines.append("best_loss:" + String(best_loss))
            meta_lines.append("scheduler_wait:" + String(scheduler_wait))
            # delegate full workspace save to RetinaFace wrapper (transfer ownership)
            if head_initialized:
                try:
                    detector.cabeca_classificacao_peso = head_peso_cls.copy()
                except _:
                    pass
            try:
                detector.cabeca_classificacao_bias = head_bias_cls.copy()
            except _:
                pass
            detector.treinamento_epoca = ep
            detector.treinamento_lr = lr_atual
            _ = detector.salvar_workspace(export_dir)
            print("Checkpoint salvo: epoca", ep)
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
                    var imgs = class_images[c].copy()
                    if len(imgs) == 0:
                        continue
                    # pick validation image varying by epoch and class (not always imgs[0])
                    var val_seed = (ep * 1664525 + c * 1013904223 + 0xdeadbeef) & 0x7fffffff
                    val_seed = (1103515245 * val_seed + 12345) & 0x7fffffff
                    var val_idx = val_seed % len(imgs)
                    var img_path = imgs[val_idx]
                    # Load original first to get real image dimensions for box coordinate scaling
                    var samp_orig_w: Int = largura; var samp_orig_h: Int = altura
                    try:
                        var samp_orig = dados_pkg.carregar_bmp_rgb(img_path)
                        if samp_orig.width > 0:
                            samp_orig_w = samp_orig.width; samp_orig_h = samp_orig.height
                    except _:
                        pass
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
                            # Pixel coords in original space → scale to training resolution
                            var ssx = Float32(largura) / Float32(max(1, samp_orig_w))
                            var ssy = Float32(altura)  / Float32(max(1, samp_orig_h))
                            s_gt_x0 = Int(s_tx0 * ssx); s_gt_y0 = Int(s_ty0 * ssy)
                            s_gt_x1 = Int(s_tx1 * ssx); s_gt_y1 = Int(s_ty1 * ssy)
                        else:
                            s_gt_x0 = Int(s_tx0 * Float32(largura))
                            s_gt_y0 = Int(s_ty0 * Float32(altura))
                            s_gt_x1 = Int(s_tx1 * Float32(largura))
                            s_gt_y1 = Int(s_ty1 * Float32(altura))
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
                        try:
                            if len(raw_w_bytes) > 0 and len(detector.cabeca_classificacao_peso.formato) >= 1:
                                _ = detector.cabeca_classificacao_peso.carregar_dados_bytes_bin(raw_w_bytes.copy())
                            if len(raw_b_bytes) > 0 and len(detector.cabeca_classificacao_bias.formato) >= 1:
                                _ = detector.cabeca_classificacao_bias.carregar_dados_bytes_bin(raw_b_bytes.copy())
                        except _:
                            pass
                        boxes = detector.inferir(bmp.pixels.copy(), largura, 1)
                    except _:
                        boxes = List[List[Int]]()

                    if len(boxes) == 0:
                        print("Epoch sample", ep, "class", class_names[c], "NO_PRED_BOX gt_box", s_gt_x0, s_gt_y0, s_gt_x1, s_gt_y1)
                    else:
                        var pb = boxes[0].copy()
                        print("Epoch sample", ep, "class", class_names[c], "pred_box", pb[0], pb[1], pb[2], pb[3], "gt_box", s_gt_x0, s_gt_y0, s_gt_x1, s_gt_y1)

                    # write sample file
                    try:
                        var sample_path = os.path.join(export_dir, "epoch_samples", "epoch_" + String(ep) + "_class_" + class_names[c] + ".txt")
                        var lines_out = List[String]()
                        if len(boxes) > 0:
                            var pb2 = boxes[0].copy()
                            lines_out.append("pred_box: " + String(pb2[0]) + " " + String(pb2[1]) + " " + String(pb2[2]) + " " + String(pb2[3]))
                        else:
                            lines_out.append("pred_box: none")
                        lines_out.append("gt_box: " + String(s_gt_x0) + " " + String(s_gt_y0) + " " + String(s_gt_x1) + " " + String(s_gt_y1))
                        _ = uteis.gravar_texto_seguro(sample_path, String("\n").join(lines_out^))
                    except _:
                        pass

                    # save visual BMP overlay: GT=verde, pred=azul
                    try:
                        var flat_rgb = List[Int]()
                        for y in range(bmp.height):
                            for x in range(bmp.width):
                                var rv = Int(bmp.pixels[y][x][0] * 255.0)
                                var gv = Int(bmp.pixels[y][x][1] * 255.0)
                                var bv = Int(bmp.pixels[y][x][2] * 255.0)
                                if rv < 0: rv = 0
                                if rv > 255: rv = 255
                                if gv < 0: gv = 0
                                if gv > 255: gv = 255
                                if bv < 0: bv = 0
                                if bv > 255: bv = 255
                                flat_rgb.append(rv)
                                flat_rgb.append(gv)
                                flat_rgb.append(bv)
                        # GT box em verde (somente se parsou)
                        if s_parsed and s_gt_x0 < s_gt_x1 and s_gt_y0 < s_gt_y1:
                            var gt_coords = List[Int]()
                            gt_coords.append(s_gt_x0); gt_coords.append(s_gt_y0)
                            gt_coords.append(s_gt_x1); gt_coords.append(s_gt_y1)
                            graficos_pkg.draw_bbox_on_flat_rgb(flat_rgb, bmp.width, bmp.height, gt_coords^, 0, 220, 0)
                        # pred box em azul
                        if len(boxes) > 0:
                            var pred_coords = boxes[0].copy()
                            graficos_pkg.draw_bbox_on_flat_rgb(flat_rgb, bmp.width, bmp.height, pred_coords^, 0, 0, 220)
                        var bmp_out_path = os.path.join(export_dir, "epoch_samples", "epoch_" + String(ep) + "_class_" + class_names[c] + ".bmp")
                        var bmp_bytes = graficos_pkg.bmp.gerar_bmp_24bits_de_rgb(flat_rgb^, bmp.width, bmp.height)
                        dados_pkg.gravar_arquivo_binario(bmp_out_path, bmp_bytes^)
                        print("[VIS] epoch", ep, "class", class_names[c], "->", bmp_out_path)
                    except _:
                        pass
                except _:
                    pass
        except _:
            pass

        # Evaluate IoU-based convergence and early stopping using validation set
        try:
            # We'll compute validation IoU by running inference on a small validation subset
            var val_iou_sum: Float32 = 0.0
            var val_iou_count: Int = 0
            var val_best_list: List[Float32] = List[Float32]()
            var val_fp_count: Int = 0
            try:
                var val_root = os.path.join(dataset_dir, "val")
                if not os.path.exists(val_root):
                    val_root = os.path.join(dataset_dir, "train")
                if os.path.exists(val_root):
                    var v_printed = 0
                    var v_max = 200
                    for cls in os.listdir(val_root):
                        var pcls = os.path.join(val_root, cls)
                        if not os.path.isdir(pcls):
                            continue
                        for f in os.listdir(pcls):
                            if v_printed >= v_max:
                                break
                            if not f.endswith('.bmp'):
                                continue
                            var img_path = os.path.join(pcls, f)
                            try:
                                var val_orig_w: Int = largura; var val_orig_h: Int = altura
                                try:
                                    var val_orig = dados_pkg.carregar_bmp_rgb(img_path)
                                    if val_orig.width > 0:
                                        val_orig_w = val_orig.width; val_orig_h = val_orig.height
                                except _:
                                    pass
                                var bmp = dados_pkg.carregar_bmp_rgb(img_path, largura, altura)
                                if bmp.width == 0:
                                    continue

                                # ground-truth box (if exists)
                                var gt_x0: Int = 0; var gt_y0: Int = 0; var gt_x1: Int = 0; var gt_y1: Int = 0
                                var has_gt: Bool = False
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
                                                var vsx = Float32(largura) / Float32(max(1, val_orig_w))
                                                var vsy = Float32(altura)  / Float32(max(1, val_orig_h))
                                                gt_x0 = Int(tx0 * vsx); gt_y0 = Int(ty0 * vsy)
                                                gt_x1 = Int(tx1 * vsx); gt_y1 = Int(ty1 * vsy)
                                            else:
                                                gt_x0 = Int(tx0 * Float32(largura))
                                                gt_y0 = Int(ty0 * Float32(altura))
                                                gt_x1 = Int(tx1 * Float32(largura))
                                                gt_y1 = Int(ty1 * Float32(altura))
                                            has_gt = True
                                except _:
                                    pass

                                var raw_w_bytes = List[Int]()
                                var raw_b_bytes = List[Int]()
                                if head_initialized:
                                    try:
                                        raw_w_bytes = head_peso_cls.dados_bytes_bin()
                                    except _:
                                        raw_w_bytes = List[Int]()
                                    try:
                                        raw_b_bytes = head_bias_cls.dados_bytes_bin()
                                    except _:
                                        raw_b_bytes = List[Int]()
                                var boxes = List[List[Int]]()
                                try:
                                    try:
                                        if len(raw_w_bytes) > 0 and len(detector.cabeca_classificacao_peso.formato) >= 1:
                                            _ = detector.cabeca_classificacao_peso.carregar_dados_bytes_bin(raw_w_bytes.copy())
                                        if len(raw_b_bytes) > 0 and len(detector.cabeca_classificacao_bias.formato) >= 1:
                                            _ = detector.cabeca_classificacao_bias.carregar_dados_bytes_bin(raw_b_bytes.copy())
                                    except _:
                                        pass
                                    boxes = detector.inferir(bmp.pixels.copy(), largura, 16)
                                except _:
                                    boxes = List[List[Int]]()

                                if has_gt:
                                    var best_iou_img: Float32 = 0.0
                                    for bi in range(len(boxes)):
                                        try:
                                            var pb = boxes[bi].copy()
                                            var predf = List[Float32](); predf.append(Float32(pb[0])); predf.append(Float32(pb[1])); predf.append(Float32(pb[2])); predf.append(Float32(pb[3]))
                                            var gtf = List[Float32](); gtf.append(Float32(gt_x0)); gtf.append(Float32(gt_y0)); gtf.append(Float32(gt_x1)); gtf.append(Float32(gt_y1))
                                            var iou_val = retina_utils.calcular_iou(predf.copy(), gtf.copy())
                                            if iou_val > best_iou_img:
                                                best_iou_img = iou_val
                                        except _:
                                            pass
                                    # count as false positive if best_iou very low
                                    if best_iou_img < Float32(0.2):
                                        val_fp_count = val_fp_count + 1
                                    val_best_list.append(best_iou_img)
                                    val_iou_sum = val_iou_sum + best_iou_img
                                    val_iou_count = val_iou_count + 1
                                v_printed = v_printed + 1
                            except _:
                                pass
                        if v_printed >= v_max:
                            break
            except _:
                pass

            var mean_val_iou: Float32 = 0.0
            var median_val_iou: Float32 = 0.0
            if val_iou_count > 0:
                mean_val_iou = val_iou_sum / Float32(val_iou_count)
                # compute median
                try:
                    # simple selection sort + median (List.sort() unavailable)
                    var tmp = val_best_list.copy()
                    for i in range(len(tmp)):
                        var minj = i
                        for j in range(i + 1, len(tmp)):
                            if tmp[j] < tmp[minj]:
                                minj = j
                        if minj != i:
                            var ttmp = tmp[i]
                            tmp[i] = tmp[minj]
                            tmp[minj] = ttmp
                    var mpos = len(tmp) // 2
                    if len(tmp) % 2 == 1:
                        median_val_iou = tmp[mpos]
                    else:
                        median_val_iou = (tmp[mpos - 1] + tmp[mpos]) / Float32(2.0)
                except _:
                    median_val_iou = mean_val_iou
            print("Epoca", ep, "Validation Mean IoU:", mean_val_iou, "median:", median_val_iou, "FP_count:", val_fp_count, "(based on", val_iou_count, "images)")

            # improvement: use composite validation score (mean IoU minus normalized FP penalty)
            var composite_val_score: Float32 = mean_val_iou
            try:
                var denom = Float32(1.0)
                if val_iou_count > 0:
                    denom = Float32(val_iou_count)
                composite_val_score = mean_val_iou - (Float32(val_fp_count) / denom) * val_fp_penalty
            except _:
                composite_val_score = mean_val_iou
            print("Epoca", ep, "Composite Val Score:", composite_val_score)

            if composite_val_score > best_val_score + iou_min_delta:
                best_val_score = composite_val_score
                # also keep track of best mean IoU for logging
                best_iou = mean_val_iou
                iou_patience_count = 0
                iou_consec_count = iou_consec_count + 1
            else:
                iou_patience_count = iou_patience_count + 1
                iou_consec_count = 0

            # Check target consecutive condition using composite score (stricter)
            if early_stop and best_val_score >= iou_target and iou_consec_count >= iou_consec_required:
                print("Early stopping: reached composite IoU target", best_val_score, "at epoch", ep)
                break

            # Patience-based stop (validation)
            if early_stop and iou_patience_count >= iou_patience:
                print("Early stopping: no validation IoU improvement for", iou_patience, "epochs. Best IoU:", best_iou)
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
                                # Load original to get image dimensions for box coordinate scaling
                                var val_orig_w: Int = largura; var val_orig_h: Int = altura
                                try:
                                    var val_orig = dados_pkg.carregar_bmp_rgb(img_path)
                                    if val_orig.width > 0:
                                        val_orig_w = val_orig.width; val_orig_h = val_orig.height
                                except _:
                                    pass
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
                                                var vsx = Float32(largura) / Float32(max(1, val_orig_w))
                                                var vsy = Float32(altura)  / Float32(max(1, val_orig_h))
                                                gt_x0 = Int(tx0 * vsx); gt_y0 = Int(ty0 * vsy)
                                                gt_x1 = Int(tx1 * vsx); gt_y1 = Int(ty1 * vsy)
                                            else:
                                                gt_x0 = Int(tx0 * Float32(largura))
                                                gt_y0 = Int(ty0 * Float32(altura))
                                                gt_x1 = Int(tx1 * Float32(largura))
                                                gt_y1 = Int(ty1 * Float32(altura))
                                except _:
                                    pass

                                var raw_w_bytes = List[Int]()
                                var raw_b_bytes = List[Int]()
                                if head_initialized:
                                    try:
                                        raw_w_bytes = head_peso_cls.dados_bytes_bin()
                                    except _:
                                        raw_w_bytes = List[Int]()
                                    try:
                                        raw_b_bytes = head_bias_cls.dados_bytes_bin()
                                    except _:
                                        raw_b_bytes = List[Int]()
                                var boxes = List[List[Int]]()
                                try:
                                    try:
                                        if len(raw_w_bytes) > 0 and len(detector.cabeca_classificacao_peso.formato) >= 1:
                                            _ = detector.cabeca_classificacao_peso.carregar_dados_bytes_bin(raw_w_bytes.copy())
                                        if len(raw_b_bytes) > 0 and len(detector.cabeca_classificacao_bias.formato) >= 1:
                                            _ = detector.cabeca_classificacao_bias.carregar_dados_bytes_bin(raw_b_bytes.copy())
                                    except _:
                                        pass
                                    boxes = detector.inferir(bmp.pixels.copy(), largura, 16)
                                except _:
                                    boxes = List[List[Int]]()

                                # print detailed per-image diagnostics (only during validation)
                                if len(boxes) == 0:
                                    print("Img", img_path, "pred_boxes", 0)
                                else:
                                    for bi in range(len(boxes)):
                                        var pb = boxes[bi].copy()
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
        try:
            detector.cabeca_classificacao_peso = head_peso_cls^
        except _:
            pass
        try:
            detector.cabeca_classificacao_bias = head_bias_cls^
        except _:
            pass
        _ = detector.salvar_workspace(export_dir)
    except _:
        pass

    return "Treino finalizado"
