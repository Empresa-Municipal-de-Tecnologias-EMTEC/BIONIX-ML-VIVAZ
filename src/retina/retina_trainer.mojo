import bionix_ml.camadas.cnn as cnn_pkg
import retina.model_detector as model_pkg
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
    # transfer local list to caller to avoid implicit copy
    return fields^


# Sanity-check an anchor vector (x, y, w, h). Returns True if anchor looks reasonable.
fn _anchor_sane(var a: List[Float32], var max_width: Int, var max_height: Int) -> Bool:
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
                          var patch_size: Int = 64, var epocas: Int = 5, var taxa_aprendizado: Float32 = 0.0001,
                          var batch_size: Int = 4, var samples_per_class: Int = 1, var randomize: Bool = True) raises -> String:

    print("[DEBUG] treinar_retina_minimal: entrada chamada; altura:", altura, "largura:", largura, "patch_size:", patch_size, "epocas:", epocas)
    try:
        var h = -1; var w = -1
        try:
            h = detector.bloco_cnn.altura
        except _:
            h = -1
        try:
            w = detector.bloco_cnn.largura
        except _:
            w = -1
        print("[DEBUG] treinar_retina_minimal: detector.bloco_cnn dimensions maybe -> altura:", h, "largura:", w)
    except _:
        pass

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
    # TEMPORARY: skip checkpoint load to isolate OOM cause (set to False to enable load)
    var _skip_checkpoint_load = True
    if not _skip_checkpoint_load:
        try:
            print("[DEBUG] treinar_retina_minimal: calling model_pkg.carregar_checkpoint with", export_dir)
            _ = model_pkg.carregar_checkpoint(detector.bloco_cnn, export_dir)
            print("[DEBUG] treinar_retina_minimal: returned from model_pkg.carregar_checkpoint")
        except _:
            pass
    else:
        print("[DEBUG] treinar_retina_minimal: skipping model_pkg.carregar_checkpoint (temp)")
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
                    _ = head_peso_cls.carregar_dados_bytes_bin(raw_w.copy())
                    var shape_b = List[Int]()
                    shape_b.append(1); shape_b.append(4)
                    head_bias_cls = tensor_defs.Tensor(shape_b^, detector.bloco_cnn.tipo_computacao)
                    if len(raw_b) > 0:
                        _ = head_bias_cls.carregar_dados_bytes_bin(raw_b.copy())
                    head_initialized = True
            except _:
                head_initialized = False
    except _:
        pass

    # anchors (fixed for image size)
    print("[DEBUG] treinar_retina_minimal: before gerar_anchors, largura:", largura)
    var anchors = anchor_gen.gerar_anchors(largura)
    print("[DEBUG] treinar_retina_minimal: after gerar_anchors; anchors len:", len(anchors))
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
    print("[DBG] anchors_checksum after generation:", anchors_checksum)
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
                print("[DBG] gerar_anchors_scan: found", bad, "corrupted anchors immediately after generation; sample indices:")
                for idx in bad_examples:
                    try:
                        print(idx)
                    except _:
                        pass
                for idx in bad_examples:
                    try:
                        var aex = anchors[idx].copy()
                        print("[DBG] gerar_anchors_scan: idx", idx, "vals:", aex[0], aex[1], aex[2], aex[3])
                    except _:
                        print("[DBG] gerar_anchors_scan: idx", idx, "(unable to format)")
            except _:
                pass
    except _:
        pass

    # build class lists from dataset_dir (expect structure: dataset_dir/train/<class>/*.bmp)
    print("[DEBUG] treinar_retina_minimal: about to build class lists from dataset_dir:", dataset_dir)
    var train_root = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_root):
        train_root = dataset_dir
    var class_names = List[String]()
    var class_images = List[List[String]]()
    var total_images: Int = 0
    var _limit_dataset_scan = True
    var _limit_total_images: Int = 500
    var _stop_scan: Bool = False
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
            # compute length before transferring ownership into class_images
            var imgs_len = len(imgs)
            class_images.append(imgs^)
            total_images = total_images + imgs_len
            print("[DEBUG] treinar_retina_minimal: class", cls, "images:", imgs_len, "total_images:", total_images)
            if _limit_dataset_scan and total_images > _limit_total_images:
                print("[DEBUG] treinar_retina_minimal: reached total_images limit (", total_images, ") - stopping scan")
                _stop_scan = True
            if _stop_scan:
                break
            print("[DEBUG] treinar_retina_minimal: built class lists; classes:", len(class_names))
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

        # build mini-batch list. Default: sample `samples_per_class` images per class.
        # If `randomize` is True, pick random images per class; otherwise use round-robin pointers.
        var batch_paths = List[String]()
        for c in range(len(class_names)):
            var imgs = class_images[c].copy()
            var nimgs = len(imgs)
            if nimgs == 0:
                continue
            # number of samples to draw for this class
            var to_draw = samples_per_class
            if to_draw <= 0:
                to_draw = 1
            if randomize:
                # simple pseudo-random selection using LCG seeded by epoch and class
                var seed = (ep * 1664525 + c * 1013904223) & 0x7fffffff
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
        print("[DBG] treinar_retina_minimal: built batch_paths len=", E, "classes=", len(class_names), "samples_per_class=", samples_per_class, "randomize=", randomize)
        # diagnostic clamp: process only 1 image per epoch to isolate memory issues
        if E > 1:
            E = 1
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
                # Do not pre-resize the full image (high peak allocations).
                # We'll attempt flat-buffer crops first; only if that fails we'll
                # lazily construct a nested `img_matrix` for the fallback path.
                img_matrix = List[List[List[Float32]]]()
                print("[DEBUG] treinar_retina_minimal: deferred full-image resize; bmp.width=", bmp.width, "bmp.height=", bmp.height, "bmp.channels=", bmp.channels, "flat_len=", len(bmp.flat_pixels))

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

                # call assigner with explicit copies to avoid moving local ownership
                var res = assigner.assignar_anchors(anchors.copy(), gt_list.copy())
                # avoid copying large lists here (reduce peak memory)
                var labels = res.labels.copy()
                var targets = res.targets.copy()

                # Diagnostic: print assigner sizes and a few sample anchors/labels
                try:
                    print("[DBG] assigner: anchors_len=", len(anchors), "labels_len=", len(labels), "targets_len=", len(targets))
                    var max_show = 5
                    var show_n = max_show if max_show < len(anchors) else len(anchors)
                    for k in range(show_n):
                        try:
                            var ar = anchors[k].copy()
                            print("[DBG] anchor[", k, "]:", ar[0], ar[1], ar[2], ar[3])
                        except _:
                            pass
                    var show_lab = 10
                    var show_lab_n = show_lab if show_lab < len(labels) else len(labels)
                    for k in range(show_lab_n):
                        try:
                            print("[DBG] label[", k, "]=", labels[k])
                        except _:
                            pass
                except _:
                    pass

                # Diagnostic: report basic /proc/meminfo lines (available on Linux/WSL)
                try:
                    var memtxt = arquivo_pkg.ler_arquivo_texto("/proc/meminfo")
                    if len(memtxt) > 0:
                        var memlines = memtxt.split("\n")
                        if len(memlines) > 0:
                            print("[DBG] /proc/meminfo:", memlines[0])
                        if len(memlines) > 1:
                            print("[DBG] /proc/meminfo:", memlines[1])
                except _:
                    pass

                # for each positive anchor update small regression head and cls head
                # limit processing of positive anchors per image to avoid peak memory spikes
                var pos_processed: Int = 0
                var _max_pos_per_image: Int = 8

                # allocate input tensor once per image and reuse to avoid repeated large allocations
                var in_shape = List[Int](); in_shape.append(1); in_shape.append(patch_size * patch_size * 3)
                var tensor_in = tensor_defs.Tensor(in_shape^, detector.bloco_cnn.tipo_computacao)

                for a_idx in range(len(anchors)):
                    if labels[a_idx] != 1:
                        continue
                    pos_processed = pos_processed + 1
                    if pos_processed > _max_pos_per_image:
                        # diagnostic - too many positives, skip remaining to reduce memory
                        if pos_processed == _max_pos_per_image + 1:
                            print("[DBG] treinar_retina_minimal: reached max positive anchors for this image; skipping remaining positives")
                        break
                    var a = anchors[a_idx].copy()
                    # defensive validation: skip anchors with nonsensical sizes or corrupted values
                    if not _anchor_sane(a, largura, altura):
                        try:
                            print("[DBG] treinar_retina_minimal: skipping corrupted/invalid anchor a_idx", a_idx, "vals:", a[0], a[1], a[2], a[3])
                        except _:
                            print("[DBG] treinar_retina_minimal: skipping corrupted/invalid anchor a_idx", a_idx, "(unable to print values)")
                        continue
                    var ax = Int(a[0] - a[2] / 2.0); var ay = Int(a[1] - a[3] / 2.0)
                    var aw = Int(a[2]); var ah = Int(a[3])
                    if ax < 0: ax = 0
                    if ay < 0: ay = 0
                    if ax + aw > largura: aw = max(1, largura - ax)
                    if ay + ah > altura: ah = max(1, altura - ay)
                    print("[DBG] anchor positive: a_idx", a_idx, "ax,ay,aw,ah:", ax, ay, aw, ah, "bmp_w,h,channels:", bmp.width, bmp.height, bmp.channels, "flat_len:", len(bmp.flat_pixels))
                    # We no longer construct an intermediate `crop_rgb` patch (huge allocations).
                    # Instead the input tensor is filled directly from the flat BMP buffer below.

                    # Use reference to flat buffer (avoid expensive full-copy)
                    var bmp_flat = bmp.flat_pixels.copy()
                    var bmp_channels = bmp.channels
                    var bmp_w = bmp.width; var bmp_h = bmp.height
                    # compare against snapshot to detect if anchors array was mutated
                    try:
                        var snap_w = anchors_snapshot[a_idx].copy()[2]
                        var cur_w = anchors[a_idx].copy()[2]
                        if (snap_w != snap_w) != (cur_w != cur_w) or snap_w != cur_w:
                            try:
                                var snap = anchors_snapshot[a_idx].copy()
                                var cur = anchors[a_idx].copy()
                                print("[DBG-ERR] treinar_retina_minimal: anchor mutated a_idx", a_idx, "snapshot=", snap[0], snap[1], snap[2], snap[3], "current=", cur[0], cur[1], cur[2], cur[3])
                            except _:
                                print("[DBG-ERR] treinar_retina_minimal: anchor mutated a_idx", a_idx, "(unable to format)")
                    except _:
                        pass
                    # quick scan for NaN/Inf in the flat buffer (diagnostic)
                    try:
                        var nan_count: Int = 0
                        var first_nan_idx: Int = -1
                        for i in range(len(bmp_flat)):
                            var vv = bmp_flat[i]
                            if vv != vv:
                                nan_count = nan_count + 1
                                if first_nan_idx == -1:
                                    first_nan_idx = i
                                if nan_count >= 10:
                                    break
                        if nan_count > 0:
                            print("[DBG-ERR] treinar_retina_minimal: bmp.flat_pixels contains NaN count=", nan_count, "first_idx=", first_nan_idx)
                    except _:
                        pass
                    # Fill tensor directly from BMP flat buffer via nearest-neighbor crop+resize
                    var yy0_local = ay; var xx0_local = ax
                    var src_h_local = ah; var src_w_local = aw
                    if src_h_local <= 0: src_h_local = 1
                    if src_w_local <= 0: src_w_local = 1
                    var dbg_oob_count = 0
                    # sanity-check tensor_in storage size
                    try:
                        if len(tensor_in.dados) < patch_size * patch_size * 3:
                            print("[DBG-ERR] treinar_retina_minimal: tensor_in.dados too small", len(tensor_in.dados), "expected", patch_size * patch_size * 3)
                    except _:
                        pass
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
                            var base = (yy * patch_size + xx) * 3
                            var idx_flat = (img_y * bmp_w + img_x) * bmp_channels
                            if idx_flat + 2 < len(bmp_flat) and idx_flat >= 0:
                                # guard write into tensor_in to avoid corrupting other memory
                                if base + 2 < len(tensor_in.dados) and base >= 0:
                                    tensor_in.dados[base + 0] = bmp_flat[idx_flat + 0]
                                    tensor_in.dados[base + 1] = bmp_flat[idx_flat + 1]
                                    tensor_in.dados[base + 2] = bmp_flat[idx_flat + 2]
                                else:
                                    if dbg_oob_count < 5:
                                        print("[DBG-ERR] treinar_retina_minimal: tensor_in write OOB; base", base, "tensor_len", len(tensor_in.dados))
                                    dbg_oob_count = dbg_oob_count + 1
                            else:
                                # detailed diagnostic print for first few occurrences per anchor
                                if dbg_oob_count < 5:
                                    print("[DBG-ERR] treinar_retina_minimal: flat access OOB; img_x", img_x, "img_y", img_y, "idx_flat", idx_flat, "flat_len", len(bmp_flat), "bmp_w", bmp_w, "bmp_h", bmp_h, "bmp_ch", bmp_channels, "ax,ay,aw,ah", ax, ay, aw, ah, "patch_size", patch_size, "yy,xx", yy, xx)
                                dbg_oob_count = dbg_oob_count + 1
                                if base + 2 < len(tensor_in.dados) and base >= 0:
                                    tensor_in.dados[base + 0] = 0.0
                                    tensor_in.dados[base + 1] = 0.0
                                    tensor_in.dados[base + 2] = 0.0
                            

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
                        var iou_val = retina_utils.calcular_iou(pred_box.copy(), gt_box.copy())
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
                    var imgs = class_images[c].copy()
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
                                _ = detector.cabeca_classificacao_peso.carregar_dados_bytes_bin(raw_w_bytes.copy())
                            if len(raw_b_bytes) > 0 and len(detector.cabeca_classificacao_bias.formato) >= 1:
                                _ = detector.cabeca_classificacao_bias.carregar_dados_bytes_bin(raw_b_bytes.copy())
                        except _:
                            pass
                        boxes = detector.inferir(bmp.pixels.copy(), largura, 1)
                        print("DBG: returned from detector.inferir for epoch sample class", class_names[c])
                    except _:
                        print("DBG: detector.inferir raised for epoch sample class", class_names[c])
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
                                    print("DBG: calling detector.inferir for validation image", img_path)
                                    try:
                                        if len(raw_w_bytes) > 0 and len(detector.cabeca_classificacao_peso.formato) >= 1:
                                            _ = detector.cabeca_classificacao_peso.carregar_dados_bytes_bin(raw_w_bytes.copy())
                                        if len(raw_b_bytes) > 0 and len(detector.cabeca_classificacao_bias.formato) >= 1:
                                            _ = detector.cabeca_classificacao_bias.carregar_dados_bytes_bin(raw_b_bytes.copy())
                                    except _:
                                        pass
                                    boxes = detector.inferir(bmp.pixels.copy(), largura, 16)
                                    print("DBG: returned from detector.inferir for validation image", img_path)
                                except _:
                                    print("DBG: detector.inferir raised for validation image", img_path)
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
