import retina.retina_anchor_generator as anchor_gen
import retina.retina_assigner as assigner
import retina.retina_nms as nms_pkg
import bionix_ml.computacao.dispatcher_tensor as dispatcher
import bionix_ml.nucleo.Tensor as tensor_defs
import bionix_ml.dados as dados_pkg
import bionix_ml.dados.arquivo as arquivo_pkg
import os
import bionix_ml.dados.bmp as bmpmod
import math
import model_detector as model_pkg
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import bionix_ml.camadas.cnn as cnn_pkg
import bionix_ml.graficos as graficos_pkg

fn inferir_retina(model_dir: String = "MODELO", input_size: Int = 640, max_per_image: Int = 16) -> List[List[Int]]:
    # Backward-compatible placeholder: prefer image-aware inferir_retina_imagem
    return List[List[Int]]()


fn inferir_retina_imagem(model_dir: String, img_pixels: List[List[List[Float32]]], input_size: Int = 640, max_per_image: Int = 16) -> List[List[Int]]:
    # Create anchors
    var anchors = anchor_gen.gerar_anchors(input_size)

    # build bloco and try to restore checkpoint
    var ctx = contexto_defs.criar_contexto_padrao("cpu")
    var bloco = model_pkg.criar_bloco_detector(input_size, input_size, 6, 3, 3, ctx)
    var loaded = False
    try:
        loaded = model_pkg.carregar_checkpoint(bloco, model_dir)
    except _:
        loaded = False

    if not loaded:
        print("[INFER] checkpoint do bloco não encontrado em", model_dir)
        return List[List[Int]]()

    # helper: convert RGB patch to grayscale and resize
    fn _crop_and_resize_from_rgb(img: List[List[List[Float32]]], x0: Int, y0: Int, x1: Int, y1: Int, out_h: Int, out_w: Int) -> List[List[Float32]]:
        var h = len(img)
        var w = 0
        if h > 0:
            w = len(img[0])
        var xx0 = x0; var yy0 = y0; var xx1 = x1; var yy1 = y1
        if xx0 < 0: xx0 = 0
        if yy0 < 0: yy0 = 0
        if xx1 >= w: xx1 = w - 1
        if yy1 >= h: yy1 = h - 1
        var src_h = yy1 - yy0 + 1
        var src_w = xx1 - xx0 + 1
        if src_h <= 0 or src_w <= 0:
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
                var p = img[yy][xx]
                var gray = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]
                row.append(Float32(gray))
            patch.append(row^)
        var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(patch^, out_h, out_w)
        return resized^

    # attempt to preload classifier head binaries as fallback (trainer exports these)
    var raw_peso_cls = List[Int]()
    var raw_bias_cls = List[Int]()
    try:
        raw_peso_cls = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "peso_cls.bin"))
    except _:
        raw_peso_cls = List[Int]()
    try:
        raw_bias_cls = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "bias_cls.bin"))
    except _:
        raw_bias_cls = List[Int]()

    # per-anchor inference
    var cls_scores: List[Float32] = List[Float32]()
    var reg_deltas: List[List[Float32]] = List[List[Float32]]()
    var patch_size = 64
    var peso_cls_tensor = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)
    var bias_cls_tensor = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)

    for i in range(len(anchors)):
        var a = anchors[i]
        var ax = Int(a[0] - a[2] / 2.0); var ay = Int(a[1] - a[3] / 2.0)
        var aw = Int(a[2]); var ah = Int(a[3])
        if ax < 0: ax = 0
        if ay < 0: ay = 0
        if ax + aw > input_size: aw = max(1, input_size - ax)
        if ay + ah > input_size: ah = max(1, input_size - ay)
        var patch = _crop_and_resize_from_rgb(img_pixels, ax, ay, ax + aw - 1, ay + ah - 1, patch_size, patch_size)

        var in_shape = List[Int]()
        in_shape.append(1); in_shape.append(patch_size * patch_size)
        var tensor_in = tensor_defs.Tensor(in_shape^, bloco.tipo_computacao)
        for yy in range(patch_size):
            for xx in range(patch_size):
                tensor_in.dados[yy * patch_size + xx] = patch[yy][xx]

        var feats = List[Float32]()
        try:
            var feats_t = cnn_pkg.extrair_features(bloco, tensor_in)
            feats = feats_t.dados.copy()^
        except _:
            # fallback zeros
            var D = 0
            if len(bloco.peso_saida.formato) >= 1:
                D = bloco.peso_saida.formato[0]
            for _ in range(D):
                feats.append(0.0)

        # compute bbox preds using bloco.peso_saida and bias
        var D = 0
        if len(feats) > 0:
            D = len(feats)
        var pred = List[Float32]()
        for j in range(4):
            var s: Float32 = 0.0
            for d in range(D):
                s = s + feats[d] * bloco.peso_saida.dados[d * 4 + j]
            s = s + bloco.bias_saida.dados[j]
            pred.append(s)

        import retina.retina_anchor_generator as anchor_gen
        import retina.retina_nms as nms_pkg
        import bionix_ml.nucleo.Tensor as tensor_defs
        import bionix_ml.dados as dados_pkg
        import bionix_ml.dados.arquivo as arquivo_pkg
        import os
        import math
        import model_detector as model_pkg
        import bionix_ml.computacao.adaptadores.contexto as contexto_defs
        import bionix_ml.camadas.cnn as cnn_pkg
        import bionix_ml.graficos as graficos_pkg


        fn treino_usa_bloco() -> Bool:
            return True


        fn carregar_bloco_retina(model_dir: String, input_size: Int = 640) -> cnn_pkg.BlocoCNN:
            var ctx = contexto_defs.criar_contexto_padrao("cpu")
            var bloco = model_pkg.criar_bloco_detector(input_size, input_size, 6, 3, 3, ctx)
            var ok = False
            try:
                ok = model_pkg.carregar_checkpoint(bloco, model_dir)
            except _:
                ok = False
            if not ok:
                print("[INFER] aviso: checkpoint não carregado em", model_dir)
            return bloco


        fn carregar_head_bytes(model_dir: String) -> List[List[Int]]:
            var raw_w = List[Int]()
            var raw_b = List[Int]()
            try:
                raw_w = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "peso_cls.bin"))
            except _:
                raw_w = List[Int]()
            try:
                raw_b = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "bias_cls.bin"))
            except _:
                raw_b = List[Int]()
            var out = List[List[Int]]()
            out.append(raw_w); out.append(raw_b)
            return out


        fn inferir_com_bloco(mut bloco: cnn_pkg.BlocoCNN, raw_peso_cls: List[Int], raw_bias_cls: List[Int], img_pixels: List[List[List[Float32]]], input_size: Int = 640, max_per_image: Int = 16) -> List[List[Int]]:
            var anchors = anchor_gen.gerar_anchors(input_size)

            fn _crop_and_resize_from_rgb(img: List[List[List[Float32]]], x0: Int, y0: Int, x1: Int, y1: Int, out_h: Int, out_w: Int) -> List[List[Float32]]:
                var h = len(img)
                var w = 0
                if h > 0:
                    w = len(img[0])
                var xx0 = x0; var yy0 = y0; var xx1 = x1; var yy1 = y1
                if xx0 < 0: xx0 = 0
                if yy0 < 0: yy0 = 0
                if xx1 >= w: xx1 = w - 1
                if yy1 >= h: yy1 = h - 1
                var src_h = yy1 - yy0 + 1
                var src_w = xx1 - xx0 + 1
                if src_h <= 0 or src_w <= 0:
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
                        var p = img[yy][xx]
                        var gray = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]
                        row.append(Float32(gray))
                    patch.append(row^)
                var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(patch^, out_h, out_w)
                return resized^

            var cls_scores: List[Float32] = List[Float32]()
            var reg_deltas: List[List[Float32]] = List[List[Float32]]()
            var patch_size = 64

            var peso_cls_tensor = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)
            var bias_cls_tensor = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)
            var cls_tensors_inited = False

            for i in range(len(anchors)):
                var a = anchors[i]
                var ax = Int(a[0] - a[2] / 2.0); var ay = Int(a[1] - a[3] / 2.0)
                var aw = Int(a[2]); var ah = Int(a[3])
                if ax < 0: ax = 0
                if ay < 0: ay = 0
                if ax + aw > input_size: aw = max(1, input_size - ax)
                if ay + ah > input_size: ah = max(1, input_size - ay)
                var patch = _crop_and_resize_from_rgb(img_pixels, ax, ay, ax + aw - 1, ay + ah - 1, patch_size, patch_size)

                var in_shape = List[Int]()
                in_shape.append(1); in_shape.append(patch_size * patch_size)
                var tensor_in = tensor_defs.Tensor(in_shape^, bloco.tipo_computacao)
                for yy in range(patch_size):
                    for xx in range(patch_size):
                        tensor_in.dados[yy * patch_size + xx] = patch[yy][xx]

                var feats = List[Float32]()
                try:
                    var feats_t = cnn_pkg.extrair_features(bloco, tensor_in)
                    feats = feats_t.dados.copy()^
                except _:
                    var D0 = 0
                    if len(bloco.peso_saida.formato) >= 1:
                        D0 = bloco.peso_saida.formato[0]
                    for _ in range(D0):
                        feats.append(0.0)

                var D = 0
                if len(feats) > 0:
                    D = len(feats)

                var pred = List[Float32]()
                for j in range(4):
                    var s: Float32 = 0.0
                    for d in range(D):
                        s = s + feats[d] * bloco.peso_saida.dados[d * 4 + j]
                    s = s + bloco.bias_saida.dados[j]
                    pred.append(s)

                if not cls_tensors_inited and len(raw_peso_cls) > 0:
                    try:
                        var cnt_vals = Int(len(raw_peso_cls) / 4)
                        if cnt_vals >= D:
                            var shape_w = List[Int]()
                            shape_w.append(D); shape_w.append(1)
                            peso_cls_tensor = tensor_defs.Tensor(shape_w^, bloco.tipo_computacao)
                            peso_cls_tensor.carregar_dados_bytes_bin(raw_peso_cls)
                            if len(raw_bias_cls) >= 4:
                                var shape_b = List[Int]()
                                shape_b.append(1); shape_b.append(1)
                                bias_cls_tensor = tensor_defs.Tensor(shape_b^, bloco.tipo_computacao)
                                bias_cls_tensor.carregar_dados_bytes_bin(raw_bias_cls)
                            cls_tensors_inited = True
                    except _:
                        cls_tensors_inited = False

                var score: Float32 = 0.01
                if cls_tensors_inited:
                    var logit: Float32 = 0.0
                    for d in range(D):
                        logit = logit + feats[d] * peso_cls_tensor.dados[d]
                    if len(bias_cls_tensor.dados) > 0:
                        logit = logit + bias_cls_tensor.dados[0]
                    if logit > 50.0:
                        score = 1.0
                    elif logit < -50.0:
                        score = 0.0
                    else:
                        score = 1.0 / (1.0 + Float32(math.exp(-Float64(logit))))

                cls_scores.append(score)
                var drow = List[Float32]()
                for v in pred:
                    drow.append(v)
                reg_deltas.append(drow)

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
