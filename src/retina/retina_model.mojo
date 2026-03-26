import model_detector as model_pkg
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import bionix_ml.nucleo.Tensor as tensor_defs
import bionix_ml.dados.arquivo as arquivo_pkg
import bionix_ml.dados as dados_pkg
import bionix_ml.uteis as uteis
import os
import bionix_ml.camadas.cnn as cnn_pkg

# Utility wrappers to create/load/save retina model components in a reusable way

fn criar_bloco_retina(input_size: Int, var num_filtros: Int = 6, var kernel_h: Int = 3, var kernel_w: Int = 3, var tipo_ctx: String = "cpu") -> cnn_pkg.BlocoCNN:
    var ctx = contexto_defs.criar_contexto_padrao(tipo_ctx)
    return model_pkg.criar_bloco_detector(input_size, input_size, num_filtros, kernel_h, kernel_w, ctx)


fn carregar_bloco_retina(mut bloco: cnn_pkg.BlocoCNN, model_dir: String) -> Bool:
    try:
        return model_pkg.carregar_checkpoint(bloco, model_dir)
    except _:
        return False


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


fn carregar_regression_head(mut bloco: cnn_pkg.BlocoCNN, model_dir: String) -> Bool:
    try:
        var raw_w = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "peso_reg.bin"))
        var raw_b = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "bias_reg.bin"))
        if len(raw_w) > 0 and len(bloco.peso_saida.formato) >= 1:
            bloco.peso_saida.carregar_dados_bytes_bin(raw_w)
        if len(raw_b) > 0 and len(bloco.bias_saida.formato) >= 1:
            bloco.bias_saida.carregar_dados_bytes_bin(raw_b)
        return True
    except _:
        return False


fn salvar_estado_modelo(mut bloco: cnn_pkg.BlocoCNN, peso_cls: tensor_defs.Tensor, bias_cls: tensor_defs.Tensor, model_dir: String, meta_lines: List[String]) -> Bool:
    try:
        try:
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
        except _:
            pass

        # save regression head (from bloco)
        try:
            if len(bloco.peso_saida.formato) >= 1:
                var raw_w = bloco.peso_saida.dados_bytes_bin()
                var ok = dados_pkg.gravar_arquivo_binario(os.path.join(model_dir, "peso_reg.bin"), raw_w)
                try:
                    if ok:
                        print("salvar_estado_modelo: saved peso_reg.bin (", len(raw_w), ")")
                    else:
                        print("salvar_estado_modelo: FAILED to save peso_reg.bin")
                except _:
                    pass
        except _:
            pass
        try:
            if len(bloco.bias_saida.formato) >= 1:
                var raw_b = bloco.bias_saida.dados_bytes_bin()
                var ok = dados_pkg.gravar_arquivo_binario(os.path.join(model_dir, "bias_reg.bin"), raw_b)
                try:
                    if ok:
                        print("salvar_estado_modelo: saved bias_reg.bin (", len(raw_b), ")")
                    else:
                        print("salvar_estado_modelo: FAILED to save bias_reg.bin")
                except _:
                    pass
        except _:
            pass

        # save classification head
        try:
            if len(peso_cls.formato) >= 1:
                var data = peso_cls.dados_bytes_bin()
                var ok = dados_pkg.gravar_arquivo_binario(os.path.join(model_dir, "peso_cls.bin"), data)
                try:
                    if ok:
                        print("salvar_estado_modelo: saved peso_cls.bin (", len(data), ")")
                    else:
                        print("salvar_estado_modelo: FAILED to save peso_cls.bin")
                except _:
                    pass
        except _:
            pass
        try:
            if len(bias_cls.formato) >= 1:
                var data_b = bias_cls.dados_bytes_bin()
                var ok = dados_pkg.gravar_arquivo_binario(os.path.join(model_dir, "bias_cls.bin"), data_b)
                try:
                    if ok:
                        print("salvar_estado_modelo: saved bias_cls.bin (", len(data_b), ")")
                    else:
                        print("salvar_estado_modelo: FAILED to save bias_cls.bin")
                except _:
                    pass
        except _:
            pass

        # save metadata lines
        try:
            var meta_path = os.path.join(model_dir, "metadata.txt")
            var okm = uteis.gravar_texto_seguro(meta_path, String("\n").join(meta_lines.copy()^))
            try:
                if okm:
                    print("salvar_estado_modelo: saved metadata.txt")
                else:
                    print("salvar_estado_modelo: FAILED to save metadata.txt")
            except _:
                pass
        except _:
            pass

        # try to save block checkpoint as well
        try:
            var okc = model_pkg.salvar_checkpoint(bloco, model_dir)
            try:
                if okc:
                    print("salvar_estado_modelo: checkpoint saved")
                else:
                    print("salvar_estado_modelo: checkpoint save FAILED")
            except _:
                pass
        except _:
            pass

        # If meta contains an numeric epoch value, create epoched copies of the saved files
        try:
            var epoch_val = String("")
            for L in meta_lines:
                try:
                    if L.startswith("epoch:"):
                        epoch_val = String(L.split(":")[1])
                        break
                except _:
                    continue

            if len(epoch_val) > 0:
                # only create copies for numeric epoch values
                var ok_digit = True
                try:
                    _ = Int(epoch_val)
                except _:
                    ok_digit = False
                if ok_digit:
                    try:
                        var files_to_copy = List[String]()
                        files_to_copy.append("peso_reg.bin")
                        files_to_copy.append("bias_reg.bin")
                        files_to_copy.append("peso_cls.bin")
                        files_to_copy.append("bias_cls.bin")
                        for fname in files_to_copy:
                            try:
                                var src = os.path.join(model_dir, fname)
                                var data = arquivo_pkg.ler_arquivo_binario(src)
                                if len(data) > 0:
                                    var dst = os.path.join(model_dir, fname.replace('.bin', '') + "_epoch_" + epoch_val + ".bin")
                                    var okd = dados_pkg.gravar_arquivo_binario(dst, data)
                                    try:
                                        if okd:
                                            print("salvar_estado_modelo: created epoch copy", dst)
                                        else:
                                            print("salvar_estado_modelo: FAILED to create epoch copy", dst)
                                    except _:
                                        pass
                            except _:
                                pass
                    except _:
                        pass
        except _:
            pass

        return True
    except _:
        return False


fn carregar_metadata(model_dir: String) -> List[String]:
    try:
        var lines = dados_pkg.carregar_txt_linhas(os.path.join(model_dir, "metadata.txt"))
        return lines
    except _:
        return List[String]()


fn inferir_com_bloco(mut bloco: cnn_pkg.BlocoCNN, raw_peso_cls: List[Int], raw_bias_cls: List[Int], img_pixels: List[List[List[Float32]]], input_size: Int = 640, max_per_image: Int = 16) -> List[List[Int]]:
    import retina.retina_anchor_generator as anchor_gen
    import retina.retina_nms as nms_pkg
    import bionix_ml.graficos as graficos_pkg
    import bionix_ml.nucleo.Tensor as tensor_defs_local
    import math

    var anchors = anchor_gen.gerar_anchors(input_size)
    var cls_scores: List[Float32] = List[Float32]()
    var reg_deltas: List[List[Float32]] = List[List[Float32]]()
    var patch_size = 64

    # prepare classification tensors from raw bytes if present
    var peso_cls_tensor = tensor_defs_local.Tensor(List[Int](), bloco.tipo_computacao)
    var bias_cls_tensor = tensor_defs_local.Tensor(List[Int](), bloco.tipo_computacao)
    var cls_tensors_inited = False
    if len(raw_peso_cls) > 0:
        try:
            # assume raw_peso_cls contains D floats (4-byte each)
            var Dguess = 0
            try:
                Dguess = bloco.peso_saida.formato[0]
            except _:
                Dguess = 0
            if Dguess > 0:
                var shape_w = List[Int](); shape_w.append(Dguess); shape_w.append(1)
                peso_cls_tensor = tensor_defs_local.Tensor(shape_w^, bloco.tipo_computacao)
                peso_cls_tensor.carregar_dados_bytes_bin(raw_peso_cls)
                if len(raw_bias_cls) > 0:
                    var shape_b = List[Int](); shape_b.append(1); shape_b.append(1)
                    bias_cls_tensor = tensor_defs_local.Tensor(shape_b^, bloco.tipo_computacao)
                    bias_cls_tensor.carregar_dados_bytes_bin(raw_bias_cls)
                cls_tensors_inited = True
        except _:
            cls_tensors_inited = False

    for i in range(len(anchors)):
        var a = anchors[i]
        var ax = Int(a[0] - a[2] / 2.0); var ay = Int(a[1] - a[3] / 2.0)
        var aw = Int(a[2]); var ah = Int(a[3])
        if ax < 0: ax = 0
        if ay < 0: ay = 0
        if ax + aw > input_size: aw = max(1, input_size - ax)
        if ay + ah > input_size: ah = max(1, input_size - ay)

        # get RGB patch and build input tensor (R,G,B interleaved)
        var patch_rgb = graficos_pkg.crop_and_resize_rgb(img_pixels, ax, ay, ax + aw - 1, ay + ah - 1, patch_size, patch_size)
        var in_shape = List[Int]()
        in_shape.append(1); in_shape.append(patch_size * patch_size * 3)
        var tensor_in = tensor_defs_local.Tensor(in_shape^, bloco.tipo_computacao)
        for yy in range(patch_size):
            for xx in range(patch_size):
                var pix = patch_rgb[yy][xx]
                var base = (yy * patch_size + xx) * 3
                tensor_in.dados[base + 0] = pix[0]
                tensor_in.dados[base + 1] = pix[1]
                tensor_in.dados[base + 2] = pix[2]

        # extract features
        var feats_t = tensor_defs_local.Tensor(List[Int](), bloco.tipo_computacao)
        try:
            feats_t = cnn_pkg.extrair_features(bloco, tensor_in)
        except _:
            # on failure return empty results for this anchor
            feats_t = tensor_defs_local.Tensor(List[Int](), bloco.tipo_computacao)
        var D = 0
        try:
            D = feats_t.formato[1]
        except _:
            D = 0
        var feats = List[Float32]()
        for d in range(D):
            feats.append(feats_t.dados[d])

        # regression prediction (dx,dy,dw,dh)
        var pred = List[Float32]()
        for j in range(4):
            var s: Float32 = 0.0
            for d in range(D):
                s = s + feats[d] * bloco.peso_saida.dados[d * 4 + j]
            s = s + bloco.bias_saida.dados[j]
            pred.append(s)

        # classification score
        var score: Float32 = 0.01
        if cls_tensors_inited and D > 0:
            var logit: Float32 = 0.0
            for d in range(min(D, peso_cls_tensor.formato[0])):
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

    # build boxes and run NMS
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
