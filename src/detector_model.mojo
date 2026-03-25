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

                        # draw rectangle border (2 px) in red
                        var color_r = 255
                        var color_g = 0
                        var color_b = 0
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
                                img_rgb[ti_top + 0] = color_r
                                img_rgb[ti_top + 1] = color_g
                                img_rgb[ti_top + 2] = color_b
                                var ti_bot = ((y1 - t) * W + x) * 3
                                img_rgb[ti_bot + 0] = color_r
                                img_rgb[ti_bot + 1] = color_g
                                img_rgb[ti_bot + 2] = color_b
                            for y in range(y0 + t, y1 - t + 1):
                                var ti_l = (y * W + (x0 + t)) * 3
                                img_rgb[ti_l + 0] = color_r
                                img_rgb[ti_l + 1] = color_g
                                img_rgb[ti_l + 2] = color_b
                                var ti_r = (y * W + (x1 - t)) * 3
                                img_rgb[ti_r + 0] = color_r
                                img_rgb[ti_r + 1] = color_g
                                img_rgb[ti_r + 2] = color_b

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
