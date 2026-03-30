import bionix_ml.camadas.cnn as cnn_pkg
import bionix_ml.computacao.dispatcher_tensor as dispatcher
import bionix_ml.nucleo.Tensor as tensor_defs
import bionix_ml.dados as dados_pkg
import bionix_ml.graficos as graficos_pkg
import math
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import bionix_ml.autograd.tipos_mlp as tipos_mlp
import io_modelo
import os
import bionix_ml.uteis.arquivo as arquivo_io
import bionix_ml.computacao.sessao as sessao_driver
import bionix_ml.computacao.storage_sessao as storage_sessao
import bionix_ml.camadas.cnn.cnn as cnn_impl

# New detector implementation using color images and framework binary export

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
    print("[DEBUG] criar_bloco_detector: parametros -> altura:", altura, "largura:", largura, "num_filtros:", num_filtros, "kernel:", kernel_h, "x", kernel_w)
    var bloco = cnn_pkg.BlocoCNN(altura, largura, num_filtros, kernel_h, kernel_w, contexto.copy(), ativacao_id, perda_id, tipo)
    print("[DEBUG] criar_bloco_detector: BlocoCNN instanciado")
    return bloco^


fn _clamp01(x: Float32) -> Float32:
    var y = x
    if y < 0.0:
        y = 0.0
    if y > 1.0:
        y = 1.0
    return y


# Train a simple linear bbox head on top of cnn features using color inputs.
# entradas: tensor [N, H*W*3] with values in [0,1]
# alvos: tensor [N,4] normalized to [0,1] -> [x0,y0,x1,y1]
fn treinar_detector_color_com_saida(
    mut bloco: cnn_pkg.BlocoCNN,
    entradas: tensor_defs.Tensor,
    alvos: tensor_defs.Tensor,
    var taxa_aprendizado: Float32 = 0.01,
    var epocas: Int = 100,
    var imprimir_cada: Int = 10,
    var model_dir: String = "",
    var dataset_root: String = "",
    var tolerancia: Float32 = 0.1,
    var early_stop: Bool = False,
) raises -> (Float32, List[Int], String):
    debug_assert(len(alvos.formato) == 2 and alvos.formato[1] == 4, "alvos deve ser [N,4]")
    debug_assert(entradas.formato[0] == alvos.formato[0], "N de entradas e alvos deve bater")

    var loss_final: Float32 = 0.0
    var ultima_pred: List[Int] = List[Int]()
    var ultima_img: String = ""

    var head_initialized = False

    for epoca in range(epocas):
        # Extract features using the bloco (expects entrada shape [N, H*W*3])
        var feats = tensor_defs.Tensor(List[Int](), bloco.tipo_computacao)
        try:
            feats = cnn_pkg.extrair_features(bloco, entradas)
        except _:
            print("Erro ao extrair features; pulando epoca", epoca)
            continue

        if not head_initialized:
            var feat_dim = feats.formato[1]
            var shape_w = List[Int]()
            shape_w.append(feat_dim); shape_w.append(4)
            bloco.peso_saida = tensor_defs.Tensor(shape_w^, bloco.tipo_computacao)
            var shape_b = List[Int]()
            shape_b.append(1); shape_b.append(4)
            bloco.bias_saida = tensor_defs.Tensor(shape_b^, bloco.tipo_computacao)
            for i in range(len(bloco.peso_saida.dados)):
                bloco.peso_saida.dados[i] = 0.001
            for j in range(len(bloco.bias_saida.dados)):
                bloco.bias_saida.dados[j] = 0.0
            head_initialized = True

        var preds = dispatcher.multiplicar_matrizes(feats, bloco.peso_saida)
        preds = dispatcher.adicionar_bias_coluna(preds, bloco.bias_saida)

        var loss = dispatcher.erro_quadratico_medio_escalar(preds, alvos)
        var grad_pred = dispatcher.gradiente_mse(preds, alvos)

        var ft = dispatcher.transpor(feats)
        var grad_w = dispatcher.multiplicar_matrizes(ft, grad_pred)

        var cols = grad_pred.formato[1]
        var rows = grad_pred.formato[0]
        var grad_b_vals = List[Float32]()
        for j in range(cols):
            grad_b_vals.append(0.0)
        for i in range(rows):
            for j in range(cols):
                grad_b_vals[j] = grad_b_vals[j] + grad_pred.dados[i * cols + j]

        for i in range(len(bloco.peso_saida.dados)):
            bloco.peso_saida.dados[i] = bloco.peso_saida.dados[i] - taxa_aprendizado * grad_w.dados[i]
        for j in range(cols):
            bloco.bias_saida.dados[j] = bloco.bias_saida.dados[j] - taxa_aprendizado * grad_b_vals[j]

        loss_final = loss

        if imprimir_cada > 0 and (epoca % imprimir_cada == 0 or epoca == epocas - 1):
            print("Epoca", epoca, "| L1-like MSE loss:", loss_final)
            # debug: print normalized target and raw preds for first sample
            try:
                if entradas.formato[0] > 0 and alvos.formato[0] > 0 and len(preds.dados) >= 4:
                    var sample_target = List[Float32]()
                    for k in range(4):
                        sample_target.append(alvos.dados[0 * 4 + k])
                    var sample_pred = List[Float32]()
                    for k in range(4):
                        sample_pred.append(preds.dados[k])
                    print("[DEBUG] alvo_normalizado:", sample_target, "preds_raw:", sample_pred)
            except _:
                pass

        # Export head weights/bias as binary using framework helper (dados_pkg)
        if len(model_dir) > 0:
            try:
                if not os.path.isdir(model_dir):
                    os.makedirs(model_dir)
                # save binary weights and bias
                var wb_path = os.path.join(model_dir, "bbox_head_weights.bin")
                var bb_path = os.path.join(model_dir, "bbox_head_bias.bin")
                var ok_w = dados_pkg.gravar_arquivo_binario(wb_path, bloco.peso_saida.dados_bytes_bin())
                var ok_b = dados_pkg.gravar_arquivo_binario(bb_path, bloco.bias_saida.dados_bytes_bin())
                if not ok_w or not ok_b:
                    print("[ERRO] Falha ao exportar pesos binários em", model_dir)

                # save metadata referencing binary files
                var meta = "feat_shape=" + String(bloco.peso_saida.formato[0]) + "," + String(bloco.peso_saida.formato[1]) + "\n"
                meta = meta + "weights_bin=bbox_head_weights.bin\n"
                meta = meta + "bias_bin=bbox_head_bias.bin\n"
                _ = io_modelo.save_metadata(os.path.join(model_dir, "metadata_detector.txt"), meta)
            except _:
                print("[ERRO] Exceção ao salvar metadados na época", epoca)

            # Save full model checkpoint using framework storage (so model can be fully restored)
            try:
                _ = salvar_checkpoint(bloco, model_dir)
            except _:
                print("[AVISO] falha ao salvar checkpoint completo em", model_dir)

        # Optional: produce a visual validation image using framework draw util
        try:
            if len(dataset_root) > 0:
                # look for first candidate with .bmp and .txt
                var found = False
                var candidate_path = ""
                var orig_info = dados_pkg.carregar_bmp_rgb("")
                try:
                    for ident in os.listdir(dataset_root):
                        var p_ident = os.path.join(dataset_root, ident)
                        if not os.path.isdir(p_ident):
                            continue
                        for f in os.listdir(p_ident):
                            if not f.endswith('.bmp'):
                                continue
                            var candidate = os.path.join(p_ident, f)
                            try:
                                orig_info = dados_pkg.carregar_bmp_rgb(candidate)^
                                candidate_path = candidate
                                found = True
                                break
                            except _:
                                continue
                        if found:
                            break
                except _:
                    found = False

                if found and orig_info.width > 0 and orig_info.height > 0:
                    # build prediction on resized color image
                    var resized = graficos_pkg.bmp.redimensionar_matriz_rgb_nearest_from_flat(orig_info.flat_pixels, orig_info.width, orig_info.height, orig_info.channels, bloco.altura, bloco.largura)
                    var flat = List[Float32]()
                    for ry in range(len(resized)):
                        for rx in range(len(resized[0])):
                            var pix = resized[ry][rx]
                            # pix is [r,g,b] normalized 0..1
                            flat.append(pix[0]); flat.append(pix[1]); flat.append(pix[2])
                    var in_shape = List[Int](); in_shape.append(1); in_shape.append(bloco.altura * bloco.largura * 3)
                    var input_tensor = tensor_defs.Tensor(in_shape^, bloco.tipo_computacao)
                    for i in range(len(flat)):
                        input_tensor.dados[i] = flat[i]

                    var feats_single = cnn_pkg.extrair_features(bloco, input_tensor)
                    var preds_single = dispatcher.multiplicar_matrizes(feats_single, bloco.peso_saida)
                    preds_single = dispatcher.adicionar_bias_coluna(preds_single, bloco.bias_saida)
                    if len(preds_single.dados) >= 4:
                        var cx0 = _clamp01(preds_single.dados[0])
                        var cy0 = _clamp01(preds_single.dados[1])
                        var cx1 = _clamp01(preds_single.dados[2])
                        var cy1 = _clamp01(preds_single.dados[3])
                        var px0 = Int(cx0 * Float32(orig_info.width - 1))
                        var py0 = Int(cy0 * Float32(orig_info.height - 1))
                        var px1 = Int(cx1 * Float32(orig_info.width - 1))
                        var py1 = Int(cy1 * Float32(orig_info.height - 1))
                        if px0 < 0: px0 = 0
                        if py0 < 0: py0 = 0
                        if px1 >= orig_info.width: px1 = orig_info.width - 1
                        if py1 >= orig_info.height: py1 = orig_info.height - 1
                        ultima_pred = List[Int]()
                        ultima_pred.append(px0); ultima_pred.append(py0); ultima_pred.append(px1); ultima_pred.append(py1)
                        ultima_img = candidate_path
                        # draw using graficos util
                        var flat_rgb = List[Int]()
                        for ry in range(orig_info.height):
                            for rx in range(orig_info.width):
                                var p = orig_info.pixels[ry][rx]
                                flat_rgb.append(Int(p[0]*255.0)); flat_rgb.append(Int(p[1]*255.0)); flat_rgb.append(Int(p[2]*255.0))
                        graficos_pkg.draw_bbox_on_flat_rgb(flat_rgb, orig_info.width, orig_info.height, ultima_pred, 0, 0, 255)
                        var bmp_bytes = graficos_pkg.bmp.gerar_bmp_24bits_de_rgb(flat_rgb, orig_info.width, orig_info.height)
                        _ = dados_pkg.gravar_arquivo_binario(os.path.join("detector_modelo", "validacao_color.bmp"), bmp_bytes^)
        except _:
            pass

    return (loss_final, ultima_pred, ultima_img)


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
        print("[DEBUG] model_detector.carregar_checkpoint: model_dir=", model_dir)
        if not os.path.isdir(model_dir):
            print("[DEBUG] model_detector.carregar_checkpoint: model_dir does not exist")
            return False
        var driver = sessao_driver.driver_sessao_disco(model_dir)
        var storage = storage_sessao.criar_storage_sessao(driver)
        print("[DEBUG] model_detector.carregar_checkpoint: calling cnn_impl._carregar_bloco_de_storage()")
        cnn_impl._carregar_bloco_de_storage(bloco, storage)
        print("[DEBUG] model_detector.carregar_checkpoint: loaded checkpoint")
        return True
    except _:
        print("[DEBUG] model_detector.carregar_checkpoint: exception during load")
        return False
