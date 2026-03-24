import bionix_ml.camadas.cnn as cnn_pkg
import bionix_ml.computacao.dispatcher_tensor as dispatcher
import bionix_ml.nucleo.Tensor as tensor_defs
import math
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import bionix_ml.autograd.tipos_mlp as tipos_mlp
import bionix_ml.perdas.bce as bce_perdas
import bionix_ml.computacao.sessao as sessao_driver
import bionix_ml.computacao.storage_sessao as storage_sessao
import os

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
    return cnn_pkg.BlocoCNN(altura, largura, num_filtros, kernel_h, kernel_w, contexto, ativacao_id, perda_id, tipo)


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
) -> Float32:
    # entradas: [N, H*W], alvos: [N,1]
    debug_assert(len(alvos.formato) == 2 and alvos.formato[1] == 1, "alvos deve ser [N,1]")
    debug_assert(entradas.formato[0] == alvos.formato[0], "N de entradas e alvos deve bater")

    var loss_final: Float32 = 0.0
    for epoca in range(epocas):
        var feats = cnn_pkg.extrair_features(bloco, entradas)
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

    return loss_final


fn salvar_checkpoint(mut bloco: cnn_pkg.BlocoCNN, var model_dir: String) -> Bool:
    try:
        var driver = sessao_driver.driver_sessao_disco(model_dir)
        var storage = storage_sessao.criar_storage_sessao(driver)
        # salva tensores usando as rotinas internas do BlocoCNN
        cnn_pkg._salvar_bloco_em_storage(bloco, storage)
        return True
    except _:
        return False


fn carregar_checkpoint(mut bloco: cnn_pkg.BlocoCNN, var model_dir: String) -> Bool:
    try:
        if not os.path.isdir(model_dir):
            return False
        var driver = sessao_driver.driver_sessao_disco(model_dir)
        var storage = storage_sessao.criar_storage_sessao(driver)
        cnn_pkg._carregar_bloco_de_storage(bloco, storage)
        return True
    except _:
        return False
