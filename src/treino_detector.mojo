# Ponto de entrada de treino (Mojo) - produção POC
# Uso: mojo -I ../../BIONIX-ML src/treino.mojo


import config as cfg
import io_modelo as io
import gerar_retangulos_face as gerador_boxes
import detector_dataset as dataset_pkg
import detector_model as model_pkg
import bionix_ml.computacao as computacao_pkg
import os


def main():
    print("Treino do detector — pipeline integrado (CPU/CUDA plugável)")

    # gerar pseudo-labels (.box) se configurado
    if cfg.GERAR_RETANGULOS_FACE:
        print("Gerador de retângulos ativado — processando dataset...")
        gerador_boxes.main()
        print("Processamento de .box finalizado")

    # backend selection: prefer CUDA if available, fallback to CPU
    var tipo = computacao_pkg.backend_nome_de_id(computacao_pkg.backend_cpu_id())

    var altura = 100
    var largura = 64

    print("Carregando dataset detector (crops " + String(largura) + "x" + String(altura) + ")...")
    var x_det, y_det = dataset_pkg.carregar_dataset_detector_pouro(cfg.DATASET_ROOT + "/treino", altura, largura, tipo)
    if len(x_det.formato) == 0 or x_det.formato[0] == 0:
        print("Falha ao construir dataset detector. Verifique arquivos e .txt de bboxes.")
        return

    print("Amostras detector:", x_det.formato[0], "| Features:", x_det.formato[1])

    # criar contexto com escolha de backend (device_kind)
    import bionix_ml.computacao.adaptadores.contexto as contexto_defs
    var device_kind = tipo
    var ctx = contexto_defs.criar_contexto_padrao(device_kind)

    mut bloco = model_pkg.criar_bloco_detector(altura, largura, 6, 3, 3, ctx)
    # ajustar tipo de computacao conforme detecção de backend
    bloco.tipo_computacao = tipo
    # tentar carregar checkpoint se existir
    var carregado = model_pkg.carregar_checkpoint(bloco, cfg.MODEL_DIR)
    if carregado:
        print("Checkpoint carregado de", cfg.MODEL_DIR)
    else:
        print("Nenhum checkpoint encontrado — treinando do zero")
    print("Treinando detector com BCEWithLogits (PoC)")
    try:
        var loss = model_pkg.treinar_detector_bce(bloco, x_det, y_det, 0.01, cfg.EPOCHS, 10)
        print("Treino finalizado. Loss final:", loss)
    except _:
        print("Erro durante o treino do detector.")

    # salvar metadados simples
    var model_dir = cfg.MODEL_DIR
    var meta_path = model_dir + "/metadata_detector.txt"
    io._ensure_parent_dir(meta_path)
    var mf = open(meta_path, 'w')
    mf.write("model: vivaz-detector-poc\n")
    mf.write("input_h: " + String(altura) + "\n")
    mf.write("input_w: " + String(largura) + "\n")
    mf.close()

    # salvar checkpoint final
    var ok = model_pkg.salvar_checkpoint(bloco, cfg.MODEL_DIR)
    if ok:
        print("Checkpoint salvo em", cfg.MODEL_DIR)
    else:
        print("Falha ao salvar checkpoint em", cfg.MODEL_DIR)

    print("Treino detector concluído.")
