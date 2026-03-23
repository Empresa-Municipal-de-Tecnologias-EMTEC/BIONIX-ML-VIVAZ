# Gera arquivos .box com retângulos heurísticos de face para cada imagem BMP
# Salva um arquivo com mesmo nome da imagem e extensão .box contendo: x0 y0 x1 y1

import os
import config as cfg
import adaptadores.detectar_face as detector
import bionix_ml.dados as dados_pkg
import bionix_ml.uteis.arquivo as arquivo_io


def _is_bmp(nome: String) -> Bool:
    var lower = nome.lower()
    return lower.endswith('.bmp')


def _processar_diretorio(diretorio: String, contador_max: Int = -1) -> Int:
    var nomes = os.listdir(diretorio)
    nomes.sort()
    var cont = 0
    for nome in nomes:
        var caminho = os.path.join(diretorio, nome)
        try:
            if os.path.isfile(caminho) and _is_bmp(nome):
                var base = nome.rsplit('.', 1)[0]
                var box_path = os.path.join(diretorio, base + '.box')
                if os.path.exists(box_path):
                    # já existe — pular
                    continue

                var res = detector.detect_and_align_bbox(caminho)
                var cropped = None
                var bbox = None
                if res != None:
                    try:
                        cropped = res[0]
                        bbox = res[1]
                    except Exception:
                        cropped = None
                        bbox = None
                if bbox == None:
                    # falha: gravar caixa inteira
                    try:
                        var info = bionix_ml.dados.carregar_bmp_rgb(caminho)
                        bbox = [0, 0, info.width, info.height]
                    except Exception:
                        bbox = [0, 0, 0, 0]

                var texto = str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3])
                _ = arquivo_io.gravar_texto_seguro(box_path, texto)
                cont += 1
                if contador_max > 0 and cont >= contador_max:
                    return cont

            elif os.path.isdir(caminho):
                var sub = _processar_diretorio(caminho, contador_max)
                if contador_max > 0 and sub >= contador_max:
                    return sub
        except Exception as e:
            print("[gerar_retangulos_face] falha processando " + str(caminho) + ": " + str(e))
    return cont


fn main():
    if not cfg.GERAR_RETANGULOS_FACE:
        print("[gerar_retangulos_face] GERAR_RETANGULOS_FACE está desabilitado em config; nada a fazer.")
        return

    var root = cfg.DATASET_ROOT
    print("[gerar_retangulos_face] iniciando em " + str(root))
    var total = _processar_diretorio(root)
    print("[gerar_retangulos_face] finalizado: gerados " + str(total) + " arquivos .box")
