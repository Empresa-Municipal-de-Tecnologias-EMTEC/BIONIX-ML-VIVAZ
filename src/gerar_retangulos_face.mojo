# Gera arquivos .box com retângulos heurísticos de face para cada imagem BMP
# Salva um arquivo com mesmo nome da imagem e extensão .box contendo: x0 y0 x1 y1

import os
import config as cfg
import adaptadores.detectar_face as detector
import bionix_ml.dados as dados_pkg
import bionix_ml.uteis.arquivo as arquivo_io
import bionix_ml.dados.bmp as bmpmod


def _is_bmp(nome: String) -> Bool:
    var lower = nome.lower()
    return lower.endswith('.bmp')


def _processar_diretorio(diretorio: String, contador_max: Int = -1) -> Int:
    var nomes = os.listdir(diretorio)
    # `os.listdir` retorna uma lista; ordenação explícita não disponível,
    # manter a ordem retornada pelo sistema para simplicidade.
    var cont = 0
    for nome in nomes:
        var caminho = os.path.join(diretorio, nome)
        try:
            if os.path.isfile(caminho) and _is_bmp(nome):
                # extrai nome base sem extensão (implementação simples)
                var base = nome
                var last_dot = -1
                for i in range(len(nome)-1, -1, -1):
                    if nome[i] == '.':
                        last_dot = i
                        break
                if last_dot >= 0:
                    var btmp = String("")
                    for i2 in range(0, last_dot):
                        btmp = btmp + nome[i2]
                    base = btmp
                var box_path = os.path.join(diretorio, base + '.box')
                if os.path.exists(box_path):
                    # já existe — pular
                    continue
                var cropped = bmpmod.zero_bmp()
                var bbox = List[Int]()
                try:
                    var res = detector.detect_and_align_bbox(caminho)
                    # avoid moving the BMPInfo (res[0]) — only use bbox
                    bbox = res[1]
                except Exception:
                    cropped = bmpmod.zero_bmp()
                    bbox = List[Int]()
                if len(bbox) == 0:
                    # falha: gravar caixa inteira
                    try:
                        var info = bionix_ml.dados.carregar_bmp_rgb(caminho)
                        bbox = [0, 0, info.width, info.height]
                    except Exception:
                        bbox = [0, 0, 0, 0]
                var texto = String(bbox[0]) + " " + String(bbox[1]) + " " + String(bbox[2]) + " " + String(bbox[3])
                _ = arquivo_io.gravar_texto_seguro(box_path, texto)
                cont += 1
                if contador_max > 0 and cont >= contador_max:
                    return cont

            elif os.path.isdir(caminho):
                var sub = _processar_diretorio(caminho, contador_max)
                if contador_max > 0 and sub >= contador_max:
                    return sub
        except Exception:
            print("[gerar_retangulos_face] falha processando " + String(caminho))
    return cont


fn main():
    if not cfg.GERAR_RETANGULOS_FACE:
        print("[gerar_retangulos_face] GERAR_RETANGULOS_FACE está desabilitado em config; nada a fazer.")
        return

    var root = cfg.DATASET_ROOT
    print("[gerar_retangulos_face] iniciando em " + String(root))
    var total = Int(0)
    try:
        total = _processar_diretorio(root)
    except Exception:
        print("[gerar_retangulos_face] falha ao processar diretório principal")
        total = 0
    print("[gerar_retangulos_face] finalizado: gerados " + String(total) + " arquivos .box")
