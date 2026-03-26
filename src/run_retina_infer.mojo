import retina.retina_infer as infer_pkg
import bionix_ml.dados as dados_pkg
import bionix_ml.graficos as graficos_pkg
import os

fn main() raises -> None:
    print("Executando inferência Retina minimal para 10 primeiras classes...")
    var train_root = os.path.join("DATASET", "train")
    if not os.path.exists(train_root):
        print("Pasta de treino não encontrada:", train_root)
        return

    var labels = List[String]()
    try:
        labels = os.listdir(train_root)
    except _:
        print("Falha ao listar classes em", train_root)
        return

    if len(labels) == 0:
        print("Nenhuma classe encontrada em", train_root)
        return

    # processar até 10 primeiras classes
    var model_root = os.path.join("MODELO", "retina_modelo")
    var out_dir = os.path.join(model_root, "validacao")
    try:
        if not os.path.exists(model_root):
            os.mkdir(model_root)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
    except _:
        pass

    var processed = 0
    for lbl in labels:
        if processed >= 10:
            break
        var lbl_path = os.path.join(train_root, lbl)
        if not os.path.isdir(lbl_path):
            continue
        var files = List[String]()
        try:
            files = os.listdir(lbl_path)
        except _:
            continue
        var img_path = String("")
        for f in files:
            if f.endswith('.bmp'):
                img_path = os.path.join(lbl_path, f)
                break
        if img_path == "":
            continue

        # carregar imagem RGB redimensionada para 640x640
        var bmp_info = dados_pkg.carregar_bmp_rgb(img_path, 640, 640)
        if bmp_info.width <= 0:
            print("Falha ao carregar imagem:", img_path)
            continue

        # converter para flat RGB bytes 0..255
        var flat_rgb = List[Int]()
        for y in range(bmp_info.height):
            for x in range(bmp_info.width):
                var px = bmp_info.pixels[y][x]
                var r = Int(px[0] * 255.0)
                var g = Int(px[1] * 255.0)
                var b = Int(px[2] * 255.0)
                if r < 0: r = 0
                if g < 0: g = 0
                if b < 0: b = 0
                if r > 255: r = 255
                if g > 255: g = 255
                if b > 255: b = 255
                flat_rgb.append(r)
                flat_rgb.append(g)
                flat_rgb.append(b)

        # obter predições (usa a rotina existente; pode ser placeholder se pesos ausentes)
        var boxes = infer_pkg.inferir_retina(model_root, 640, 16)

        # desenhar caixas previstas em azul
        for box in boxes:
            graficos_pkg.draw_bbox_on_flat_rgb(mut flat_rgb, bmp_info.width, bmp_info.height, box, 0, 0, 255)

        # salvar arquivo de validação
        var out_name = lbl + "_pred.bmp"
        var out_path = os.path.join(out_dir, out_name)
        var bmp_bytes = graficos_pkg.bmp.gerar_bmp_24bits_de_rgb(flat_rgb, bmp_info.width, bmp_info.height)
        dados_pkg.gravar_arquivo_binario(out_path, bmp_bytes^)
        print("Salvou:", out_path)
        processed = processed + 1

    print("Inferência de validação concluída. Saída:", out_dir)
