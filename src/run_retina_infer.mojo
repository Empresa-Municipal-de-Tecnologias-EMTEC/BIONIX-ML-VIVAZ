import retina.retina_infer as infer_pkg
import bionix_ml.dados as dados_pkg
import bionix_ml.graficos as graficos_pkg
import bionix_ml.uteis as uteis
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

    var model_root = os.path.join("MODELO", "retina_modelo")
    var out_dir = os.path.join(model_root, "validacao_inferencia")
    try:
        if not os.path.exists(model_root):
            os.mkdir(model_root)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
    except _:
        pass

    # carregar bloco e cabeça uma vez
    mut bloco_global = infer_pkg.carregar_bloco_retina(model_root, 640)
    var head = infer_pkg.carregar_head_bytes(model_root)
    var raw_peso_cls = head[0]
    var raw_bias_cls = head[1]

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

        print("[RUN] carregando imagem:", img_path)
        var bmp_info = dados_pkg.carregar_bmp_rgb(img_path, 640, 640)
        print("[RUN] carregou imagem -> size:", bmp_info.width, "x", bmp_info.height)
        if bmp_info.width <= 0:
            print("Falha ao carregar imagem:", img_path)
            continue

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

        print("[RUN] executando inferência para label:", lbl)
        var boxes = List[List[Int]]()
        try:
            boxes = infer_pkg.inferir_com_bloco(bloco_global, raw_peso_cls, raw_bias_cls, bmp_info.pixels, 640, 8)
            print("[RUN] inferencia retornou boxes count:", len(boxes))
        except _:
            print("[RUN] exceção durante inferência para:", img_path)
            boxes = List[List[Int]]()

        var flat_pred = flat_rgb.copy()^
        for box in boxes:
            graficos_pkg.draw_bbox_on_flat_rgb(flat_pred, bmp_info.width, bmp_info.height, box, 0, 0, 255)

        var file_box = List[Int]()
        try:
            var txt_path = img_path.replace('.bmp', '.txt')
            if not os.path.exists(txt_path):
                var txt_path_box = img_path.replace('.bmp', '.box')
                if os.path.exists(txt_path_box):
                    txt_path = txt_path_box
            if os.path.exists(txt_path):
                var linhas = dados_pkg.carregar_txt_linhas(txt_path)
                if len(linhas) > 0:
                    var parts = linhas[0].replace("\t", " ").replace(",", " ").split(" ")
                    var campos = List[String]()
                    for p in parts:
                        var ps = p.strip()
                        if len(ps) != 0:
                            campos.append(String(ps))
                    if len(campos) >= 4:
                        var vals = List[Float32]()
                        var mx: Float32 = 0.0
                        for i in range(4):
                            try:
                                var v = Float32(uteis.parse_float_ascii(String(campos[i])))
                                vals.append(v)
                                if v > mx: mx = v
                            except _:
                                vals.append(0.0)
                        var scale_norm = False
                        if mx <= 1.01:
                            scale_norm = True
                        var w = bmp_info.width; var h = bmp_info.height
                        for i in range(4):
                            var vv = vals[i]
                            if scale_norm:
                                if i == 0 or i == 2:
                                    file_box.append(Int(vv * w))
                                else:
                                    file_box.append(Int(vv * h))
                            else:
                                file_box.append(Int(vv))
        except _:
            file_box = List[Int]()

        var out_name = lbl + "_pred.bmp"
        var out_path = os.path.join(out_dir, out_name)
        var bmp_bytes = graficos_pkg.bmp.gerar_bmp_24bits_de_rgb(flat_pred, bmp_info.width, bmp_info.height)
        dados_pkg.gravar_arquivo_binario(out_path, bmp_bytes^)
        print("Salvou:", out_path)

        if len(file_box) >= 4:
            var flat_box = flat_rgb.copy()^
            var fx0 = file_box[0]; var fy0 = file_box[1]; var fx1 = file_box[2]; var fy1 = file_box[3]
            var coords = List[Int]()
            coords.append(fx0); coords.append(fy0); coords.append(fx1); coords.append(fy1)
            graficos_pkg.draw_bbox_on_flat_rgb(flat_box, bmp_info.width, bmp_info.height, coords^, 0, 255, 0)
            var out_name_box = lbl + "_box.bmp"
            var out_path_box = os.path.join(out_dir, out_name_box)
            var bmp_bytes_box = graficos_pkg.bmp.gerar_bmp_24bits_de_rgb(flat_box, bmp_info.width, bmp_info.height)
            dados_pkg.gravar_arquivo_binario(out_path_box, bmp_bytes_box^)
            print("Salvou (ground-truth):", out_path_box)
        processed = processed + 1

    print("Inferência de validação concluída. Saída:", out_dir)
