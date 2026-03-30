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
import retina.model_detector as model_pkg
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import bionix_ml.camadas.cnn as cnn_pkg
import bionix_ml.graficos as graficos_pkg
import bionix_ml.uteis as uteis
import retina.retina_model as model_utils

fn inferir_retina_imagem(model_dir: String, img_pixels: List[List[List[Float32]]], input_size: Int = 640, max_per_image: Int = 16) -> List[List[Int]]:
    # Per-image inference using RetinaFace wrapper. Returns list of boxes.
    var params = model_utils.BlocoRetinaFaceParametros(input_size)
    var detector = model_utils.RetinaFace(params^, model_dir)
    try:
        var loaded = detector.carregar_workspace(model_dir)
    except _:
        loaded = False
    if not loaded:
        print("[INFER] checkpoint do bloco não encontrado em", model_dir)
        return List[List[Int]]()

    try:
            var boxes = detector.inferir(img_pixels.copy(), input_size, max_per_image)
        return boxes
    except _:
        return List[List[Int]]()


fn validar_10_classes(model_dir: String, dataset_dir: String, out_dir: String, input_size: Int = 640, max_per_image: Int = 8, n_classes: Int = 10) -> Bool:
    # Run inference over up to `n_classes` classes (one image per class), saving overlays to out_dir.
    var params = model_utils.BlocoRetinaFaceParametros(input_size)
    var detector = model_utils.RetinaFace(params^, model_dir)
    try:
        var loaded = detector.carregar_workspace(model_dir)
    except _:
        loaded = False
    if not loaded:
        print("[INFER] checkpoint do bloco não encontrado em", model_dir)
        return False

    # ensure out_dir exists
    try:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    except _:
        pass

    # determine class folders
    var train_root = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_root):
        train_root = dataset_dir

    var labels = List[String]()
    try:
        labels = os.listdir(train_root)
    except _:
        print("[INFER] falha ao listar classes em", train_root)
        return False

    var processed = 0
    for lbl in labels:
        if processed >= n_classes:
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

        print("[INFER] carregando imagem:", img_path)
        var bmp = bmpmod.zero_bmp()
        try:
            bmp = dados_pkg.carregar_bmp_rgb(img_path, input_size, input_size)
        except _:
            print("[INFER] excecao: falha ao carregar imagem:", img_path)
            continue
        if bmp.width <= 0:
            print("[INFER] falha ao carregar imagem:", img_path)
            continue

        var boxes = List[List[Int]]()
        try:
            boxes = detector.inferir(bmp.pixels, input_size, max_per_image)
        except _:
            boxes = List[List[Int]]()

        # build flat rgb base
        var flat_rgb = List[Int]()
        for y in range(bmp.height):
            for x in range(bmp.width):
                var px = bmp.pixels[y][x]
                flat_rgb.append(Int(px[0] * 255.0)); flat_rgb.append(Int(px[1] * 255.0)); flat_rgb.append(Int(px[2] * 255.0))

        # draw predicted boxes (blue)
        var flat_pred = flat_rgb.copy()
        for box in boxes:
            graficos_pkg.draw_bbox_on_flat_rgb(flat_pred, bmp.width, bmp.height, box, 0, 0, 255)

        # save predicted overlay
        try:
            var out_name = lbl + "_pred.bmp"
            var out_path = os.path.join(out_dir, out_name)
            var bmp_bytes = graficos_pkg.bmp.gerar_bmp_24bits_de_rgb(flat_pred, bmp.width, bmp.height)
            dados_pkg.gravar_arquivo_binario(out_path, bmp_bytes^)
            print("[INFER] salvo preditos:", out_path)
        except _:
            pass

        # read ground-truth .box or .txt and draw (green)
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
                        var w = bmp.width; var h = bmp.height
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

        if len(file_box) >= 4:
            var flat_box = flat_rgb.copy()
            var coords = List[Int]()
            coords.append(file_box[0]); coords.append(file_box[1]); coords.append(file_box[2]); coords.append(file_box[3])
            graficos_pkg.draw_bbox_on_flat_rgb(flat_box, bmp.width, bmp.height, coords^, 0, 255, 0)
            try:
                var out_name_box = lbl + "_box.bmp"
                var out_path_box = os.path.join(out_dir, out_name_box)
                var bmp_bytes_box = graficos_pkg.bmp.gerar_bmp_24bits_de_rgb(flat_box, bmp.width, bmp.height)
                dados_pkg.gravar_arquivo_binario(out_path_box, bmp_bytes_box^)
                print("[INFER] salvo ground-truth:", out_path_box)
            except _:
                pass

        processed = processed + 1

    return True
