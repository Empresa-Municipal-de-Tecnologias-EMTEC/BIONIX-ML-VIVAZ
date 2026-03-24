import bionix_ml.dados as dados_pkg
import bionix_ml.graficos as graficos_pkg
import bionix_ml.uteis as uteis
import bionix_ml.computacao as computacao_pkg
import bionix_ml.nucleo.Tensor as tensor_defs
import os

# Dataset loader: reads dataset tree with .bmp + .txt (.box) files and yields crops as tensors
fn carregar_dataset_detector_pouro(var dir_dataset: String, var altura: Int, var largura: Int, var tipo: String) -> (tensor_defs.Tensor, tensor_defs.Tensor):
    # Simple loader adapted from examples: returns (X, Y) tensors or empty tensors on failure
    var det = List[tensor_defs.Tensor]()
    try:
        det = List[tensor_defs.Tensor]()
        # reuse example loader in a minimal form
        var classes = List[String]()
        classes = os.listdir(dir_dataset)
    except _:
        return tensor_defs.Tensor(List[Int]() , tipo), tensor_defs.Tensor(List[Int](), tipo)

    # Delegate to example-like construction but keep small and safe
    var positivos = List[List[List[Float32]]]()
    var negativos = List[List[List[Float32]]]()

    for nome in classes:
        var caminho = os.path.join(dir_dataset, nome)
        if not os.path.isdir(caminho):
            continue
        var arquivos = List[String]()
        try:
            arquivos = os.listdir(caminho)
        except _:
            continue
        for f in arquivos:
            if not f.endswith(".bmp"):
                continue
            var bmp_path = os.path.join(caminho, f)
            var txt_path = bmp_path.replace('.bmp', '.txt')
            var img = List[List[Float32]]()
            try:
                img = dados_pkg.carregar_bmp_grayscale_matriz(bmp_path)
            except _:
                continue
            var bboxes = List[List[Int]]()
            try:
                var linhas = dados_pkg.carregar_txt_linhas(txt_path)
                for l in linhas:
                    var parts = l.replace("\t", " ").replace(",", " ").split(" ")
                    var campos = List[String]()
                    for p in parts:
                        if p.strip() != "":
                            campos.append(p.strip())
                    if len(campos) >= 4:
                        var b = List[Int]()
                        for i in range(4):
                            try:
                                var v = Float32(uteis.parse_float_ascii(String(campos[i])))
                                b.append(Int(v))
                            except _:
                                b.append(0)
                        bboxes.append(b^)
            except _:
                bboxes = List[List[Int]]()
            for b in bboxes:
                var crop = _crop_matrix(img, b[0], b[1], b[2], b[3])
                if len(crop) == 0:
                    continue
                var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(crop, altura, largura)
                positivos.append(resized^)
            # negatives: simple random windows
            var h = len(img)
            var w = 0
            if h > 0:
                w = len(img[0])
            var tries = 0
            var created = 0
            while tries < 20 and created < max(1, len(bboxes)):
                tries = tries + 1
                var rx = (tries * 17) % max(1, max(1, w - largura + 1))
                var ry = (tries * 13) % max(1, max(1, h - altura + 1))
                var box = List[Int]()
                box.append(rx); box.append(ry); box.append(rx + largura - 1); box.append(ry + altura - 1)
                var ok = True
                for bb in bboxes:
                    if _iou(box, bb) > 0.1:
                        ok = False
                        break
                if not ok:
                    continue
                var cropn = _crop_matrix(img, box[0], box[1], box[2], box[3])
                if len(cropn) == 0:
                    continue
                var resizedn = graficos_pkg.redimensionar_matriz_grayscale_nearest(cropn, altura, largura)
                negativos.append(resizedn^)
                created = created + 1

    var total = len(positivos) + len(negativos)
    if total == 0:
        return tensor_defs.Tensor(List[Int](), tipo), tensor_defs.Tensor(List[Int](), tipo)

    var features = altura * largura
    var formato_x = List[Int]()
    formato_x.append(total)
    formato_x.append(features)
    var formato_y = List[Int]()
    formato_y.append(total)
    formato_y.append(1)

    var x_t = tensor_defs.Tensor(formato_x^, tipo)
    var y_t = tensor_defs.Tensor(formato_y^, tipo)
    var idx = 0
    for p in positivos:
        for yy in range(altura):
            for xx in range(largura):
                x_t.dados[idx * features + yy * largura + xx] = p[yy][xx]
        y_t.dados[idx] = 1.0
        idx = idx + 1
    for n in negativos:
        for yy in range(altura):
            for xx in range(largura):
                x_t.dados[idx * features + yy * largura + xx] = n[yy][xx]
        y_t.dados[idx] = 0.0
        idx = idx + 1

    return x_t.copy(), y_t.copy()


# Reuse IO helpers from example: IoU and bbox parsing small helpers
fn _iou(boxA: List[Int], boxB: List[Int]) -> Float32:
    var xA = max(boxA[0], boxB[0])
    var yA = max(boxA[1], boxB[1])
    var xB = min(boxA[2], boxB[2])
    var yB = min(boxA[3], boxB[3])
    var interW = xB - xA + 1
    var interH = yB - yA + 1
    if interW <= 0 or interH <= 0:
        return 0.0
    var interArea = Float32(interW * interH)
    var boxAArea = Float32((boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1))
    var boxBArea = Float32((boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1))
    return interArea / (boxAArea + boxBArea - interArea)


fn _crop_matrix(var m: List[List[Float32]], var x1: Int, var y1: Int, var x2: Int, var y2: Int) -> List[List[Float32]]:
    var h = len(m)
    if h == 0:
        return List[List[Float32]]()
    var w = len(m[0])
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 >= w:
        x2 = w - 1
    if y2 >= h:
        y2 = h - 1
    if x2 < x1 or y2 < y1:
        return List[List[Float32]]()
    var out = List[List[Float32]]()
    for yy in range(y1, y2 + 1):
        var row = List[Float32]()
        for xx in range(x1, x2 + 1):
            row.append(m[yy][xx])
        out.append(row^)
    return out^
