import bionix_ml.dados as dados_pkg
import bionix_ml.graficos as graficos_pkg
import bionix_ml.uteis as uteis
import bionix_ml.computacao as computacao_pkg
import bionix_ml.nucleo.Tensor as tensor_defs
import os
# diagnostics.write_trace instrumentation removed; keep file available for later one-shot usage

# Dataset loader: reads dataset tree with .bmp + .txt (.box) files and yields crops as tensors
fn carregar_dataset_detector_pouro(var dir_dataset: String, var altura: Int, var largura: Int, var tipo: String, var max_classes: Int = 100000) -> List[tensor_defs.Tensor]:
    # Simple loader adapted from examples: returns [X, Y] tensors or empty tensors on failure
    var det = List[tensor_defs.Tensor]()
    var classes = List[String]()
    try:
        det = List[tensor_defs.Tensor]()
        # reuse example loader in a minimal form
        classes = os.listdir(dir_dataset)
    except _:
        var out_fail = List[tensor_defs.Tensor]()
        out_fail.append(tensor_defs.Tensor(List[Int]() , tipo))
        out_fail.append(tensor_defs.Tensor(List[Int](), tipo))
        return out_fail^

    # Delegate to example-like construction but keep small and safe
    var positivos = List[List[List[Float32]]]()
    var negativos = List[List[List[Float32]]]()

    var seen = 0
    for nome in classes:
        if seen >= max_classes:
            break
        var caminho = os.path.join(dir_dataset, nome)
        seen = seen + 1
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
            # accept either .txt or legacy .box files
            var txt_path = bmp_path.replace('.bmp', '.txt')
            if not os.path.exists(txt_path):
                var txt_path_box = bmp_path.replace('.bmp', '.box')
                if os.path.exists(txt_path_box):
                    txt_path = txt_path_box
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
                        var ps = p.strip()
                        if len(ps) != 0:
                            campos.append(String(ps))
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
                var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(crop^, altura, largura)
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
                var resizedn = graficos_pkg.redimensionar_matriz_grayscale_nearest(cropn^, altura, largura)
                negativos.append(resizedn^)
                created = created + 1

    var total = len(positivos) + len(negativos)
    if total == 0:
        var out_fail2 = List[tensor_defs.Tensor]()
        out_fail2.append(tensor_defs.Tensor(List[Int](), tipo))
        out_fail2.append(tensor_defs.Tensor(List[Int](), tipo))
        return out_fail2^

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
                var out_idx = idx * features + yy * largura + xx
                # original behavior: write pixel into tensor (debug instrumentation removed)
                x_t.dados[out_idx] = p[yy][xx]
        y_t.dados[idx] = 1.0
        idx = idx + 1
    for n in negativos:
        for yy in range(altura):
            for xx in range(largura):
                var out_idx = idx * features + yy * largura + xx
                # original behavior: write pixel into tensor (debug instrumentation removed)
                x_t.dados[out_idx] = n[yy][xx]
        y_t.dados[idx] = 0.0
        idx = idx + 1

    var out_list = List[tensor_defs.Tensor]()
    out_list.append(x_t.copy())
    out_list.append(y_t.copy())
    return out_list^


# Color loader: returns inputs resized to (altura, largura) with 3 channels and bbox targets normalized to [0,1]
fn carregar_dataset_detector_bbox_color(var dir_dataset: String, var altura: Int, var largura: Int, var tipo: String, var max_classes: Int = 100000) -> List[tensor_defs.Tensor]:
    var classes = List[String]()
    try:
        classes = os.listdir(dir_dataset)
    except _:
        var out_fail = List[tensor_defs.Tensor]()
        out_fail.append(tensor_defs.Tensor(List[Int]() , tipo))
        out_fail.append(tensor_defs.Tensor(List[Int](), tipo))
        return out_fail^

    var features_list = List[List[List[List[Float32]]]]() # [H][W][3]
    var bboxes_list = List[List[Float32]]()

    var seen = 0
    for nome in classes:
        if seen >= max_classes:
            break
        var caminho = os.path.join(dir_dataset, nome)
        seen = seen + 1
        if not os.path.isdir(caminho):
            continue
        var arquivos = List[String]()
        try:
            arquivos = os.listdir(caminho)
        except _:
            continue
        for f in arquivos:
            if not f.endswith('.bmp'):
                continue
            var bmp_path = os.path.join(caminho, f)
            var info = None
            try:
                info = dados_pkg.carregar_bmp_rgb(bmp_path)^
            except _:
                continue
            if info.width == 0 or info.height == 0:
                continue

            # resize color matrix to target using flat-buffer helper
                try:
                    var resized = graficos_pkg.bmp.redimensionar_matriz_rgb_nearest_from_flat(info.flat_pixels.copy(), info.width, info.height, info.channels, altura, largura)
                except _:
                    try:
                        var resized = graficos_pkg.redimensionar_matriz_rgb_nearest(info.pixels.copy(), altura, largura)
                    except _:
                        continue
            features_list.append(resized^)

            # parse first bbox in .txt as before
            var txt_path = bmp_path.replace('.bmp', '.txt')
            var target_box = List[Float32]()
            try:
                var linhas = dados_pkg.carregar_txt_linhas(txt_path)
                if len(linhas) > 0:
                    var parts = linhas[0].replace("\t", " ").replace(",", " ").split(" ")
                    var campos = List[String]()
                    for p in parts:
                        var ps = p.strip()
                        if len(ps) != 0:
                            campos.append(String(ps))
                    if len(campos) >= 4:
                        var w = info.width
                        var h = info.height
                        for i in range(4):
                            try:
                                var v = Float32(uteis.parse_float_ascii(String(campos[i])))
                                if i % 2 == 0:
                                    target_box.append(v / Float32(max(1, w - 1)))
                                else:
                                    target_box.append(v / Float32(max(1, h - 1)))
                            except _:
                                target_box.append(0.0)
            except _:
                target_box = List[Float32]()

            if len(target_box) < 4:
                target_box = List[Float32](); target_box.append(0.0); target_box.append(0.0); target_box.append(0.0); target_box.append(0.0)
            bboxes_list.append(target_box^)

    var total = len(features_list)
    if total == 0:
        var out_fail2 = List[tensor_defs.Tensor]()
        out_fail2.append(tensor_defs.Tensor(List[Int](), tipo))
        out_fail2.append(tensor_defs.Tensor(List[Int](), tipo))
        return out_fail2^

    var features = altura * largura * 3
    var formato_x = List[Int](); formato_x.append(total); formato_x.append(features)
    var formato_y = List[Int](); formato_y.append(total); formato_y.append(4)

    var x_t = tensor_defs.Tensor(formato_x^, tipo)
    var y_t = tensor_defs.Tensor(formato_y^, tipo)
    var idx = 0
    for p in features_list:
        # p is [H][W][3]
        for yy in range(altura):
            for xx in range(largura):
                var pix = p[yy][xx]
                # order R,G,B channels
                x_t.dados[idx * features + (yy * largura + xx) * 3 + 0] = pix[0]
                x_t.dados[idx * features + (yy * largura + xx) * 3 + 1] = pix[1]
                x_t.dados[idx * features + (yy * largura + xx) * 3 + 2] = pix[2]
        var tb = bboxes_list[idx]
        for j in range(4):
            y_t.dados[idx * 4 + j] = tb[j]
        idx = idx + 1

    var out_list = List[tensor_defs.Tensor]()
    out_list.append(x_t.copy())
    out_list.append(y_t.copy())
    return out_list^


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


# New loader: returns full-image inputs resized to (altura, largura) and bbox targets normalized to [0,1]
fn carregar_dataset_detector_bbox(var dir_dataset: String, var altura: Int, var largura: Int, var tipo: String, var max_classes: Int = 100000) -> List[tensor_defs.Tensor]:
    var det = List[tensor_defs.Tensor]()
    var classes = List[String]()
    try:
        classes = os.listdir(dir_dataset)
    except _:
        var out_fail = List[tensor_defs.Tensor]()
        out_fail.append(tensor_defs.Tensor(List[Int]() , tipo))
        out_fail.append(tensor_defs.Tensor(List[Int](), tipo))
        return out_fail^

    var features_list = List[List[List[Float32]]]()
    var bboxes_list = List[List[Float32]]()

    var seen = 0
    for nome in classes:
        if seen >= max_classes:
            break
        var caminho = os.path.join(dir_dataset, nome)
        seen = seen + 1
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
            var img = List[List[Float32]]()
            try:
                img = dados_pkg.carregar_bmp_grayscale_matriz(bmp_path)
            except _:
                continue
            var h = len(img)
            var w = 0
            if h > 0:
                w = len(img[0])
            if h == 0 or w == 0:
                continue

            # Resize the full image to target resolution and use as input
            var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(img.copy(), altura, largura)
            features_list.append(resized^)

            # collect first bbox found as target (if multiple, store first); normalized to [0,1]
            var txt_path = bmp_path.replace('.bmp', '.txt')
            var target_box = List[Float32]()
            try:
                var linhas = dados_pkg.carregar_txt_linhas(txt_path)
                if len(linhas) > 0:
                    var parts = linhas[0].replace("\t", " ").replace(",", " ").split(" ")
                    var campos = List[String]()
                    for p in parts:
                        var ps = p.strip()
                        if len(ps) != 0:
                            campos.append(String(ps))
                    if len(campos) >= 4:
                        for i in range(4):
                            try:
                                var v = Float32(uteis.parse_float_ascii(String(campos[i])))
                                # normalize
                                if i % 2 == 0:
                                    # x coord
                                    target_box.append(v / Float32(max(1, w - 1)))
                                else:
                                    # y coord
                                    target_box.append(v / Float32(max(1, h - 1)))
                            except _:
                                target_box.append(0.0)
            except _:
                target_box = List[Float32]()

            if len(target_box) < 4:
                # fallback: zero box
                target_box = List[Float32]()
                target_box.append(0.0); target_box.append(0.0); target_box.append(0.0); target_box.append(0.0)

            bboxes_list.append(target_box^)

    var total = len(features_list)
    if total == 0:
        var out_fail2 = List[tensor_defs.Tensor]()
        out_fail2.append(tensor_defs.Tensor(List[Int](), tipo))
        out_fail2.append(tensor_defs.Tensor(List[Int](), tipo))
        return out_fail2^

    var features = altura * largura
    var formato_x = List[Int]()
    formato_x.append(total)
    formato_x.append(features)
    var formato_y = List[Int]()
    formato_y.append(total)
    formato_y.append(4)

    var x_t = tensor_defs.Tensor(formato_x^, tipo)
    var y_t = tensor_defs.Tensor(formato_y^, tipo)
    var idx = 0
    for p in features_list:
        for yy in range(altura):
            for xx in range(largura):
                x_t.dados[idx * features + yy * largura + xx] = p[yy][xx]
        var tb = bboxes_list[idx]
        for j in range(4):
            y_t.dados[idx * 4 + j] = tb[j]
        idx = idx + 1

    var out_list = List[tensor_defs.Tensor]()
    out_list.append(x_t.copy())
    out_list.append(y_t.copy())
    return out_list^
