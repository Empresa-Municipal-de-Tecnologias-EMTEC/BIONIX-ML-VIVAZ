import reconhecedor.arcface_model as arc_model
import bionix_ml.camadas.cnn as cnn_pkg
import bionix_ml.nucleo.Tensor as tensor_defs
import bionix_ml.dados as dados_pkg
import bionix_ml.dados.arquivo as arquivo_pkg
import bionix_ml.uteis as uteis
import bionix_ml.graficos as graficos_pkg
import math
import os


# Divide string por espaço, tab ou vírgula
fn _split_fields(line: String) -> List[String]:
    var fields = List[String]()
    var cur = String("")
    for ch in line:
        if ch == '\t' or ch == ',' or ch == ' ':
            if len(cur) > 0:
                fields.append(cur)
                cur = String("")
        else:
            cur = cur + String(ch)
    if len(cur) > 0:
        fields.append(cur)
    return fields^


# ─── Loop de treino ────────────────────────────────────────────────────────────
#
# Formato do dataset esperado (mesmo do RetinaFace):
#   dataset_dir/
#     train/
#       <identidade_A>/  ← nome do diretório = rótulo da classe
#         img1.bmp
#         img2.bmp
#       <identidade_B>/
#         ...
#
# Cada imagem deve ter um arquivo `.box` opcional (x0 y0 x1 y1 em pixels ou [0,1])
# com o recorte do rosto. Se não existir, a imagem inteira é usada.
#
# Perda: entropia cruzada softmax sobre a cabeça de classificação (ArcFace-lite).
# As cabeças de projeção e classificação são atualizadas por gradiente manual (SGD).

fn treinar_arcface(mut modelo: arc_model.ArcFace,
                   var dataset_dir: String,
                   var epocas: Int = 5,
                   var taxa_aprendizado: Float32 = 0.0001,
                   var batch_size: Int = 4) raises -> String:

    var ps = modelo.parametros.patch_size
    var E  = modelo.parametros.embed_dim

    # Carrega checkpoint existente
    _ = modelo.carregar()
    var ep_inicio = 0
    if modelo.treinamento_epoca >= 0:
        ep_inicio = modelo.treinamento_epoca + 1
    print("ArcFace treino: dataset=", dataset_dir, " epocas=", epocas, " lr=", taxa_aprendizado)
    if ep_inicio > 0:
        print("Retomando da epoca", ep_inicio)

    # ── Descoberta das classes ──────────────────────────────────────────────────
    var train_root = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_root):
        train_root = dataset_dir

    var class_names  = List[String]()
    var class_images = List[List[String]]()
    try:
        for cls in os.listdir(train_root):
            var pcls = os.path.join(train_root, cls)
            if not os.path.isdir(pcls):
                continue
            var imgs = List[String]()
            try:
                for f in os.listdir(pcls):
                    if f.endswith('.bmp'):
                        imgs.append(os.path.join(pcls, f))
            except _:
                pass
            if len(imgs) > 0:
                class_names.append(cls)
                class_images.append(imgs^)
    except _:
        pass

    var C = len(class_names)
    if C == 0:
        return "Falha: nenhuma classe encontrada em " + train_root

    # Actualiza num_classes no modelo caso necessário
    if modelo.parametros.num_classes != C:
        modelo.parametros.num_classes = C

    print("Classes encontradas:", C)

    # ── Pré-inicializa cabeças se já temos feat_dim salvo ──────────────────────
    # (será re-inicializado na primeira forward se necessário)
    var heads_ok = modelo.heads_inicializadas

    # ── LR schedule ────────────────────────────────────────────────────────────
    var lr_atual  = taxa_aprendizado
    var best_loss: Float32 = 1e30
    var sched_wait: Int = 0

    # ── Percorre épocas ────────────────────────────────────────────────────────
    for ep in range(ep_inicio, ep_inicio + epocas):
        print("Epoca", ep, "iniciando... (C=", C, " lr=", lr_atual, ")")

        var soma_loss: Float32 = 0.0
        var count: Int = 0

        # Percorre imagens de cada classe (round-robin, 1 img/classe/época)
        for c in range(C):
            var imgs = class_images[c].copy()
            if len(imgs) == 0:
                continue

            # Seleciona imagem via LCG (pseudo-aleatório)
            var seed = (ep * 1664525 + c * 1013904223 + 12345) & 0x7fffffff
            var idx  = seed % len(imgs)
            var img_path = imgs[idx]

            # Carrega BMP RGB
            var bmp = dados_pkg.carregar_bmp_rgb(img_path)
            if bmp.width == 0:
                continue

            # Lê .box se existir
            var x0: Int = 0; var y0: Int = 0
            var x1: Int = bmp.width - 1; var y1: Int = bmp.height - 1
            try:
                var box_path = img_path.replace('.bmp', '.box')
                var lines = dados_pkg.carregar_txt_linhas(box_path)
                if len(lines) > 0:
                    var p = _split_fields(lines[0])
                    if len(p) >= 4:
                        var tx0 = Float32(uteis.parse_float_ascii(String(p[0])))
                        var ty0 = Float32(uteis.parse_float_ascii(String(p[1])))
                        var tx1 = Float32(uteis.parse_float_ascii(String(p[2])))
                        var ty1 = Float32(uteis.parse_float_ascii(String(p[3])))
                        if tx0 > 1.5 or ty0 > 1.5:
                            x0 = Int(tx0); y0 = Int(ty0); x1 = Int(tx1); y1 = Int(ty1)
                        else:
                            x0 = Int(tx0 * Float32(bmp.width  - 1))
                            y0 = Int(ty0 * Float32(bmp.height - 1))
                            x1 = Int(tx1 * Float32(bmp.width  - 1))
                            y1 = Int(ty1 * Float32(bmp.height - 1))
            except _:
                pass

            # Crop & resize → tensor grayscale [1, ps*ps]
            var patch = graficos_pkg.crop_and_resize_rgb(bmp.pixels, x0, y0, x1, y1, ps, ps)
            var in_shape = List[Int](); in_shape.append(1); in_shape.append(ps * ps)
            var tensor_in = tensor_defs.Tensor(in_shape^, modelo.bloco_cnn.tipo_computacao)
            for yy in range(ps):
                for xx in range(ps):
                    var pix = patch[yy][xx].copy()
                    tensor_in.dados[yy * ps + xx] = Float32(0.299)*pix[0] + Float32(0.587)*pix[1] + Float32(0.114)*pix[2]

            # Forward: BlocoCNN → features
            var feats = cnn_pkg.extrair_features(modelo.bloco_cnn, tensor_in)
            var D = feats.formato[1]
            if D == 0:
                continue

            # Inicializa cabeças na primeira amostra
            if not heads_ok:
                modelo._init_heads(D)
                heads_ok = True

            # Verifica coerência da projeção: [D, E]
            if not (len(modelo.proj_peso.formato) == 2 and
                    modelo.proj_peso.formato[0] == D and
                    modelo.proj_peso.formato[1] == E):
                modelo._init_heads(D)
                heads_ok = True

            # Projeção [D→E]
            var proj = List[Float32]()
            for e in range(E):
                var s: Float32 = modelo.proj_bias.dados[e]
                for d in range(D):
                    s = s + feats.dados[d] * modelo.proj_peso.dados[d * E + e]
                proj.append(s)

            # L2-normalização do embedding
            var norm: Float32 = 1e-8
            for v in proj:
                norm = norm + v * v
            norm = Float32(math.sqrt(Float64(norm)))
            var emb = List[Float32]()
            for v in proj:
                emb.append(v / norm)

            # ── Cabeça de classificação: logits = emb × cls_peso + cls_bias ────
            if not (len(modelo.cls_peso.formato) == 2 and
                    modelo.cls_peso.formato[0] == E and
                    modelo.cls_peso.formato[1] == C):
                modelo._init_heads(D)
                heads_ok = True

            var logits = List[Float32]()
            for k in range(C):
                var s: Float32 = modelo.cls_bias.dados[k]
                for e in range(E):
                    s = s + emb[e] * modelo.cls_peso.dados[e * C + k]
                logits.append(s)

            # ── Softmax + cross-entropy loss ────────────────────────────────────
            var max_l: Float32 = logits[0]
            for v in logits:
                if v > max_l: max_l = v
            var sum_e: Float32 = 0.0
            var exps = List[Float32]()
            for v in logits:
                var ex = Float32(math.exp(Float64(v - max_l)))
                exps.append(ex)
                sum_e = sum_e + ex
            var probs = List[Float32]()
            for ex in exps:
                probs.append(ex / sum_e)

            var loss: Float32 = 0.0
            var p_true = probs[c]
            if p_true > 1e-8:
                loss = -Float32(math.log(Float64(p_true)))
            soma_loss = soma_loss + loss
            count = count + 1

            # ── Gradientes (SGD manual) ─────────────────────────────────────────
            # grad_logits = probs − one_hot(c)
            var grad_logits = probs.copy()
            grad_logits[c] = grad_logits[c] - 1.0

            # Clamp gradiente
            for k in range(C):
                if grad_logits[k] > 10.0: grad_logits[k] = 10.0
                if grad_logits[k] < -10.0: grad_logits[k] = -10.0

            # ∂L/∂cls_peso[e,k] = emb[e] * grad_logits[k]
            for e in range(E):
                for k in range(C):
                    var gw = emb[e] * grad_logits[k]
                    if gw > 10.0: gw = 10.0
                    if gw < -10.0: gw = -10.0
                    modelo.cls_peso.dados[e * C + k] = modelo.cls_peso.dados[e * C + k] - lr_atual * gw
            for k in range(C):
                modelo.cls_bias.dados[k] = modelo.cls_bias.dados[k] - lr_atual * grad_logits[k]

            # ∂L/∂emb[e] = Σ_k grad_logits[k] * cls_peso[e,k]
            var grad_emb = List[Float32]()
            for e in range(E):
                var g: Float32 = 0.0
                for k in range(C):
                    g = g + grad_logits[k] * modelo.cls_peso.dados[e * C + k]
                grad_emb.append(g)

            # ∂L/∂proj[e] via L2-norm: grad_proj = (I - emb⊗emb) * grad_emb / norm
            var grad_proj = List[Float32]()
            var dot: Float32 = 0.0
            for e in range(E):
                dot = dot + grad_emb[e] * emb[e]
            for e in range(E):
                var gp = (grad_emb[e] - dot * emb[e]) / norm
                if gp > 10.0: gp = 10.0
                if gp < -10.0: gp = -10.0
                grad_proj.append(gp)

            # ∂L/∂proj_peso[d,e] = feats[d] * grad_proj[e]
            for d in range(D):
                for e in range(E):
                    var gw = feats.dados[d] * grad_proj[e]
                    if gw > 10.0: gw = 10.0
                    if gw < -10.0: gw = -10.0
                    modelo.proj_peso.dados[d * E + e] = modelo.proj_peso.dados[d * E + e] - lr_atual * gw
            for e in range(E):
                modelo.proj_bias.dados[e] = modelo.proj_bias.dados[e] - lr_atual * grad_proj[e]

        # ── Fim de época ───────────────────────────────────────────────────────
        var avg_loss: Float32 = 0.0
        if count > 0:
            avg_loss = soma_loss / Float32(count)
        print("Epoca", ep, "Avg CE loss:", avg_loss, " Amostras:", count, " LR:", lr_atual)

        modelo.treinamento_epoca = ep
        modelo.treinamento_lr    = lr_atual
        _ = modelo.salvar()
        print("Checkpoint salvo: epoca", ep)

        # ReduceLROnPlateau
        if avg_loss + 1e-6 < best_loss:
            best_loss  = avg_loss
            sched_wait = 0
        else:
            sched_wait = sched_wait + 1
            if sched_wait >= 3:
                lr_atual = lr_atual * 0.5
                if lr_atual < 1e-7:
                    lr_atual = 1e-7
                sched_wait = 0
                print("LR reduzido para", lr_atual)

    return "OK"
