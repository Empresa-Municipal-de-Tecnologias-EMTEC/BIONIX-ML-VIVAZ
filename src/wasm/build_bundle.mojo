# Exporta os pesos do modelo ArcFace treinado para um bundle binário que o
# módulo WebAssembly carrega no browser.
#
# Saída em build/:
#   arcface_bundle.bin  — pesos CNN + projeção + classificação
#   arcface_gallery.bin — embeddings da galeria (para identificação)
#   arcface_gallery.json — nomes das identidades da galeria
#
# Formato de arcface_bundle.bin:
#   0..3   : magic "BNFX"
#   4      : versao (1)
#   5      : num_kernels (ex. 8)
#   6      : kernel_h (3)
#   7      : kernel_w (3)
#   8..9   : patch_size uint16_LE
#   10..11 : embed_dim  uint16_LE
#   12..15 : feat_dim   uint32_LE
#   16..19 : num_classes uint32_LE
#   20..23 : reservado (0)
#   [24..] : float32_LE[num_kernels * kernel_h * kernel_w]  — kernels
#            float32_LE[feat_dim * embed_dim]               — proj_peso
#            float32_LE[embed_dim]                          — proj_bias
#            float32_LE[embed_dim * num_classes]            — cls_peso
#            float32_LE[num_classes]                        — cls_bias
#
# Formato de arcface_gallery.bin:
#   0..3   : n_classes uint32_LE
#   4..7   : embed_dim uint32_LE
#   [8..]  : para cada classe:
#              byte     name_len
#              char[name_len]  name ASCII
#              float32_LE[embed_dim]  embedding

import reconhecedor.arcface_model as arc_model
import reconhecedor.arcface_infer  as infer_pkg
import bionix_ml.uteis as uteis
import os
import math

# ─── Helpers: acumulam bytes em List[UInt8] ───────────────────────────────────

fn _wb(mut buf: List[UInt8], v: Int):
    buf.append(UInt8(v & 0xFF))

fn _wu16(mut buf: List[UInt8], v: Int):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))

fn _wu32(mut buf: List[UInt8], v: Int):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8)  & 0xFF))
    buf.append(UInt8((v >> 16) & 0xFF))
    buf.append(UInt8((v >> 24) & 0xFF))

fn _float32_bits(var v: Float32) -> Int:
    if v == 0.0: return 0
    var af = v
    var sign = 0
    if af < 0.0: sign = 1; af = -af
    var frep = math.frexp(Float64(af))
    var m = frep[0]; var e = Int(frep[1])
    var exp_bits = e + 126
    if exp_bits <= 0: return sign << 31
    var frac = (m * 2.0 - 1.0)
    var frac_bits = Int(frac * Float64(1 << 23)) & 0x7FFFFF
    return (sign << 31) | ((exp_bits & 0xFF) << 23) | frac_bits

fn _wf32(mut buf: List[UInt8], v: Float32):
    var bits = _float32_bits(v)
    buf.append(UInt8(bits & 0xFF))
    buf.append(UInt8((bits >> 8)  & 0xFF))
    buf.append(UInt8((bits >> 16) & 0xFF))
    buf.append(UInt8((bits >> 24) & 0xFF))

fn _wf32_list(mut buf: List[UInt8], data: List[Float32]):
    for v in data:
        _wf32(buf, v)

fn _wstr(mut buf: List[UInt8], s: String):
    var sb = s.as_bytes()
    for c in sb:
        buf.append(c)

# ─── Exportação dos pesos ─────────────────────────────────────────────────────

fn main() raises:
    var model_dir   = os.path.join("MODELO", "arcface_modelo")
    var dataset_dir = "DATASET"
    var build_dir   = "build"

    try:
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
    except _: pass

    print("Carregando modelo ArcFace de:", model_dir)
    var params  = arc_model.ArcFaceParametros(64, 8, 3, 3, 128, 0, "cpu")
    var arcface = arc_model.ArcFace(params^, model_dir)
    if not arcface.carregar():
        print("ERRO: checkpoint nao encontrado em", model_dir)
        print("Execute o treino primeiro: pixi run run_arcface_train_debug")
        return

    var num_k: Int = 0
    var kh: Int = 3
    var kw: Int = 3
    try:
        if len(arcface.bloco_kernels) > 0:
            num_k = len(arcface.bloco_kernels)
            try:
                kh = arcface.bloco_kernels[0].formato[0]
                kw = arcface.bloco_kernels[0].formato[1]
            except _:
                kh = 3; kw = 3
        else:
            num_k = arcface.bloco_cnn.num_filtros
            kh = arcface.bloco_cnn.kernel_h
            kw = arcface.bloco_cnn.kernel_w
    except _:
        try:
            num_k = arcface.bloco_cnn.num_filtros
            kh = arcface.bloco_cnn.kernel_h
            kw = arcface.bloco_cnn.kernel_w
        except _:
            num_k = 0; kh = 3; kw = 3
    var ps       = arcface.parametros.patch_size
    var E        = arcface.parametros.embed_dim
    var C        = arcface.parametros.num_classes
    var conv_h   = ps - kh + 1
    var conv_w   = ps - kw + 1
    var pool_h   = conv_h // 2
    var pool_w   = conv_w // 2
    var D        = num_k * pool_h * pool_w

    print("  Kernels: ", num_k, "x", kh, "x", kw)
    print("  feat_dim:", D, " embed_dim:", E, " classes:", C)

    # ── Escreve arcface_bundle.bin ────────────────────────────────────────────
    var bundle_path = os.path.join(build_dir, "arcface_bundle.bin")
    var bundle_buf = List[UInt8](capacity=64 * 1024 * 1024)  # 64 MB cap

    # Header 24 bytes
    _wstr(bundle_buf, "BNFX")     # magic
    _wb(bundle_buf, 1)             # version
    _wb(bundle_buf, num_k)         # num_kernels
    _wb(bundle_buf, kh)            # kernel_h
    _wb(bundle_buf, kw)            # kernel_w
    _wu16(bundle_buf, ps)          # patch_size
    _wu16(bundle_buf, E)           # embed_dim
    _wu32(bundle_buf, D)           # feat_dim
    _wu32(bundle_buf, C)           # num_classes
    _wu32(bundle_buf, 0)           # reserved

    # Kernels (prefer tensor-backed `bloco_kernels`, fallback to legacy `bloco_cnn.kernels`)
    for ki in range(num_k):
        try:
            if len(arcface.bloco_kernels) > 0:
                _wf32_list(bundle_buf, arcface.bloco_kernels[ki].dados)
            else:
                _wf32_list(bundle_buf, arcface.bloco_cnn.kernels[ki].dados)
        except _:
            # write zeros if missing
            var zeros = List[Float32]()
            for _ in range(kh * kw): zeros.append(0.0)
            _wf32_list(bundle_buf, zeros)

    # proj_peso [D × E], proj_bias [E]
    _wf32_list(bundle_buf, arcface.proj_peso.dados)
    _wf32_list(bundle_buf, arcface.proj_bias.dados)

    # cls_peso [E × C], cls_bias [C]
    _wf32_list(bundle_buf, arcface.cls_peso.dados)
    _wf32_list(bundle_buf, arcface.cls_bias.dados)

    var bundle_f = open(bundle_path, "w")
    bundle_f.write_bytes(bundle_buf)
    bundle_f.close()
    print("  Bundle gravado:", bundle_path)

    # ── Constrói galeria e exporta ────────────────────────────────────────────
    print("Construindo galeria de embeddings de:", dataset_dir)
    var galeria = infer_pkg.Galeria()
    try:
        galeria = infer_pkg.construir_galeria(arcface, dataset_dir)
    except _: pass
    print("  Galeria:", galeria.tamanho(), "identidades")

    # arcface_gallery.bin
    var gallery_path = os.path.join(build_dir, "arcface_gallery.bin")
    var gallery_buf  = List[UInt8](capacity=galeria.tamanho() * (E * 4 + 256))
    _wu32(gallery_buf, galeria.tamanho())
    _wu32(gallery_buf, E)
    for i in range(galeria.tamanho()):
        var nome = galeria.nomes[i]
        var nb   = nome.as_bytes()
        _wb(gallery_buf, min(len(nb), 255))
        for j in range(min(len(nb), 255)):
            gallery_buf.append(nb[j])
        _wf32_list(gallery_buf, galeria.embeddings[i])
    var gallery_f = open(gallery_path, "w")
    gallery_f.write_bytes(gallery_buf)
    gallery_f.close()
    print("  Galeria bin gravada:", gallery_path)

    # arcface_gallery.json  (só nomes)
    var json_path = os.path.join(build_dir, "arcface_gallery.json")
    var names_json = "{\"names\":["
    for i in range(galeria.tamanho()):
        if i > 0: names_json += ","
        names_json += "\"" + galeria.nomes[i] + "\""
    names_json += "]}"
    var jf2 = open(json_path, "w")
    jf2.write(names_json)
    jf2.close()
    print("  Galeria JSON gravada:", json_path)

    # ── Copia modelos para build/ (para deploy junto ao binário) ─────────────
    print("Exportação concluída. Arquivos em:", build_dir)
    print("")
    print("Próximos passos:")
    print("  1. pixi run build_wasm     (requer emcc instalado)")
    print("  2. pixi run build_api")
    print("  3. Sirva build/ com um servidor HTTP e abra index.html")
