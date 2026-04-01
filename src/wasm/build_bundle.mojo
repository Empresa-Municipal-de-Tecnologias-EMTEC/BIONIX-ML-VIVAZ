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

# ─── Helpers de escrita binária ───────────────────────────────────────────────

fn _wb(f: FileHandle, v: Int):
    var b = List[UInt8]()
    b.append(UInt8(v & 0xFF))
    f.write_bytes(b)

fn _wu16(f: FileHandle, v: Int):
    var b = List[UInt8]()
    b.append(UInt8(v & 0xFF))
    b.append(UInt8((v >> 8) & 0xFF))
    f.write_bytes(b)

fn _wu32(f: FileHandle, v: Int):
    var b = List[UInt8]()
    b.append(UInt8(v & 0xFF))
    b.append(UInt8((v >> 8) & 0xFF))
    b.append(UInt8((v >> 16) & 0xFF))
    b.append(UInt8((v >> 24) & 0xFF))
    f.write_bytes(b)

# Converte Float32 para bytes LE usando a mesma lógica do Tensor.dados_bytes_bin()
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

fn _write_f32(f: FileHandle, v: Float32):
    var bits = _float32_bits(v)
    var b = List[UInt8]()
    b.append(UInt8(bits & 0xFF))
    b.append(UInt8((bits >> 8) & 0xFF))
    b.append(UInt8((bits >> 16) & 0xFF))
    b.append(UInt8((bits >> 24) & 0xFF))
    f.write_bytes(b)

fn _write_f32_list(f: FileHandle, data: List[Float32]):
    var b = List[UInt8](capacity=len(data) * 4)
    for v in data:
        var bits = _float32_bits(v)
        b.append(UInt8(bits & 0xFF))
        b.append(UInt8((bits >> 8) & 0xFF))
        b.append(UInt8((bits >> 16) & 0xFF))
        b.append(UInt8((bits >> 24) & 0xFF))
    f.write_bytes(b)

fn _write_str_bytes(f: FileHandle, s: String):
    var b = List[UInt8]()
    var sb = s.as_bytes()
    for c in sb:
        b.append(c)
    f.write_bytes(b)

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

    var num_k    = arcface.bloco_cnn.num_filtros
    var kh       = arcface.bloco_cnn.kernel_h
    var kw       = arcface.bloco_cnn.kernel_w
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
    var f = open(bundle_path, "wb")

    # Header 24 bytes
    _write_str_bytes(f, "BNFX")    # magic
    _wb(f, 1)                      # version
    _wb(f, num_k)                  # num_kernels
    _wb(f, kh)                     # kernel_h
    _wb(f, kw)                     # kernel_w
    _wu16(f, ps)                   # patch_size
    _wu16(f, E)                    # embed_dim
    _wu32(f, D)                    # feat_dim
    _wu32(f, C)                    # num_classes
    _wu32(f, 0)                    # reserved

    # Kernels
    for ki in range(num_k):
        _write_f32_list(f, arcface.bloco_cnn.kernels[ki].dados)

    # proj_peso [D × E], proj_bias [E]
    _write_f32_list(f, arcface.proj_peso.dados)
    _write_f32_list(f, arcface.proj_bias.dados)

    # cls_peso [E × C], cls_bias [C]
    _write_f32_list(f, arcface.cls_peso.dados)
    _write_f32_list(f, arcface.cls_bias.dados)

    f.close()
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
    var gf = open(gallery_path, "wb")
    _wu32(gf, galeria.tamanho())
    _wu32(gf, E)
    for i in range(galeria.tamanho()):
        var nome = galeria.nomes[i]
        var nb   = nome.as_bytes()
        _wb(gf, min(len(nb), 255))
        for j in range(min(len(nb), 255)):
            var b = List[UInt8]()
            b.append(nb[j])
            gf.write_bytes(b)
        _write_f32_list(gf, galeria.embeddings[i])
    gf.close()
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
