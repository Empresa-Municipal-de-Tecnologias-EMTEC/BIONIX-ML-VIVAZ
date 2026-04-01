# Bionix API HTTP — servidor TCP/HTTP sequencial em Mojo com POSIX sockets.
#
# Endpoints:
#   GET  /health              → {"status":"ok","version":"1.0.0"}
#   POST /detectar            → body: BMP bytes
#                             → {"boxes":[[x0,y0,x1,y1],...]}
#   POST /identificar         → body: BMP bytes
#                             → {"identidade":"X","score":0.95,"box":[x0,y0,x1,y1]}
#   POST /verificar_par       → body: uint32_LE(tam_A) + BMP_A + BMP_B
#                             → {"mesma_pessoa":true,"score":0.92}
#   POST /embedding           → body: BMP bytes
#                             → {"embedding":[...128 floats]}
#   GET  /galeria             → {"identidades":["A","B",...]}
#   POST /galeria/construir   → rebuild gallery from dataset
#                             → {"ok":true,"classes":46}
#   POST /galeria/adicionar   → body: byte[0]=name_len, bytes[1..name_len]=name, rest=BMP
#                             → {"ok":true,"galeria_size":47}
#   OPTIONS *                 → CORS preflight (204)

from sys.ffi import external_call
from memory import UnsafePointer
import reconhecedor.arcface_model as arc_model
import reconhecedor.arcface_infer as infer_pkg
import retina.retina_model as retina_mod
import bionix_ml.dados as dados_pkg
import bionix_ml.uteis as uteis
import os
import math

alias MAX_RECV  = 4 * 1024 * 1024   # 4 MB — cobre imagens de face em alta res
alias BACKLOG   = 64
alias TMP_A     = "/tmp/bionix_a.bmp"
alias TMP_B     = "/tmp/bionix_b.bmp"
alias VERSION   = "1.0.0"

# ─── Helpers de socket ────────────────────────────────────────────────────────

fn _tcp_listen(port: Int) raises -> Int32:
    var fd = external_call["socket", Int32](Int32(2), Int32(1), Int32(0))
    if fd < 0:
        raise Error("socket() falhou")
    # SO_REUSEADDR
    var opt = UnsafePointer[Int32].alloc(1)
    opt[0] = 1
    _ = external_call["setsockopt", Int32](fd, Int32(1), Int32(2), opt, Int32(4))
    opt.free()
    # sockaddr_in: 2 bytes family + 2 bytes port_BE + 4 bytes addr + 8 zeros = 16 bytes
    var addr = UnsafePointer[UInt8].alloc(16)
    for i in range(16): addr[i] = 0
    addr[0] = 2; addr[1] = 0                       # AF_INET (little-endian machine)
    addr[2] = UInt8((port >> 8) & 0xFF)            # port big-endian high byte
    addr[3] = UInt8(port & 0xFF)                   # port big-endian low  byte
    if external_call["bind", Int32](fd, addr, Int32(16)) < 0:
        addr.free()
        raise Error("bind() falhou — porta " + String(port) + " em uso?")
    addr.free()
    if external_call["listen", Int32](fd, Int32(BACKLOG)) < 0:
        raise Error("listen() falhou")
    return fd

fn _tcp_accept(sfd: Int32) -> Int32:
    return external_call["accept", Int32](sfd, Int(0), Int(0))

fn _tcp_close(fd: Int32):
    _ = external_call["close", Int32](fd)

# ─── Helpers de bytes/string ──────────────────────────────────────────────────

# Encontra primeiro índice do byte `b` em buf[start..end).
fn _find_byte(buf: UnsafePointer[UInt8], start: Int, end: Int, b: UInt8) -> Int:
    for i in range(start, end):
        if buf[i] == b:
            return i
    return -1

# Compara buf[off..off+len(s)) com os bytes de s. Retorna True se igual.
fn _eq_at(buf: UnsafePointer[UInt8], buf_len: Int, off: Int, s: String) -> Bool:
    var sb = s.as_bytes()
    var n = len(sb)
    if off + n > buf_len:
        return False
    for i in range(n):
        if buf[off + i] != sb[i]:
            return False
    return True

# Converte buf[0..n) para String (ASCII).
fn _to_str(buf: UnsafePointer[UInt8], n: Int) -> String:
    var s = String("")
    for i in range(n):
        var b = Int(buf[i])
        if b < 128:
            s += chr(b)
    return s

# Lê uint32 little-endian da posição off.
fn _u32le(buf: UnsafePointer[UInt8], off: Int) -> Int:
    return Int(buf[off]) | (Int(buf[off+1]) << 8) | (Int(buf[off+2]) << 16) | (Int(buf[off+3]) << 24)

# Analisa Content-Length em cabeçalhos (buf[0..hdr_end)).
fn _parse_content_length(buf: UnsafePointer[UInt8], hdr_end: Int) -> Int:
    var tag = "Content-Length: "
    var tb = tag.as_bytes()
    var tn = len(tb)
    for i in range(hdr_end - tn):
        var found = True
        for j in range(tn):
            if buf[i + j] != tb[j]:
                found = False
                break
        if found:
            var val = 0
            var k = i + tn
            while k < hdr_end:
                var d = Int(buf[k])
                if d >= 48 and d <= 57:
                    val = val * 10 + d - 48
                else:
                    break
                k += 1
            return val
    return 0

# Recebe request HTTP completo. Retorna (total_bytes, header_end, content_length).
fn _recv_http(fd: Int32, buf: UnsafePointer[UInt8]) -> (Int, Int, Int):
    var total = 0
    var header_end = -1
    var content_length = 0
    while total < MAX_RECV:
        var want = min(8192, MAX_RECV - total)
        var n = Int(external_call["recv", Int](fd, buf + total, want, Int32(0)))
        if n <= 0:
            break
        total += n
        if header_end < 0:
            # procura \r\n\r\n
            var limit = max(0, total - 4)
            for i in range(limit, total - 3):
                if buf[i]==13 and buf[i+1]==10 and buf[i+2]==13 and buf[i+3]==10:
                    header_end = i + 4
                    content_length = _parse_content_length(buf, header_end)
                    break
        if header_end > 0 and (total - header_end) >= content_length:
            break
    return (total, header_end, content_length)

# ─── Envio de respostas ────────────────────────────────────────────────────────

fn _send_json(fd: Int32, status: Int, body: String):
    var reason = String("OK")
    if status == 400: reason = "Bad Request"
    elif status == 404: reason = "Not Found"
    elif status == 500: reason = "Internal Server Error"
    var hdr = "HTTP/1.1 " + String(status) + " " + reason + "\r\n"
    hdr += "Content-Type: application/json; charset=utf-8\r\n"
    hdr += "Access-Control-Allow-Origin: *\r\n"
    hdr += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    hdr += "Access-Control-Allow-Headers: Content-Type\r\n"
    hdr += "Content-Length: " + String(len(body)) + "\r\n"
    hdr += "Connection: close\r\n\r\n"
    var full = hdr + body
    var b = full.as_bytes()
    _ = external_call["send", Int](fd, b.unsafe_ptr(), len(b), Int32(0))

fn _send_cors(fd: Int32):
    var r = "HTTP/1.1 204 No Content\r\n"
    r += "Access-Control-Allow-Origin: *\r\n"
    r += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    r += "Access-Control-Allow-Headers: Content-Type\r\n"
    r += "Content-Length: 0\r\nConnection: close\r\n\r\n"
    var b = r.as_bytes()
    _ = external_call["send", Int](fd, b.unsafe_ptr(), len(b), Int32(0))

# ─── Construtores JSON ─────────────────────────────────────────────────────────

fn _jf(v: Float32) -> String:
    return String(Float64(v))

fn _json_boxes(boxes: List[List[Int]]) -> String:
    var s = String("[")
    for i in range(len(boxes)):
        if i > 0: s += ","
        var b = boxes[i]
        s += "[" + String(b[0]) + "," + String(b[1]) + "," + String(b[2]) + "," + String(b[3]) + "]"
    return s + "]"

fn _json_emb(emb: List[Float32]) -> String:
    var s = String("[")
    for i in range(len(emb)):
        if i > 0: s += ","
        s += _jf(emb[i])
    return s + "]"

fn _json_strs(names: List[String]) -> String:
    var s = String("[")
    for i in range(len(names)):
        if i > 0: s += ","
        s += "\"" + names[i] + "\""
    return s + "]"

# ─── Escrita de arquivo BMP temporário ─────────────────────────────────────────

fn _write_tmp(buf: UnsafePointer[UInt8], n: Int, path: String) -> Bool:
    try:
        var f = open(path, "wb")
        var CHUNK = 65536
        var off = 0
        while off < n:
            var cs = min(CHUNK, n - off)
            var b = List[UInt8](capacity=cs)
            for i in range(cs):
                b.append(buf[off + i])
            f.write_bytes(b)
            off += cs
        f.close()
        return True
    except _:
        return False

fn _del_tmp(path: String):
    try: os.remove(path)
    except _: pass

# ─── Handlers ─────────────────────────────────────────────────────────────────

fn _h_health(fd: Int32):
    _send_json(fd, 200, "{\"status\":\"ok\",\"versao\":\"" + VERSION + "\"}")

fn _h_detectar(fd: Int32, bp: UnsafePointer[UInt8], bn: Int,
               mut retina: retina_mod.RetinaFace, retina_ok: Bool):
    if not retina_ok or bn == 0:
        _send_json(fd, 200, "{\"boxes\":[]}")
        return
    if not _write_tmp(bp, bn, TMP_A):
        _send_json(fd, 400, "{\"erro\":\"BMP invalido\"}")
        return
    var boxes = List[List[Int]]()
    try:
        var bmp = dados_pkg.carregar_bmp_rgb(TMP_A)
        if bmp.width > 0:
            boxes = retina.inferir(bmp.pixels, 320, 16)
    except _: pass
    _del_tmp(TMP_A)
    _send_json(fd, 200, "{\"boxes\":" + _json_boxes(boxes) + "}")

fn _h_identificar(fd: Int32, bp: UnsafePointer[UInt8], bn: Int,
                  mut arcface: arc_model.ArcFace, galeria: infer_pkg.Galeria,
                  mut retina: retina_mod.RetinaFace, retina_ok: Bool):
    if bn == 0:
        _send_json(fd, 400, "{\"erro\":\"body vazio\"}")
        return
    if not _write_tmp(bp, bn, TMP_A):
        _send_json(fd, 400, "{\"erro\":\"BMP invalido\"}")
        return
    var nome  = String("desconhecido")
    var score = Float32(0.0)
    var bx    = List[Int](); bx.append(0); bx.append(0); bx.append(0); bx.append(0)
    try:
        var bmp = dados_pkg.carregar_bmp_rgb(TMP_A)
        if bmp.width > 0:
            var x0 = 0; var y0 = 0
            var x1 = bmp.width - 1; var y1 = bmp.height - 1
            if retina_ok:
                var boxes = retina.inferir(bmp.pixels, 320, 4)
                if len(boxes) > 0:
                    var b = boxes[0]
                    x0 = b[0]; y0 = b[1]; x1 = b[2]; y1 = b[3]
                    bx[0]=x0; bx[1]=y0; bx[2]=x1; bx[3]=y1
            var emb = arcface.embed_caixa(bmp.pixels, x0, y0, x1, y1)
            if len(emb) > 0 and galeria.tamanho() > 0:
                var best_sim = Float32(-1.0)
                var best_idx = -1
                for i in range(galeria.tamanho()):
                    var sim = arcface.similaridade(emb, galeria.embeddings[i])
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = i
                score = best_sim
                if best_idx >= 0 and best_sim >= 0.5:
                    nome = galeria.nomes[best_idx]
    except _: pass
    _del_tmp(TMP_A)
    var resp = "{\"identidade\":\"" + nome + "\",\"score\":" + _jf(score)
    resp += ",\"box\":[" + String(bx[0]) + "," + String(bx[1]) + "," + String(bx[2]) + "," + String(bx[3]) + "]}"
    _send_json(fd, 200, resp)

fn _h_verificar_par(fd: Int32, bp: UnsafePointer[UInt8], bn: Int,
                    mut arcface: arc_model.ArcFace):
    if bn < 8:
        _send_json(fd, 400, "{\"erro\":\"body muito curto — esperado uint32_LE(tam_A)+BMP_A+BMP_B\"}")
        return
    var len_a = _u32le(bp, 0)
    if len_a <= 0 or Int(len_a) + 4 >= bn:
        _send_json(fd, 400, "{\"erro\":\"tam_A invalido\"}")
        return
    var len_b = bn - 4 - Int(len_a)
    if len_b <= 0:
        _send_json(fd, 400, "{\"erro\":\"imagem B ausente\"}")
        return
    if not _write_tmp(bp + 4, Int(len_a), TMP_A) or not _write_tmp(bp + 4 + Int(len_a), len_b, TMP_B):
        _send_json(fd, 400, "{\"erro\":\"BMP invalido\"}")
        return
    var sim_out = Float32(0.0)
    var mesma   = False
    try:
        mesma = infer_pkg.verificar(arcface, TMP_A, TMP_B, sim_out, 0.6)
    except _: pass
    _del_tmp(TMP_A); _del_tmp(TMP_B)
    var ms = "true" if mesma else "false"
    _send_json(fd, 200, "{\"mesma_pessoa\":" + ms + ",\"score\":" + _jf(sim_out) + "}")

fn _h_embedding(fd: Int32, bp: UnsafePointer[UInt8], bn: Int,
                mut arcface: arc_model.ArcFace):
    if bn == 0:
        _send_json(fd, 400, "{\"erro\":\"body vazio\"}")
        return
    if not _write_tmp(bp, bn, TMP_A):
        _send_json(fd, 400, "{\"erro\":\"BMP invalido\"}")
        return
    var emb = List[Float32]()
    try:
        emb = infer_pkg.embed_arquivo(arcface, TMP_A)
    except _: pass
    _del_tmp(TMP_A)
    _send_json(fd, 200, "{\"embedding\":" + _json_emb(emb) + "}")

fn _h_galeria_get(fd: Int32, galeria: infer_pkg.Galeria):
    _send_json(fd, 200, "{\"identidades\":" + _json_strs(galeria.nomes) + ",\"total\":" + String(galeria.tamanho()) + "}")

fn _h_galeria_construir(fd: Int32, mut arcface: arc_model.ArcFace,
                         mut galeria: infer_pkg.Galeria, dataset_dir: String):
    try:
        galeria = infer_pkg.construir_galeria(arcface, dataset_dir)
    except _: pass
    _send_json(fd, 200, "{\"ok\":true,\"classes\":" + String(galeria.tamanho()) + "}")

fn _h_galeria_adicionar(fd: Int32, bp: UnsafePointer[UInt8], bn: Int,
                         mut arcface: arc_model.ArcFace, mut galeria: infer_pkg.Galeria):
    if bn < 2:
        _send_json(fd, 400, "{\"erro\":\"body muito curto\"}")
        return
    var name_len = Int(bp[0])
    if name_len <= 0 or name_len >= bn - 1:
        _send_json(fd, 400, "{\"erro\":\"name_len invalido\"}")
        return
    var nome    = _to_str(bp + 1, name_len)
    var bmp_off = 1 + name_len
    var bmp_len = bn - bmp_off
    if not _write_tmp(bp + bmp_off, bmp_len, TMP_A):
        _send_json(fd, 400, "{\"erro\":\"BMP invalido\"}")
        return
    try:
        var emb = infer_pkg.embed_arquivo(arcface, TMP_A)
        if len(emb) > 0:
            galeria.adicionar(nome, emb^)
    except _: pass
    _del_tmp(TMP_A)
    _send_json(fd, 200, "{\"ok\":true,\"galeria_size\":" + String(galeria.tamanho()) + "}")

# ─── Ponto de entrada ─────────────────────────────────────────────────────────

fn main() raises:
    var port        = 8080
    var arcface_dir = os.path.join("MODELO", "arcface_modelo")
    var retina_dir  = os.path.join("MODELO", "retina_modelo")
    var dataset_dir = "DATASET"

    # Aceita porta como primeiro argumento: ./bionix_api 9090
    try:
        from sys import argv
        var args = argv()
        if len(args) > 1:
            port = Int(Float64(uteis.parse_float_ascii(args[1])))
        if len(args) > 2:
            dataset_dir = String(args[2])
    except _: pass

    print("┌─ Bionix API v" + VERSION + " ─────────────────────────────────────┐")
    print("│  Porta     : " + String(port))
    print("│  ArcFace   : " + arcface_dir)
    print("│  RetinFace : " + retina_dir)
    print("│  Dataset   : " + dataset_dir)
    print("└─────────────────────────────────────────────────────────────────┘")

    # Carrega ArcFace
    var arc_p   = arc_model.ArcFaceParametros(64, 8, 3, 3, 128, 0, "cpu")
    var arcface = arc_model.ArcFace(arc_p^, arcface_dir)
    var arc_ok  = arcface.carregar()
    print("  ArcFace carregado :", arc_ok)

    # Carrega RetinaFace (opcional — detecção de rosto)
    var ret_p   = retina_mod.BlocoRetinaFaceParametros(320, 6, 3, 3, "cpu", 32, 16, 0.01, 0.5)
    var retina  = retina_mod.RetinaFace(ret_p^, retina_dir)
    var ret_ok  = False
    try:
        ret_ok = retina.carregar_workspace(retina_dir)
    except _: pass
    print("  RetinaFace carregado:", ret_ok)

    # Constrói galeria inicial a partir do dataset
    var galeria = infer_pkg.Galeria()
    if arc_ok:
        try:
            galeria = infer_pkg.construir_galeria(arcface, dataset_dir)
            print("  Galeria         :", galeria.tamanho(), "identidades")
        except _: pass

    var sfd = _tcp_listen(port)
    print("Servidor iniciado → http://0.0.0.0:" + String(port))
    print("Rotas disponíveis:")
    print("  GET  /health")
    print("  POST /detectar | /identificar | /verificar_par | /embedding")
    print("  GET  /galeria")
    print("  POST /galeria/construir | /galeria/adicionar")

    var buf = UnsafePointer[UInt8].alloc(MAX_RECV)

    while True:
        var cfd = _tcp_accept(sfd)
        if cfd < 0:
            continue

        var rr      = _recv_http(cfd, buf)
        var total   = rr[0]
        var hdr_end = rr[1]
        var body_cl = rr[2]

        if total == 0 or hdr_end < 0:
            _tcp_close(cfd)
            continue

        # Encontra primeira linha: METHOD SP PATH SP HTTP/...
        var sp1  = _find_byte(buf, 0, min(total, 20), 32)   # primeiro espaço
        var lf1  = _find_byte(buf, 0, min(total, 512), 13)  # CR da primeira linha
        var sp2  = _find_byte(buf, sp1 + 1, min(total, 512), 32) if sp1 > 0 else -1

        var body_ptr = buf + hdr_end
        var body_len = min(body_cl, total - hdr_end)

        # Roteamento por comparação de bytes
        if _eq_at(buf, total, 0, "OPTIONS"):
            _send_cors(cfd)
        elif sp2 > sp1 and sp1 > 0:
            var path_off = sp1 + 1
            var path_len = sp2 - sp1 - 1
            var is_get  = _eq_at(buf, total, 0, "GET")
            var is_post = _eq_at(buf, total, 0, "POST")
            if _eq_at(buf, total, path_off, "/health"):
                _h_health(cfd)
            elif is_post and _eq_at(buf, total, path_off, "/detectar"):
                _h_detectar(cfd, body_ptr, body_len, retina, ret_ok)
            elif is_post and _eq_at(buf, total, path_off, "/identificar"):
                _h_identificar(cfd, body_ptr, body_len, arcface, galeria, retina, ret_ok)
            elif is_post and _eq_at(buf, total, path_off, "/verificar_par"):
                _h_verificar_par(cfd, body_ptr, body_len, arcface)
            elif is_post and _eq_at(buf, total, path_off, "/embedding"):
                _h_embedding(cfd, body_ptr, body_len, arcface)
            elif is_get and _eq_at(buf, total, path_off, "/galeria"):
                _h_galeria_get(cfd, galeria)
            elif is_post and _eq_at(buf, total, path_off, "/galeria/construir"):
                _h_galeria_construir(cfd, arcface, galeria, dataset_dir)
            elif is_post and _eq_at(buf, total, path_off, "/galeria/adicionar"):
                _h_galeria_adicionar(cfd, body_ptr, body_len, arcface, galeria)
            else:
                var path_str = _to_str(buf + path_off, path_len)
                _send_json(cfd, 404, "{\"erro\":\"rota nao encontrada\",\"path\":\"" + path_str + "\"}")
        else:
            _send_json(cfd, 400, "{\"erro\":\"request malformado\"}")

        _tcp_close(cfd)
