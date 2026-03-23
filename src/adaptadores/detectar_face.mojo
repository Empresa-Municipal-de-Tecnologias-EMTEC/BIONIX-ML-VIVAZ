# Adapter de detecção/alinhamento simples para desenvolvimento em vivaz
# Implementação provisória: não faz detecção real — retorna o BMPInfo original
# ou carrega o BMP quando recebe um caminho. Substituir por detector real depois.

import bionix_ml.dados as dados_pkg
import bionix_ml.dados.bmp as bmpmod


def _center_crop_bmpinfo(var info: bmpmod.BMPInfo, frac: Float32 = 0.7) -> bmpmod.BMPInfo:
    # Retorna um novo BMPInfo que é um crop central aproximado.
    try:
        var w = info.width
        var h = info.height
        if w <= 0 or h <= 0:
            return info^

        var short = Int(0)
        if w < h:
            short = w
        else:
            short = h
        var crop_size = Int(Float32(short) * frac)
        if crop_size <= 0:
            return info^

        var cx = w // 2
        var cy = h // 2
        var half = crop_size // 2
        var x0 = cx - half
        var y0 = cy - half
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        var x1 = x0 + crop_size
        var y1 = y0 + crop_size
        if x1 > w:
            x1 = w
            var tmpx = w - crop_size
            if tmpx > 0:
                x0 = tmpx
            else:
                x0 = 0
        if y1 > h:
            y1 = h
            var tmpy = h - crop_size
            if tmpy > 0:
                y0 = tmpy
            else:
                y0 = 0

        # Extrai pixels RGB se presente
        var new_pixels = List[List[List[Float32]]]()
        if len(info.pixels) > 0:
            for ry in range(y0, y1):
                var row = List[List[Float32]]()
                for rx in range(x0, x1):
                    row.append(info.pixels[ry][rx])
                new_pixels.append(row)

        # Extrai grayscale se presente
        var new_gray = List[List[Float32]]()
        if len(info.grayscale) > 0:
            for ry in range(y0, y1):
                var rowg = List[Float32]()
                for rx in range(x0, x1):
                    rowg.append(info.grayscale[ry][rx])
                new_gray.append(rowg)

        # Extrai preto_branco se presente
        var new_pb = List[List[Float32]]()
        if len(info.preto_branco) > 0:
            for ry in range(y0, y1):
                var rowpb = List[Float32]()
                for rx in range(x0, x1):
                    rowpb.append(info.preto_branco[ry][rx])
                new_pb.append(rowpb)

        # Cropping logic disabled in this adapter to avoid referencing
        # `BMPInfo` constructor directly; return the original info as fallback.
        return info^
    except Exception:
        print("[adaptadores.detectar_face] crop falhou")
        return info^


def detect_and_align_bbox(path:String)->bmpmod.BMPInfo, List[Int]:
    # Similar a detect_and_align, mas retorna o bbox relativo à imagem original
    try:
        var info = bmpmod.zero_bmp()
        try:
            info = dados_pkg.carregar_bmp_rgb(path)^
        except Exception:
            return bmpmod.zero_bmp(), List[Int]()

        if info.width <= 0 or info.height <= 0:
            return bmpmod.zero_bmp(), List[Int]()

        var w = info.width
        var h = info.height
        if w <= 0 or h <= 0:
            return info^, List[Int]()

        var short = w if w < h else h
        if short <= 64:
            # retorna bbox inteira
            return info^, [0, 0, w, h]

        var crop_size = Int(Float32(short) * Float32(0.7))
        if crop_size <= 0:
            return info^, [0, 0, w, h]

        var cx = w // 2
        var cy = h // 2
        var half = crop_size // 2
        var x0 = cx - half
        var y0 = cy - half
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        var x1 = x0 + crop_size
        var y1 = y0 + crop_size
        if x1 > w:
            x1 = w
            var tmpx2 = w - crop_size
            if tmpx2 > 0:
                x0 = tmpx2
            else:
                x0 = 0
        if y1 > h:
            y1 = h
            var tmpy2 = h - crop_size
            if tmpy2 > 0:
                y0 = tmpy2
            else:
                y0 = 0

        var cropped = _center_crop_bmpinfo(info^, 0.7)^
        return cropped^, [x0, y0, x1, y1]
    except Exception:
        print("[adaptadores.detectar_face] detect_and_align_bbox falhou")
        return bmpmod.zero_bmp(), List[Int]()


def detect_and_align(path:String) -> bmpmod.BMPInfo:
    # Heurística rápida: carregar BMP (se caminho) e retornar crop central
    try:
        var info = bmpmod.zero_bmp()
        try:
            info = dados_pkg.carregar_bmp_rgb(path)^
        except Exception:
            return bmpmod.zero_bmp()

        if info.width <= 0 or info.height <= 0:
            return bmpmod.zero_bmp()

        # Se já for pequeno, retorna sem alteração
        var short = Int(0)
        if info.width < info.height:
            short = info.width
        else:
            short = info.height
        if short <= 64:
            return info^

        # Aplicar crop central como proxy de detecção/alinhamento
        var cropped = _center_crop_bmpinfo(info^, 0.7)^
        return cropped^
    except Exception:
        print("[adaptadores.detectar_face] falha simples")
        return bmpmod.zero_bmp()
