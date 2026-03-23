# Pré-processamento: carregar BMP, detectar/alinhar face, redimensionar preservando aspecto
# Integra com utilitários do BIONIX-ML quando disponíveis. Tem comportamento seguro
# de fallback: usa a imagem inteira e redimensiona pelo menor lado.

import bionix_ml.dados as dados_pkg
import adaptadores.detectar_face as det
import bionix_ml.dados.bmp as bmpmod
 


def _has_detector_adapter():
    try:
        return True
    except Exception:
        try:
            return True
        except Exception:
            return False


def detect_and_align(image_path: String) -> bmpmod.BMPInfo:
    # estiver disponível — o caller usará fallback de redimensionamento.
    try:
        return det.detect_and_align(image_path)
    except Exception:
        return bmpmod.zero_bmp()


# Note: resize function inlined in `load_and_preprocess` to avoid requiring
# a typed BMPInfo parameter in function signature (parser requires explicit arg types).


def load_and_preprocess(path: String, target_short_side: Int) -> bmpmod.BMPInfo:
    # Carrega, tenta detectar/alinhar e redimensiona. Retorna o objeto carregado (BMP)
    #BMP Zerado
    var img_info = bmpmod.BMPInfo(0, 0, 0, 0, bmpmod.BMP_MODO_GRAYSCALE, List[List[List[Float32]]](), List[List[Float32]](), List[List[Float32]]())^
    try:
        img_info = dados_pkg.carregar_bmp_rgb(path)^
    except Exception:
        print("[pre_processamento] falha ao carregar imagem")
        return bmpmod.zero_bmp()
    var face = detect_and_align(path)^
    if face.width <= 0 or face.height <= 0:
        # Falha de detecção / adaptador ausente: usamos a imagem inteira e redimensionamos
        try:
            var w = img_info.width
            var h = img_info.height
            try:
                if w <= 0 or h <= 0:
                    return img_info^
            except Exception:
                return img_info^
            if w < h:
                new_w = target_short_side
                new_h = Int((h * target_short_side) / w)
            else:
                new_h = target_short_side
                new_w = Int((w * target_short_side) / h)
            try:
                if len(path) > 0:
                    return dados_pkg.carregar_bmp_rgb(path, new_h, new_w)^
            except Exception:
                pass
            return img_info^
        except Exception:
            print("[pre_processamento] falha no resize")
            return img_info^
    else:
        # Se detect_and_align retornou um BMP (adapter loaded), redimensiona o crop quando possível
        try:
            var w = face.width
            var h = face.height
            try:
                if w <= 0 or h <= 0:
                    return face^
            except Exception:
                return face^
            if w < h:
                new_w = target_short_side
                new_h = Int((h * target_short_side) / w)
            else:
                new_h = target_short_side
                new_w = Int((w * target_short_side) / h)
            # No reliable source path for the crop; return face as-is
            return face^
        except Exception:
            print("[pre_processamento] falha no resize do crop")
            return face^
