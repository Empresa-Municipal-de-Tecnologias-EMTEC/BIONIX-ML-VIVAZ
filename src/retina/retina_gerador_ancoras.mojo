import bionix_ml.nucleo.Tensor as tensor_defs
import math

# detector de NaN/Inf ou valores absurdos
fn _verificar_valor(var v: Float32) -> Bool:
    try:
        if v != v:
            return False
        if v < -1e12 or v > 1e12:
            return False
        return True
    except _:
        return False

fn gerar_ancoras(tamanho_entrada: Int = 640, passos: List[Int] = [8,16,32],
                 escalas: List[List[Float32]] = List[List[Float32]](), multiplicadores: List[Float32] = [4,8,12], proporcoes: List[Float32] = [0.5, 1.0, 1.5]) -> List[List[Float32]]:
    # Gera anchors como [cx, cy, w, h] em pixels para cada nível.
    var out: List[List[Float32]] = List[List[Float32]]()
    var escalas_local: List[List[Float32]] = escalas.copy()
    if len(escalas_local) == 0:
        var default_scales: List[List[Float32]] = List[List[Float32]]()
        for s in passos:
            var lvl: List[Float32] = List[Float32]()
            for m in multiplicadores:
                lvl.append(Float32(s) * Float32(m))
            default_scales.append(lvl^)
        escalas_local = default_scales^

    for idx in range(len(passos)):
        var stride = passos[idx]
        var lvl_scales = escalas_local[idx].copy()
        var feat_size = tamanho_entrada // stride
        for i in range(feat_size):
            var cy = Float32((i + 0.5) * stride)
            for j in range(feat_size):
                var cx = Float32((j + 0.5) * stride)
                for sc in lvl_scales:
                    for r in proporcoes:
                        var w = sc * Float32(math.sqrt(r))
                        var h = sc / Float32(math.sqrt(r))
                        var a: List[Float32] = List[Float32]()
                        a.append(cx); a.append(cy); a.append(w); a.append(h)
                        # verifica se os valores são numéricos e razoáveis antes de adicionar-los à lista final, para evitar propagação de âncoras corrompidas
                        var ok = True
                        for vv in a:
                            if not _verificar_valor(vv):
                                ok = False
                                break
                        if not ok:
                            try:
                                print("[DBG] gerar_ancoras: ignorando ancora corrompida no passo", stride, "cx,cy,w,h=", cx, cy, w, h)
                            except _:
                                print("[DBG] gerar_ancoras: ignorando ancora corrompida (impossível pormatar os valores)")
                            continue
                        out.append(a^)
    escalas_local = escalas_local^
    return out^


fn gerar_ancoras_por_nivel(tamanho_entrada: Int = 640, passos: List[Int] = [8,16,32],
                           escalas: List[List[Float32]] = List[List[Float32]](), multiplicadores: List[Float32] = [4,8,12], proporcoes: List[Float32] = [0.5, 1.0, 1.5]) -> List[List[List[Float32]]]:
    # Returns anchors grouped by level: List[level] -> List[anchors] where anchor=[cx,cy,w,h]
    var out_levels: List[List[List[Float32]]] = List[List[List[Float32]]]()
    var escalas_local: List[List[Float32]] = escalas.copy()
    if len(escalas_local) == 0:
        var default_scales: List[List[Float32]] = List[List[Float32]]()
        for s in passos:
            var lvl: List[Float32] = List[Float32]()
            for m in multiplicadores:
                lvl.append(Float32(s) * Float32(m))
            default_scales.append(lvl^)
        escalas_local = default_scales^

    for idx in range(len(passos)):
        var stride = passos[idx]
        var lvl_scales = escalas_local[idx].copy()
        var feat_size = tamanho_entrada // stride
        var level_list: List[List[Float32]] = List[List[Float32]]()
        for i in range(feat_size):
            var cy = Float32((i + 0.5) * stride)
            for j in range(feat_size):
                var cx = Float32((j + 0.5) * stride)
                for sc in lvl_scales:
                    for r in proporcoes:
                        var w = sc * Float32(math.sqrt(r))
                        var h = sc / Float32(math.sqrt(r))
                        var a: List[Float32] = List[Float32]()
                        a.append(cx); a.append(cy); a.append(w); a.append(h)
                        var ok = True
                        for vv in a:
                            if not _verificar_valor(vv):
                                ok = False
                                break
                        if not ok:
                            continue
                        level_list.append(a^)
        out_levels.append(level_list^)
    return out_levels^
