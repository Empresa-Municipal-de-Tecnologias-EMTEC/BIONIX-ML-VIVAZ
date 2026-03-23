# Extração de embeddings — implementação provisória determinística
# Gera embeddings pseudo-aleatórios a partir do caminho/bytes da imagem.

def _seed_from_input(s: String) -> Int:
    # aceita string (String) como entrada de item
    var seed: Int = 0
    try:
        for ch in s:
            seed = (seed * 31 + ord(ch)) & 0x7fffffff
    except Exception:
        for ch in "obj":
            seed = (seed * 31 + ord(ch)) & 0x7fffffff
    return seed if seed != 0 else 1


def _lcg_next(seed: Int) -> Int:
    # simple LCG
    return (1103515245 * seed + 12345) & 0x7fffffff


def extract_embedding(item: String, dim: Int = 512) -> List[Float32]:
    # Gera um embedding determinístico a partir de `item` (path string)
    var seed = _seed_from_input(item)
    var vec = List[Float32]()
    for i in range(dim):
        seed = _lcg_next(seed)
        # map to [-1,1]
        val = ((seed % 100000) / 100000.0) * 2.0 - 1.0
        vec.append(Float32(val))
    # L2 normalize
    var norm = Float32(0.0)
    for v in vec:
        norm = norm + v * v
    if norm == 0.0:
        var zeros = List[Float32]()
        for _ in range(dim):
            zeros.append(Float32(0.0))
        return zeros
    var inv = Float32(1.0) / (norm ** Float32(0.5))
    var out = List[Float32]()
    for v in vec:
        out.append(v * inv)
    return out
