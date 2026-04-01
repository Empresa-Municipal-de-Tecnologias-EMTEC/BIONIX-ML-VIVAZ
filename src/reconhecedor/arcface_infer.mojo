import reconhecedor.arcface_model as arc_model
import bionix_ml.dados as dados_pkg
import bionix_ml.graficos as graficos_pkg
import os


# ─── Galeria ────────────────────────────────────────────────────────────────
# Armazena pares (nome de identidade, vetor de embedding) em listas paralelas.
struct Galeria(Movable):
    var nomes:      List[String]
    var embeddings: List[List[Float32]]

    fn __init__(out self):
        self.nomes      = List[String]()
        self.embeddings = List[List[Float32]]()

    fn adicionar(mut self, var nome: String, var emb: List[Float32]):
        self.nomes.append(nome^)
        self.embeddings.append(emb^)

    fn tamanho(self) -> Int:
        return len(self.nomes)


# ─── Funções de inferência ───────────────────────────────────────────────────

# Calcula embedding de uma imagem BMP dado o recorte.
# Se x1 == 0 usa a imagem inteira.
fn embed_arquivo(mut modelo: arc_model.ArcFace,
                 var img_path: String,
                 var x0: Int = 0, var y0: Int = 0,
                 var x1: Int = 0, var y1: Int = 0) raises -> List[Float32]:
    var bmp = dados_pkg.carregar_bmp_rgb(img_path)
    if bmp.width == 0:
        return List[Float32]()
    var ax0 = x0; var ay0 = y0
    var ax1 = x1 if x1 > 0 else bmp.width  - 1
    var ay1 = y1 if y1 > 0 else bmp.height - 1
    var ps = modelo.parametros.patch_size
    var patch = graficos_pkg.crop_and_resize_rgb(bmp.pixels, ax0, ay0, ax1, ay1, ps, ps)
    return modelo.embed(patch^)


# Verifica se duas imagens pertencem à mesma pessoa.
# Preenche `sim_out` com a similaridade calculada; retorna True se >= threshold.
fn verificar(mut modelo: arc_model.ArcFace,
             var img_a: String, var img_b: String,
             mut sim_out: Float32,
             var threshold: Float32 = 0.6) raises -> Bool:
    var emb_a = embed_arquivo(modelo, img_a)
    var emb_b = embed_arquivo(modelo, img_b)
    if len(emb_a) == 0 or len(emb_b) == 0:
        sim_out = Float32(0.0)
        return False
    sim_out = modelo.similaridade(emb_a, emb_b)
    return sim_out >= threshold


# Identifica a identidade mais próxima numa Galeria.
# Preenche `sim_out` com a melhor similaridade e retorna o nome.
# Se nenhuma identidade superar `threshold` retorna "desconhecido".
fn identificar(mut modelo: arc_model.ArcFace,
               var img_path: String,
               galeria: Galeria,
               mut nome_out: String,
               mut sim_out: Float32,
               var threshold: Float32 = 0.5) raises:
    var emb = embed_arquivo(modelo, img_path)
    if len(emb) == 0:
        nome_out = "desconhecido"
        sim_out  = Float32(0.0)
        return
    var best_sim:  Float32 = -1.0
    var best_idx:  Int     = -1
    for i in range(galeria.tamanho()):
        var sim = modelo.similaridade(emb, galeria.embeddings[i])
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    sim_out = best_sim
    if best_idx >= 0 and best_sim >= threshold:
        nome_out = galeria.nomes[best_idx]
    else:
        nome_out = "desconhecido"


# Constrói uma Galeria a partir do dataset de treino.
# Para cada identidade usa a primeira imagem disponível.
fn construir_galeria(mut modelo: arc_model.ArcFace,
                     var dataset_dir: String) raises -> Galeria:
    var galeria = Galeria()
    var train_root = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_root):
        train_root = dataset_dir
    try:
        for cls in os.listdir(train_root):
            var pcls = os.path.join(train_root, cls)
            if not os.path.isdir(pcls):
                continue
            var found = False
            try:
                for f in os.listdir(pcls):
                    if not found and f.endswith('.bmp'):
                        var emb = embed_arquivo(modelo, os.path.join(pcls, f))
                        if len(emb) > 0:
                            galeria.adicionar(cls, emb^)
                        found = True
            except _:
                pass
    except _:
        pass
    return galeria^
