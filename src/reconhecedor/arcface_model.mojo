import bionix_ml.camadas.cnn as cnn_pkg
import bionix_ml.camadas as camadas_pkg
import bionix_ml.camadas.cnn.cnn as cnn_impl
import bionix_ml.nucleo.Tensor as tensor_defs
import bionix_ml.dados.arquivo as arquivo_pkg
import bionix_ml.dados as dados_pkg
import bionix_ml.uteis as uteis
import bionix_ml.computacao.dispatcher_tensor as dispatcher
import bionix_ml.graficos as graficos_pkg
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import bionix_ml.computacao.sessao as sessao_driver
import bionix_ml.computacao.storage_sessao as storage_sessao
import math
import os


# ─── Parâmetros ────────────────────────────────────────────────────────────────

struct ArcFaceParametros(Movable, Copyable):
    # Lado do patch de entrada (quadrado, escala de cinza) — deve ser >= kernel_h+1
    var patch_size: Int
    # Número de filtros do BlocoCNN
    var num_filtros: Int
    # Tamanho do kernel convolucional
    var kernel_h: Int
    var kernel_w: Int
    # Dimensão do vetor de embedding (saída do modelo)
    var embed_dim: Int
    # Número de identidades conhecidas (usado na cabeça de classificação durante treino)
    var num_classes: Int
    # Tipo de backend de computação ("cpu", "cuda", ...)
    var tipo_ctx: String

    fn __init__(out self,
                var patch_size_in: Int = 64,
                var num_filtros_in: Int = 8,
                var kernel_h_in: Int = 3,
                var kernel_w_in: Int = 3,
                var embed_dim_in: Int = 128,
                var num_classes_in: Int = 0,
                var tipo_ctx_in: String = "cpu"):
        self.patch_size   = patch_size_in
        self.num_filtros  = num_filtros_in
        self.kernel_h     = kernel_h_in
        self.kernel_w     = kernel_w_in
        self.embed_dim    = embed_dim_in
        self.num_classes  = num_classes_in
        self.tipo_ctx     = tipo_ctx_in^


# ─── Modelo ────────────────────────────────────────────────────────────────────

# ArcFace encapsula tudo necessário para calcular embeddings faciais normalizados.
#
# Fluxo:
#   Imagem RGB → crop & resize (patch_size) → tons de cinza
#     → BlocoCNN → feature vector (D)
#     → Linear [D → embed_dim] → L2-normalização → embedding (embed_dim,)
#
# Durante o treino uma cabeça de classificação [embed_dim → num_classes] é usada
# para calcular a perda de entropia cruzada com margem cosseno (ArcFace-lite).
#
# Arquivos salvos em `diretorio_modelo/`:
#   cnn_kernel_*.tensor.txt  — kernels da convolução (via storage_sessao)
#   proj_peso.bin / proj_bias.bin — camada de projeção D→embed_dim
#   cls_peso.bin  / cls_bias.bin  — cabeça classificadora (apenas treino)
#   arcface_state.txt               — estado de treino (época, lr)
struct ArcFace(Movable):
    var bloco_cnn:     cnn_pkg.BlocoCNN
    var bloco_kernels:  List[tensor_defs.Tensor]
    var bloco_peso_saida: tensor_defs.Tensor
    var bloco_bias_saida: tensor_defs.Tensor
    var tipo_computacao: String
    var proj_peso:     tensor_defs.Tensor   # [D, embed_dim]
    var proj_bias:     tensor_defs.Tensor   # [1, embed_dim]
    var cls_peso:      tensor_defs.Tensor   # [embed_dim, num_classes]
    var cls_bias:      tensor_defs.Tensor   # [1, num_classes]
    var parametros:    ArcFaceParametros
    var diretorio_modelo: String
    var treinamento_epoca: Int
    var treinamento_lr: Float32
    var heads_inicializadas: Bool

    fn __init__(out self,
                var params_in: ArcFaceParametros = ArcFaceParametros(),
                var dir_modelo_in: String = "MODELO/arcface_modelo"):
        self.parametros = params_in^
        # Create BlocoCNN via factory to keep construction centralized
        try:
            import bionix_ml.camadas as camadas_pkg
            self.bloco_cnn = camadas_pkg.criar_bloco_cnn(self.parametros.patch_size, self.parametros.patch_size, self.parametros.num_filtros, self.parametros.kernel_h, self.parametros.kernel_w, self.parametros.tipo_ctx)
            try:
                self.bloco_cnn.contexto = contexto_defs.criar_contexto_padrao(self.parametros.tipo_ctx)
            except _:
                pass
        except _:
            # Fallback: use centralized factory to create BlocoCNN
            try:
                self.bloco_cnn = camadas_pkg.criar_bloco_cnn(
                    self.parametros.patch_size, self.parametros.patch_size,
                    self.parametros.num_filtros,
                    self.parametros.kernel_h, self.parametros.kernel_w,
                    self.parametros.tipo_ctx)
                try:
                    self.bloco_cnn.contexto = contexto_defs.criar_contexto_padrao(self.parametros.tipo_ctx)
                except _:
                    pass
            except _:
                # as a last resort, construct directly (rare)
                self.bloco_cnn = cnn_pkg.BlocoCNN(
                    self.parametros.patch_size, self.parametros.patch_size,
                    self.parametros.num_filtros,
                    self.parametros.kernel_h, self.parametros.kernel_w,
                    contexto_defs.criar_contexto_padrao(self.parametros.tipo_ctx))
        # populate tensor-backed fields for migration/adapters
        try:
            self.tipo_computacao = String(self.bloco_cnn.tipo_computacao)
        except _:
            self.tipo_computacao = String(self.parametros.tipo_ctx)
        self.bloco_kernels = List[tensor_defs.Tensor]()
        try:
            for k in self.bloco_cnn.kernels:
                try: self.bloco_kernels.append(k.copy())
                except _: continue
        except _:
            pass
        try:
            self.bloco_peso_saida = self.bloco_cnn.peso_saida.copy()
        except _:
            self.bloco_peso_saida = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
        try:
            self.bloco_bias_saida = self.bloco_cnn.bias_saida.copy()
        except _:
            self.bloco_bias_saida = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
        # placeholders; inicializados na primeira chamada de treino/embed
        self.proj_peso  = tensor_defs.Tensor(List[Int](), self.bloco_cnn.tipo_computacao)
        self.proj_bias  = tensor_defs.Tensor(List[Int](), self.bloco_cnn.tipo_computacao)
        self.cls_peso   = tensor_defs.Tensor(List[Int](), self.bloco_cnn.tipo_computacao)
        self.cls_bias   = tensor_defs.Tensor(List[Int](), self.bloco_cnn.tipo_computacao)
        self.diretorio_modelo  = dir_modelo_in^
        self.treinamento_epoca = -1
        self.treinamento_lr    = 0.0
        self.heads_inicializadas = False

    # Inicializa as cabeças de projeção e classificação para uma dada dimensão de feature.
    fn _init_heads(mut self, var feat_dim: Int):
        var E = self.parametros.embed_dim
        var C = self.parametros.num_classes

        var w_shape = List[Int](); w_shape.append(feat_dim); w_shape.append(E)
        self.proj_peso = tensor_defs.Tensor(w_shape^, self.bloco_cnn.tipo_computacao)
        var b_shape = List[Int](); b_shape.append(1); b_shape.append(E)
        self.proj_bias = tensor_defs.Tensor(b_shape^, self.bloco_cnn.tipo_computacao)

        # inicialização He-uniforme
        var seed = 198723
        var fan_in = Float32(feat_dim)
        var scale = Float32(math.sqrt(Float64(2.0 / fan_in)))
        for i in range(len(self.proj_peso.dados)):
            seed = (seed * 1664525 + 1013904223 + i) % 2147483647
            var u = Float32(seed) / Float32(2147483647)
            self.proj_peso.dados[i] = (u * 2.0 - 1.0) * scale
        for i in range(len(self.proj_bias.dados)):
            self.proj_bias.dados[i] = 0.0

        if C > 0:
            var cw_shape = List[Int](); cw_shape.append(E); cw_shape.append(C)
            self.cls_peso = tensor_defs.Tensor(cw_shape^, self.bloco_cnn.tipo_computacao)
            var cb_shape = List[Int](); cb_shape.append(1); cb_shape.append(C)
            self.cls_bias = tensor_defs.Tensor(cb_shape^, self.bloco_cnn.tipo_computacao)
            for i in range(len(self.cls_peso.dados)):
                seed = (seed * 1103515245 + 12345 + i) % 2147483647
                var u = Float32(seed) / Float32(2147483647)
                self.cls_peso.dados[i] = (u * 2.0 - 1.0) * 0.01
            for i in range(len(self.cls_bias.dados)):
                self.cls_bias.dados[i] = 0.0

        self.heads_inicializadas = True

    # Extrai embedding L2-normalizado de um patch RGB (lista [H][W] de [R,G,B] em [0,1]).
    # Retorna lista de `embed_dim` floats (norma ≈ 1).
    fn embed(mut self, var patch_rgb: List[List[List[Float32]]]) raises -> List[Float32]:
        var ps = self.parametros.patch_size

        # Grayscale tensor [1, ps*ps]
        var in_shape = List[Int](); in_shape.append(1); in_shape.append(ps * ps)
        var tensor_in = tensor_defs.Tensor(in_shape^, self.bloco_cnn.tipo_computacao)
        for yy in range(ps):
            for xx in range(ps):
                var pix = patch_rgb[yy][xx].copy()
                var gray = Float32(0.299) * pix[0] + Float32(0.587) * pix[1] + Float32(0.114) * pix[2]
                tensor_in.dados[yy * ps + xx] = gray

        # BlocoCNN features [1, D]
        # Use tensor-backed temporary BlocoCNN created from stored tensors to allow migration
        var tmp_bloco = camadas_pkg.criar_bloco_de_tensores(
            ps, ps, self.parametros.num_filtros, self.parametros.kernel_h, self.parametros.kernel_w,
            self.bloco_kernels.copy(), self.bloco_peso_saida.copy(), self.bloco_bias_saida.copy(), self.tipo_computacao)
        var feats = cnn_pkg.extrair_features(tmp_bloco, tensor_in)
        var D = feats.formato[1]

        if not self.heads_inicializadas or len(self.proj_peso.dados) == 0:
            self._init_heads(D)

        # Projeção linear [1,D] × [D,E] → [1,E]
        var proj = List[Float32]()
        var E = self.parametros.embed_dim
        for e in range(E):
            var s: Float32 = self.proj_bias.dados[e]
            for d in range(D):
                s = s + feats.dados[d] * self.proj_peso.dados[d * E + e]
            proj.append(s)

        # L2-normalização
        var norm: Float32 = 1e-8
        for v in proj:
            norm = norm + v * v
        norm = Float32(math.sqrt(Float64(norm)))
        var out = List[Float32]()
        for v in proj:
            out.append(v / norm)
        return out^

    # Conveniência: recebe imagem RGB completa (pixels[y][x][c]) e caixa de recorte.
    # Retorna embedding normalizado.
    fn embed_caixa(mut self, var img_pixels: List[List[List[Float32]]],
                   var x0: Int, var y0: Int, var x1: Int, var y1: Int) raises -> List[Float32]:
        var ps = self.parametros.patch_size
        var patch = graficos_pkg.crop_and_resize_rgb(img_pixels, x0, y0, x1, y1, ps, ps)
        return self.embed(patch^)

    # Similaridade cosseno entre dois embeddings (já normalizados → produto escalar).
    fn similaridade(self, a: List[Float32], b: List[Float32]) -> Float32:
        var s: Float32 = 0.0
        var n = len(a)
        if len(b) < n:
            n = len(b)
        for i in range(n):
            s = s + a[i] * b[i]
        return s

    # ─── Persistência ──────────────────────────────────────────────────────────

    fn salvar(mut self) -> Bool:
        var dir = self.diretorio_modelo
        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
        except _:
            pass
        # Salva kernels via storage_sessao (mesmo mecanismo do Retina)
        try:
            var driver  = sessao_driver.driver_sessao_disco(dir)
            var storage = storage_sessao.criar_storage_sessao(driver)
            cnn_impl._salvar_bloco_em_storage(self.bloco_cnn, storage)
        except _:
            pass
        # Salva cabeças
        try:
            _ = self.proj_peso.salvar_em_arquivo(dir, "proj_peso.bin")
            _ = self.proj_bias.salvar_em_arquivo(dir, "proj_bias.bin")
        except _:
            pass
        if self.heads_inicializadas and len(self.cls_peso.dados) > 0:
            try:
                _ = self.cls_peso.salvar_em_arquivo(dir, "cls_peso.bin")
                _ = self.cls_bias.salvar_em_arquivo(dir, "cls_bias.bin")
            except _:
                pass
        # Metadados
        try:
            var lines = List[String]()
            lines.append("epoch:" + String(self.treinamento_epoca))
            lines.append("lr:" + String(self.treinamento_lr))
            lines.append("patch_size:" + String(self.parametros.patch_size))
            lines.append("embed_dim:" + String(self.parametros.embed_dim))
            lines.append("num_classes:" + String(self.parametros.num_classes))
            _ = uteis.gravar_texto_seguro(os.path.join(dir, "arcface_state.txt"),
                                          String("\n").join(lines^))
        except _:
            pass
        return True

    fn carregar(mut self) -> Bool:
        var dir = self.diretorio_modelo
        if not os.path.isdir(dir):
            return False
        # Carrega kernels
        try:
            var driver  = sessao_driver.driver_sessao_disco(dir)
            var storage = storage_sessao.criar_storage_sessao(driver)
            cnn_impl._carregar_bloco_de_storage(self.bloco_cnn, storage)
        except _:
            pass
        # Carrega projeção se existir
        try:
            var D = self.bloco_cnn.peso_saida.formato[0]
            var E = self.parametros.embed_dim
            if D > 0:
                var w_shape = List[Int](); w_shape.append(D); w_shape.append(E)
                self.proj_peso = tensor_defs.Tensor(w_shape^, self.bloco_cnn.tipo_computacao)
                _ = self.proj_peso.carregar_de_arquivo(dir, "proj_peso.bin")
                var b_shape = List[Int](); b_shape.append(1); b_shape.append(E)
                self.proj_bias = tensor_defs.Tensor(b_shape^, self.bloco_cnn.tipo_computacao)
                _ = self.proj_bias.carregar_de_arquivo(dir, "proj_bias.bin")
                self.heads_inicializadas = True
        except _:
            pass
        # Carrega estado
        try:
            var txt = arquivo_pkg.ler_arquivo_texto(os.path.join(dir, "arcface_state.txt"))
            for L in txt.split("\n"):
                if L.startswith("epoch:"):
                    self.treinamento_epoca = Int(L[6:])
                elif L.startswith("lr:"):
                    self.treinamento_lr = Float32(Float64(L[3:]))
        except _:
            pass
        return True

    # Treina o modelo. Chama arcface_trainer.treinar_arcface.
    fn treinar(mut self, var dataset_dir: String, var epocas: Int = 5,
               var lr: Float32 = 0.05, var batch_size: Int = 8,
               var batch_size_fim: Int = 128) -> String:
        try:
            import reconhecedor.arcface_trainer as trainer
            return trainer.treinar_arcface(self, dataset_dir, epocas, lr, batch_size, batch_size_fim)
        except e:
            return "Falha: " + String(e)
