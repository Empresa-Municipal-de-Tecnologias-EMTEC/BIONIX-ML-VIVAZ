import retina.model_detector as model_pkg
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import bionix_ml.nucleo.Tensor as tensor_defs
import bionix_ml.dados.arquivo as arquivo_pkg
import bionix_ml.dados as dados_pkg
import bionix_ml.uteis as uteis
import os
import bionix_ml.camadas.cnn as cnn_pkg

# Utility wrappers to create/load/save retina model components in a reusable way


 # BlocoRetinaFaceParametros:
 # - Parâmetros de configuração do bloco Retina (tamanhos, filtros, thresholds)
 # - Usado para construir/serializar o comportamento do `RetinaFace` wrapper
struct BlocoRetinaFaceParametros(Movable, Copyable):
    var input_size: Int
    var num_filtros: Int
    var kernel_h: Int
    var kernel_w: Int
    var tipo_ctx: String
    var patch_size: Int
    var max_per_image: Int
    var score_threshold: Float32
    var nms_iou: Float32

    fn __init__(out self,
                var input_size_in: Int = 640,
                var num_filtros_in: Int = 6,
                var kernel_h_in: Int = 3,
                var kernel_w_in: Int = 3,
                var tipo_ctx_in: String = "cpu",
                var patch_size_in: Int = 64,
                var max_per_image_in: Int = 16,
                var score_threshold_in: Float32 = 0.01,
                var nms_iou_in: Float32 = 0.5):
        self.input_size = input_size_in
        self.num_filtros = num_filtros_in
        self.kernel_h = kernel_h_in
        self.kernel_w = kernel_w_in
        self.tipo_ctx = tipo_ctx_in^
        self.patch_size = patch_size_in
        self.max_per_image = max_per_image_in
        self.score_threshold = score_threshold_in
        self.nms_iou = nms_iou_in


struct RetinaFace(Movable):
    # RetinaFace encapsula o detector leve usado neste repositório.
    # Componentes principais armazenados nesta struct:
    # - `bloco_cnn` (BlocoCNN): extrator de features (kernels + peso_saida/bias_saida)
    # - `cabeca_classificacao_peso` e `cabeca_classificacao_bias`: cabeça linear de classificação (D x 1, 1 x 1)
    # - `bloco_cnn.peso_saida` e `bloco_cnn.bias_saida`: cabeça de regressão (D x 4, 1 x 4)
    # - `parametros`: hiperparâmetros como input_size, patch_size, thresholds
    #
    # Diagrama (fluxo simplificado):
    #
    #   Input image (RGB)
    #            |
    #       gerar_anchors(input_size)
    #            |
    #   for each anchor -> crop & resize patch (patch_size)
    #            |
    #       BlocoCNN.extrair_features(patch)  -->  feature vector (D)
    #            |                                    |
    #    ------------------------------         ------------------
    #    |                            |         |                |
    # Regression head (D x 4)    Classifier head (D x 1)     (optional)
    #    |                            |                           
    #   deltas -> decode box         logit -> sigmoid -> score
    #    |                            |                           
    #    \------------------filter by score & apply NMS-----------/
    #                            |
    #                     final boxes (x0,y0,x1,y1)
    #
    # Persistência / arquivos:
    # - peso_reg.bin / bias_reg.bin : cabeça de regressão (saved from bloco_cnn)
    # - peso_cls.bin  / bias_cls.bin  : cabeça de classificação (cabeca_classificacao_peso / cabeca_classificacao_bias)
    # - metadata.txt : metadados de treino/época
    # - checkpoint do bloco: salvo via model_pkg.salvar_checkpoint
    
    # `bloco_cnn`: o extrator de features (BlocoCNN) que contém kernels e as
    #    tensões de saída usadas para regressão (`peso_saida`, `bias_saida`).
    var bloco_cnn: cnn_pkg.BlocoCNN
    # `cabeca_classificacao_peso`: tensor D x 1 usado para calcular o logit
    #    da cabeça de classificação (multiplicado pelo vetor de features).
    var cabeca_classificacao_peso: tensor_defs.Tensor
    # `cabeca_classificacao_bias`: bias escalar (1 x 1) para a cabeça de classificação.
    var cabeca_classificacao_bias: tensor_defs.Tensor
    # `parametros`: hiperparâmetros do wrapper (input_size, patch_size, thresholds)
    var parametros: BlocoRetinaFaceParametros
    # `diretorio_modelo`: caminho onde salvar/carregar pesos e metadata
    var diretorio_modelo: String
    # `ancoras`: cache local das anchors geradas para o `input_size`
    var ancoras: List[List[Float32]]
    # Estado de treinamento (opcional): preenchido pelo treinador ou manualmente
    var treinamento_epoca: Int
    var treinamento_lr: Float32
    var treinamento_meta: String

    fn __init__(out self, var parametros_in: BlocoRetinaFaceParametros = BlocoRetinaFaceParametros(), var diretorio_modelo_in: String = "MODELO/retina_modelo"):
        print("[DEBUG] RetinaFace.__init__: start; parametros.input_size=", parametros_in.input_size)
        self.parametros = parametros_in^
        # create underlying CNN block (may allocate memory)
        print("[DEBUG] RetinaFace.__init__: criando bloco_cnn via model_pkg.criar_bloco_detector()")
        self.bloco_cnn = model_pkg.criar_bloco_detector(self.parametros.input_size, self.parametros.input_size, self.parametros.num_filtros, self.parametros.kernel_h, self.parametros.kernel_w, contexto_defs.criar_contexto_padrao(self.parametros.tipo_ctx))
        print("[DEBUG] RetinaFace.__init__: bloco_cnn criado")
        # placeholder head tensors (initialized lazily)
        self.cabeca_classificacao_peso = tensor_defs.Tensor(List[Int](), self.bloco_cnn.tipo_computacao)
        self.cabeca_classificacao_bias = tensor_defs.Tensor(List[Int](), self.bloco_cnn.tipo_computacao)
        self.diretorio_modelo = diretorio_modelo_in^
        self.ancoras = List[List[Float32]]()
        self.treinamento_epoca = -1
        self.treinamento_lr = 0.0
        self.treinamento_meta = String("")

    fn carregar_heads(mut self, var model_dir: String) -> Bool:
        try:
            var D = 0
            try:
                D = self.bloco_cnn.peso_saida.formato[0]
            except _:
                D = 0
            if D <= 0:
                return False
            # Prepare tensors with correct shape then ask them to load from files
            var shape_w = List[Int]()
            shape_w.append(D); shape_w.append(1)
            self.cabeca_classificacao_peso = tensor_defs.Tensor(shape_w^, self.bloco_cnn.tipo_computacao)
            var shape_b = List[Int]()
            shape_b.append(1); shape_b.append(1)
            self.cabeca_classificacao_bias = tensor_defs.Tensor(shape_b^, self.bloco_cnn.tipo_computacao)
            var ok_w = False
            var ok_b = False
            try:
                ok_w = self.cabeca_classificacao_peso.carregar_de_arquivo(model_dir, "peso_cls.bin")
            except _:
                ok_w = False
            try:
                ok_b = self.cabeca_classificacao_bias.carregar_de_arquivo(model_dir, "bias_cls.bin")
            except _:
                ok_b = False
            return ok_w or ok_b
        except _:
            return False

    fn salvar(mut self, var model_dir: String, var meta_lines: List[String]) -> Bool:
        try:
            return self.salvar_workspace(model_dir)
        except _:
            return False

    # Salva o workspace completo do detector no diretório `model_dir`.
    # Isso inclui:
    # - BlocoCNN (kernels, peso_saida, bias_saida) via `bloco_cnn.salvar_estado`
    # - Cabeça de classificação (pesos/bias) via `salvar_estado_modelo`
    # - Metadados do detector e estado de treino em `retina_state.txt`
    fn salvar_workspace(mut self, var model_dir: String) -> Bool:
        try:
            # preparar metadados básicos
            var meta_lines = List[String]()
            meta_lines.append("input_size:" + String(self.parametros.input_size))
            meta_lines.append("patch_size:" + String(self.parametros.patch_size))
            meta_lines.append("num_filtros:" + String(self.parametros.num_filtros))
            meta_lines.append("score_threshold:" + String(self.parametros.score_threshold))
            meta_lines.append("nms_iou:" + String(self.parametros.nms_iou))
            # include training state
            meta_lines.append("train_epoch:" + String(self.treinamento_epoca))
            meta_lines.append("train_lr:" + String(self.treinamento_lr))
            if len(self.treinamento_meta) > 0:
                meta_lines.append("train_meta:" + self.treinamento_meta)

            # ensure dir exists
            try:
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
            except _:
                pass

            # Save block via BlocoCNN and export heads/metadata
            try:
                var ok_block = model_pkg.salvar_checkpoint(self.bloco_cnn, model_dir)
            except _:
                ok_block = False
            try:
                if len(self.bloco_cnn.peso_saida.formato) >= 1:
                    _ = self.bloco_cnn.peso_saida.salvar_em_arquivo(model_dir, "peso_reg.bin")
            except _:
                pass
            try:
                if len(self.bloco_cnn.bias_saida.formato) >= 1:
                    _ = self.bloco_cnn.bias_saida.salvar_em_arquivo(model_dir, "bias_reg.bin")
            except _:
                pass
            try:
                if len(self.cabeca_classificacao_peso.formato) >= 1:
                    _ = self.cabeca_classificacao_peso.salvar_em_arquivo(model_dir, "peso_cls.bin")
            except _:
                pass
            try:
                if len(self.cabeca_classificacao_bias.formato) >= 1:
                    _ = self.cabeca_classificacao_bias.salvar_em_arquivo(model_dir, "bias_cls.bin")
            except _:
                pass
            var ok_block: Bool = False
            var okc: Bool = False
            try:
                okc = model_pkg.salvar_checkpoint(self.bloco_cnn, model_dir)
            except _:
                okc = False
            # consider save successful if either block checkpoint or exported files existed
            if not ok_block and not okc:
                return False

            # also write a simple state file with one-per-line key:val for quick parsing
            try:
                var state_lines = List[String]()
                state_lines.append("epoch:" + String(self.treinamento_epoca))
                state_lines.append("lr:" + String(self.treinamento_lr))
                state_lines.append("meta:" + self.treinamento_meta)
                uteis.gravar_texto_seguro(os.path.join(model_dir, "retina_state.txt"), String("\n").join(state_lines.copy()))
            except _:
                pass

            return True
        except _:
            return False

    # Carrega todo o workspace do detector a partir do diretório `model_dir`.
    # Retorna True se conseguiu carregar pelo menos o bloco ou as cabeças.
    fn carregar_workspace(mut self, var model_dir: String) -> Bool:
        var any_ok: Bool = False
        try:
            # carregar bloco (checkpoint)
            try:
                var okb = model_pkg.carregar_checkpoint(self.bloco_cnn, model_dir)
                if okb:
                    any_ok = True
            except _:
                pass

            # carregar heads de classificação (peso/bias)
            try:
                var okh = self.carregar_heads(model_dir)
                if okh:
                    any_ok = True
            except _:
                pass

            # carregar treino/state file
            try:
                var txt = arquivo_pkg.ler_arquivo_texto(os.path.join(model_dir, "retina_state.txt"))
                if len(txt) > 0:
                    var lines = txt.split("\n")
                    for L in lines:
                        try:
                            if L.startswith("epoch:"):
                                self.treinamento_epoca = Int(L.split(":")[1])
                            elif L.startswith("lr:"):
                                self.treinamento_lr = Float32(Float64(L.split(":")[1]))
                            elif L.startswith("meta:"):
                                self.treinamento_meta = String(L.split(":", 1)[1])
                        except _:
                            continue
            except _:
                pass

            return any_ok
        except _:
            return False

    fn inferir(mut self, img_pixels: List[List[List[Float32]]], var input_size: Int = -1, var max_per_image: Int = -1) -> List[List[Int]]:
        import retina.retina_anchor_generator as anchor_gen
        import retina.retina_nms as nms_pkg
        import bionix_ml.graficos as graficos_pkg
        import bionix_ml.nucleo.Tensor as tensor_defs_local
        import math

        var in_size = input_size if input_size > 0 else self.parametros.input_size
        var maxp = max_per_image if max_per_image > 0 else self.parametros.max_per_image

        var anchors = anchor_gen.gerar_anchors(in_size)
        var cls_scores: List[Float32] = List[Float32]()
        var reg_deltas: List[List[Float32]] = List[List[Float32]]()
        var patch_size = self.parametros.patch_size

        # Prepare classification tensors: prefer in-memory heads, otherwise try to load from files
        var peso_cls_tensor = tensor_defs_local.Tensor(List[Int](), self.bloco_cnn.tipo_computacao)
        var bias_cls_tensor = tensor_defs_local.Tensor(List[Int](), self.bloco_cnn.tipo_computacao)
        var cls_tensors_inited = False
        try:
            if len(self.cabeca_classificacao_peso.formato) >= 1:
                peso_cls_tensor = self.cabeca_classificacao_peso.copy()
                bias_cls_tensor = self.cabeca_classificacao_bias.copy()
                cls_tensors_inited = True
            else:
                # try load from canonical files
                var Dguess = 0
                try:
                    Dguess = self.bloco_cnn.peso_saida.formato[0]
                except _:
                    Dguess = 0
                if Dguess > 0:
                    var shape_w = List[Int](); shape_w.append(Dguess); shape_w.append(1)
                    peso_cls_tensor = tensor_defs_local.Tensor(shape_w^, self.bloco_cnn.tipo_computacao)
                    try:
                        var raw_w = arquivo_pkg.ler_arquivo_binario(os.path.join(self.diretorio_modelo, "peso_cls.bin"))
                        if len(raw_w) > 0:
                            _ = peso_cls_tensor.carregar_dados_bytes_bin(raw_w.copy())
                            cls_tensors_inited = True
                    except _:
                        pass
                    try:
                        var raw_b = arquivo_pkg.ler_arquivo_binario(os.path.join(self.diretorio_modelo, "bias_cls.bin"))
                        if len(raw_b) > 0:
                            var shape_b = List[Int](); shape_b.append(1); shape_b.append(1)
                            bias_cls_tensor = tensor_defs_local.Tensor(shape_b^, self.bloco_cnn.tipo_computacao)
                            _ = bias_cls_tensor.carregar_dados_bytes_bin(raw_b.copy())
                    except _:
                        pass
        except _:
            cls_tensors_inited = False

        for i in range(len(anchors)):
            var a = anchors[i].copy()
            var ax = Int(a[0] - a[2] / 2.0); var ay = Int(a[1] - a[3] / 2.0)
            var aw = Int(a[2]); var ah = Int(a[3])
            if ax < 0: ax = 0
            if ay < 0: ay = 0
            if ax + aw > in_size: aw = max(1, in_size - ax)
            if ay + ah > in_size: ah = max(1, in_size - ay)

            var patch_rgb = graficos_pkg.crop_and_resize_rgb(img_pixels, ax, ay, ax + aw - 1, ay + ah - 1, patch_size, patch_size)
            var in_shape = List[Int]()
            in_shape.append(1); in_shape.append(patch_size * patch_size * 3)
            var tensor_in = tensor_defs_local.Tensor(in_shape^, self.bloco_cnn.tipo_computacao)
            for yy in range(patch_size):
                for xx in range(patch_size):
                    var pix = patch_rgb[yy][xx].copy()
                    var base = (yy * patch_size + xx) * 3
                    tensor_in.dados[base + 0] = pix[0]
                    tensor_in.dados[base + 1] = pix[1]
                    tensor_in.dados[base + 2] = pix[2]

            var feats_t = tensor_defs_local.Tensor(List[Int](), self.bloco_cnn.tipo_computacao)
            try:
                feats_t = cnn_pkg.extrair_features(self.bloco_cnn, tensor_in)
            except _:
                feats_t = tensor_defs_local.Tensor(List[Int](), self.bloco_cnn.tipo_computacao)
            var D = 0
            try:
                D = feats_t.formato[1]
            except _:
                D = 0
            var feats = List[Float32]()
            for d in range(D):
                feats.append(feats_t.dados[d])

            var pred = List[Float32]()
            for j in range(4):
                var s: Float32 = 0.0
                for d in range(D):
                    s = s + feats[d] * self.bloco_cnn.peso_saida.dados[d * 4 + j]
                s = s + self.bloco_cnn.bias_saida.dados[j]
                pred.append(s)

            var score: Float32 = 0.01
            if cls_tensors_inited and D > 0:
                var logit: Float32 = 0.0
                for d in range(min(D, peso_cls_tensor.formato[0])):
                    logit = logit + feats[d] * peso_cls_tensor.dados[d]
                if len(bias_cls_tensor.dados) > 0:
                    logit = logit + bias_cls_tensor.dados[0]
                if logit > 50.0:
                    score = 1.0
                elif logit < -50.0:
                    score = 0.0
                else:
                    score = 1.0 / (1.0 + Float32(math.exp(-Float64(logit))))

            cls_scores.append(score)
            var drow = List[Float32]()
            for v in pred:
                drow.append(v)
            # transfer newly created row into reg_deltas to avoid implicit copy
            reg_deltas.append(drow^)

        var boxes: List[List[Float32]] = List[List[Float32]]()
        for i in range(len(anchors)):
            var a = anchors[i].copy()
            var dx = reg_deltas[i][0]; var dy = reg_deltas[i][1]; var dw = reg_deltas[i][2]; var dh = reg_deltas[i][3]
            var cx = a[0] + dx * a[2]
            var cy = a[1] + dy * a[3]
            var w = a[2] * Float32(math.exp(Float64(dw)))
            var h = a[3] * Float32(math.exp(Float64(dh)))
            var x0 = cx - w/2.0
            var y0 = cy - h/2.0
            var x1 = cx + w/2.0
            var y1 = cy + h/2.0
            var outb = List[Float32]()
            outb.append(x0); outb.append(y0); outb.append(x1); outb.append(y1)
            # transfer freshly-built box to avoid implicit copy
            boxes.append(outb^)

        var keep = nms_pkg.non_max_suppression(boxes, cls_scores, self.parametros.nms_iou)
        var kept_boxes = List[List[Int]]()
        for k in keep:
            if len(kept_boxes) >= maxp:
                break
            var b = boxes[k].copy()
            var ib = List[Int]()
            ib.append(Int(b[0])); ib.append(Int(b[1])); ib.append(Int(b[2])); ib.append(Int(b[3]))
            # transfer the small int-list into kept_boxes
            kept_boxes.append(ib^)
        # transfer ownership of kept_boxes to caller to avoid implicit copy
        return kept_boxes^

    fn treinar(mut self, var dataset_dir: String, var altura: Int = 640, var largura: Int = 640,
               var patch_size: Int = 64, var epocas: Int = 5, var taxa_aprendizado: Float32 = 0.0001,
               var batch_size: Int = 4) -> String:
        # Import trainer lazily to avoid circular import at module load time
        try:
            import retina.retina_trainer as trainer
            return trainer.treinar_retina_minimal(self, dataset_dir, altura, largura, patch_size, epocas, taxa_aprendizado, batch_size)
        except _:
            # Fallback: if import fails, return error string
            return "Falha: não foi possível iniciar treinador"

    # A cabeça de regressão (pesos e bias) é armazenada dentro do `bloco_cnn` como
    # `bloco_cnn.peso_saida` e `bloco_cnn.bias_saida`. Esta função retorna uma
    # tupla (peso, bias) pronta para uso e evita duplicar estado dentro do wrapper.
    fn obter_cabeca_regressao(self) -> (tensor_defs.Tensor, tensor_defs.Tensor):
        try:
            return (self.bloco_cnn.peso_saida, self.bloco_cnn.bias_saida)
        except _:
            var empty_w = tensor_defs.Tensor(List[Int](), self.bloco_cnn.tipo_computacao)
            var empty_b = tensor_defs.Tensor(List[Int](), self.bloco_cnn.tipo_computacao)
            return (empty_w, empty_b)
