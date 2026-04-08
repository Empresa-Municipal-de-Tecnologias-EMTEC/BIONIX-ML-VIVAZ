import retina.model_detector as model_pkg
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import bionix_ml.nucleo.Tensor as tensor_defs
import bionix_ml.dados.arquivo as arquivo_pkg
import bionix_ml.dados as dados_pkg
import bionix_ml.uteis as uteis
import os
import bionix_ml.camadas.cnn as cnn_pkg
import bionix_ml.camadas.cnn.cnn as cnn_impl
import bionix_ml.computacao.dispatcher_tensor as dispatcher_tensor
import retina.retina_gerador_ancoras as gerador_ancoras_pkg
import retina.retina_supressao_por_pontuacao as supressao_por_pontuacao_pkg
import bionix_ml.graficos as graficos_pkg
import bionix_ml.nucleo.Tensor as tensor_defs_local
import math
import retina.retina_trainer as trainer
import bionix_ml.camadas as camadas_pkg

# Utilitários para configuração, construção, inferência e persistência do modelo RetinaFace.

 # BlocoRetinaFaceParametros:
 # - Parâmetros de configuração do bloco Retina (tamanhos, filtros, thresholds)
 # - Usado para construir/serializar o comportamento do `RetinaFace`
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
    # Diagrama (fluxo atual — conv→FPN→cabeças conv):
    #
    #   Imagem de entrada (RGB, ex.: 640x640)
    #                |
    #             Backbone
    #    (ResNet/MobileNet — extrai mapas P3,P4,P5)
    #                |
    #               FPN
    #    (combina mapas multi‑escala, produz P3',P4',P5')
    #          /      |        \
    #         v       v         v
    #     Cabeça 1  Cabeça 2  Cabeça 3
    #    (conv cls) (conv cls) (conv cls)  -> logits por local
    #    (conv reg) (conv reg) (conv reg)  -> deltas tx,ty,tw,th
    #                |
    #          Decodificação + Pós‑processamento
    #    (deltas+anchors -> boxes) + Filtragem + NMS
    #                |
    #            Saídas multitarefa:
    #    - Face score (confiança)
    #    - Face box (x,y,w,h)
    #    - Landmarks (5 pts)
    #    - (Opcional) reconstrução 3D / mesh
    #
    # Persistência / arquivos:
    # - peso_reg.bin / bias_reg.bin : cabeça de regressão (saved from bloco_cnn)
    # - peso_cls.bin  / bias_cls.bin  : cabeça de classificação (cabeca_classificacao_peso / cabeca_classificacao_bias)
    # - metadata.txt : metadados de treino/época
    # - checkpoint do bloco: salvo via model_pkg.salvar_checkpoint
    
    # Removido `BlocoCNN` patch-based. Substituído por tensores explícitos
    # que representam kernels e cabeças para compatibilidade e persistência.
    var tipo_computacao: String
    var bloco_kernels: List[tensor_defs.Tensor]
    var bloco_peso_saida: tensor_defs.Tensor
    var bloco_bias_saida: tensor_defs.Tensor
    # Novo: flags e tensores para topologia conv-FPN (opcional)
    # Quando `usar_conv_fpn` for True, espera-se que o detector execute
    # o pipeline: Backbone -> FPN -> cabeças conv por nível (cls/reg).
    # Os tensores abaixo são placeholders para pesos/viéses que serão
    # preenchidos por `configurar_conv_fpn` ou por carregamento de checkpoint.
    var usar_conv_fpn: Bool
    var backbone_tipo: String
    var backbone_pesos: tensor_defs.Tensor
    var fpn_pesos: tensor_defs.Tensor
    var head_cls_pesos_conv: tensor_defs.Tensor
    # Bias/offsets para cabeças conv (separados dos pesos)
    var head_cls_bias_conv: tensor_defs.Tensor
    var head_reg_bias_conv: tensor_defs.Tensor
    var head_reg_pesos_conv: tensor_defs.Tensor
    # Parâmetros do gerador de âncoras (para garantir determinismo ao salvar/carregar)
    var anchor_passos: List[Int]
    var anchor_escalas: List[List[Float32]]
    var anchor_multiplicadores: List[Float32]
    var anchor_proporcoes: List[Float32]
    # Regression target weights (wx, wy, ww, wh). Targets generated by the assigner
    # will be multiplied by these weights: tgt = raw_delta * weight.
    # During decode we must divide network outputs by these weights.
    var bbox_reg_weights: List[Float32]
    # Multi-task loss weights
    var cls_loss_weight: Float32
    var reg_loss_weight: Float32
    var lmk_loss_weight: Float32
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
        # Inicializar tensores do bloco como formato vazio (compatibilidade)
        self.tipo_computacao = String(self.parametros.tipo_ctx)
        self.bloco_kernels = List[tensor_defs.Tensor]()
        self.bloco_peso_saida = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
        self.bloco_bias_saida = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
        # placeholder head tensors (initialized lazily)
        self.cabeca_classificacao_peso = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
        self.cabeca_classificacao_bias = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
        self.diretorio_modelo = diretorio_modelo_in^
        self.ancoras = List[List[Float32]]()
        self.treinamento_epoca = -1
        self.treinamento_lr = 0.0
        self.treinamento_meta = String("")
        # defaults for conv-FPN scaffolding
        self.usar_conv_fpn = False
        self.backbone_tipo = String("")
        self.backbone_pesos = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
        self.fpn_pesos = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
        self.head_cls_pesos_conv = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
        self.head_cls_bias_conv = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
        self.head_reg_bias_conv = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
        self.head_reg_pesos_conv = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
        # default anchor generation params (kept to reproduce anchors deterministically)
        self.anchor_passos = List[Int]()
        self.anchor_passos.append(8); self.anchor_passos.append(16); self.anchor_passos.append(32)
        self.anchor_escalas = List[List[Float32]]()
        self.anchor_multiplicadores = List[Float32]()
        self.anchor_multiplicadores.append(4.0); self.anchor_multiplicadores.append(8.0); self.anchor_multiplicadores.append(12.0)
        self.anchor_proporcoes = List[Float32]()
        self.anchor_proporcoes.append(0.5); self.anchor_proporcoes.append(1.0); self.anchor_proporcoes.append(1.5)
        # default bbox regression weights and loss weights
        self.bbox_reg_weights = List[Float32]()
        self.bbox_reg_weights.append(Float32(0.1)); self.bbox_reg_weights.append(Float32(0.1)); self.bbox_reg_weights.append(Float32(1.0)); self.bbox_reg_weights.append(Float32(1.0))
        self.cls_loss_weight = Float32(1.0)
        self.reg_loss_weight = Float32(1.0)
        self.lmk_loss_weight = Float32(0.5)

    fn carregar_heads(mut self, var model_dir: String) -> Bool:
        try:
            var D = 0
            try:
                if len(self.bloco_peso_saida.formato) >= 1:
                    D = self.bloco_peso_saida.formato[0]
                else:
                    D = 0
            except _:
                D = 0
            var shape_w = List[Int]()
            shape_w.append(D); shape_w.append(1)
            self.cabeca_classificacao_peso = tensor_defs.Tensor(shape_w^, self.tipo_computacao)
            var shape_b = List[Int]()
            shape_b.append(1); shape_b.append(1)
            self.cabeca_classificacao_bias = tensor_defs.Tensor(shape_b^, self.tipo_computacao)
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

            # Persist conv-FPN and head tensors as the canonical model artifacts.
            try:
                if len(self.head_cls_pesos_conv.formato) >= 1:
                    _ = self.head_cls_pesos_conv.salvar_em_arquivo(model_dir, "peso_head_cls_conv.bin")
            except _:
                pass
            try:
                if len(self.head_reg_pesos_conv.formato) >= 1:
                    _ = self.head_reg_pesos_conv.salvar_em_arquivo(model_dir, "peso_head_reg_conv.bin")
            except _:
                pass
            try:
                if len(self.backbone_pesos.formato) >= 1:
                    _ = self.backbone_pesos.salvar_em_arquivo(model_dir, "peso_backbone.bin")
            except _:
                pass
            try:
                if len(self.fpn_pesos.formato) >= 1:
                    _ = self.fpn_pesos.salvar_em_arquivo(model_dir, "peso_fpn.bin")
            except _:
                pass
            try:
                # salvar biases das cabeças conv (se existirem)
                if len(self.head_cls_bias_conv.formato) >= 1:
                    _ = self.head_cls_bias_conv.salvar_em_arquivo(model_dir, "bias_head_cls_conv.bin")
            except _:
                pass
            try:
                if len(self.head_reg_bias_conv.formato) >= 1:
                    _ = self.head_reg_bias_conv.salvar_em_arquivo(model_dir, "bias_head_reg_conv.bin")
            except _:
                pass
            try:
                # salvar tensores do bloco (kernels + peso_saida/bias_saida) para compatibilidade
                for idx in range(len(self.bloco_kernels)):
                    try:
                        _ = self.bloco_kernels[idx].salvar_em_arquivo(model_dir, "bloco_kernel_" + String(idx) + ".bin")
                    except _:
                        pass
                try:
                    if len(self.bloco_peso_saida.formato) >= 1:
                        _ = self.bloco_peso_saida.salvar_em_arquivo(model_dir, "bloco_peso_saida.bin")
                except _:
                    pass
                try:
                    if len(self.bloco_bias_saida.formato) >= 1:
                        _ = self.bloco_bias_saida.salvar_em_arquivo(model_dir, "bloco_bias_saida.bin")
                except _:
                    pass
            except _:
                pass
            # Also persist legacy classification heads if present for compatibility
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

            # also write a simple state file with one-per-line key:val for quick parsing
            try:
                var state_lines = List[String]()
                state_lines.append("epoch:" + String(self.treinamento_epoca))
                state_lines.append("lr:" + String(self.treinamento_lr))
                state_lines.append("meta:" + self.treinamento_meta)
                state_lines.append("input_size:" + String(self.parametros.input_size))
                state_lines.append("patch_size:" + String(self.parametros.patch_size))
                try:
                    state_lines.append("tipo_computacao:" + self.tipo_computacao)
                except _:
                    state_lines.append("tipo_computacao:")
                # pipeline flags and types
                state_lines.append("usar_conv_fpn:" + ("1" if self.usar_conv_fpn else "0"))
                try:
                    state_lines.append("backbone_tipo:" + self.backbone_tipo)
                except _:
                    state_lines.append("backbone_tipo:")
                # bbox_reg_weights
                try:
                    var brw_s = List[String]()
                    for v in self.bbox_reg_weights:
                        brw_s.append(String(v))
                    state_lines.append("bbox_reg_weights:" + String(",").join(brw_s.copy()))
                except _:
                    pass
                # anchor params
                try:
                    state_lines.append("anchor_passos:" + String(",").join([String(x) for x in self.anchor_passos]))
                except _:
                    pass
                try:
                    var mults = List[String]()
                    for v in self.anchor_multiplicadores: mults.append(String(v))
                    state_lines.append("anchor_multiplicadores:" + String(",").join(mults.copy()))
                except _:
                    pass
                try:
                    var props = List[String]()
                    for v in self.anchor_proporcoes: props.append(String(v))
                    state_lines.append("anchor_proporcoes:" + String(",").join(props.copy()))
                except _:
                    pass
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
            # Load conv-FPN artifacts and heads (conv tensors are canonical now).
            try:
                any_ok = False
            except _:
                any_ok = False

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
                            elif L.startswith("usar_conv_fpn:"):
                                try:
                                    self.usar_conv_fpn = True if L.split(":")[1] == "1" else False
                                except _:
                                    pass
                            elif L.startswith("backbone_tipo:"):
                                try:
                                    self.backbone_tipo = String(L.split(":", 1)[1])
                                except _:
                                    pass
                            elif L.startswith("tipo_computacao:"):
                                try:
                                    self.tipo_computacao = String(L.split(":", 1)[1])
                                except _:
                                    pass
                            elif L.startswith("bbox_reg_weights:"):
                                try:
                                    var parts = L.split(":", 1)[1].split(",")
                                    self.bbox_reg_weights = List[Float32]()
                                    for p in parts:
                                        try: self.bbox_reg_weights.append(Float32(Float64(p)))
                                        except _: continue
                                except _:
                                    pass
                            elif L.startswith("anchor_passos:"):
                                try:
                                    var parts = L.split(":", 1)[1].split(",")
                                    self.anchor_passos = List[Int]()
                                    for p in parts:
                                        try: self.anchor_passos.append(Int(p))
                                        except _: continue
                                except _:
                                    pass
                            elif L.startswith("anchor_multiplicadores:"):
                                try:
                                    var parts = L.split(":", 1)[1].split(",")
                                    self.anchor_multiplicadores = List[Float32]()
                                    for p in parts:
                                        try: self.anchor_multiplicadores.append(Float32(Float64(p)))
                                        except _: continue
                                except _:
                                    pass
                            elif L.startswith("anchor_proporcoes:"):
                                try:
                                    var parts = L.split(":", 1)[1].split(",")
                                    self.anchor_proporcoes = List[Float32]()
                                    for p in parts:
                                        try: self.anchor_proporcoes.append(Float32(Float64(p)))
                                        except _: continue
                                except _:
                                    pass
                        except _:
                            continue
            except _:
                pass

            # Try to load conv-FPN weights and head tensors
            try:
                try:
                    var raw_bb = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "peso_backbone.bin"))
                    if len(raw_bb) > 0:
                        try: _ = self.backbone_pesos.carregar_dados_bytes_bin(raw_bb.copy())
                        except _: pass
                except _:
                    pass
            except _:
                pass
            try:
                try:
                    var raw_fpn = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "peso_fpn.bin"))
                    if len(raw_fpn) > 0:
                        try: _ = self.fpn_pesos.carregar_dados_bytes_bin(raw_fpn.copy())
                        except _: pass
                except _:
                    pass
            except _:
                pass
            try:
                try:
                    var raw_hc = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "peso_head_cls_conv.bin"))
                    if len(raw_hc) > 0:
                        try: _ = self.head_cls_pesos_conv.carregar_dados_bytes_bin(raw_hc.copy()); any_ok = True
                        except _: pass
                except _:
                    pass
            except _:
                pass
            try:
                try:
                    var raw_hr = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "peso_head_reg_conv.bin"))
                    if len(raw_hr) > 0:
                        try: _ = self.head_reg_pesos_conv.carregar_dados_bytes_bin(raw_hr.copy()); any_ok = True
                        except _: pass
                except _:
                    pass
            except _:
                pass
            try:
                try:
                    var raw_bhc = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "bias_head_cls_conv.bin"))
                    if len(raw_bhc) > 0:
                        try: _ = self.head_cls_bias_conv.carregar_dados_bytes_bin(raw_bhc.copy()); any_ok = True
                        except _: pass
                except _:
                    pass
            except _:
                pass
            try:
                try:
                    var raw_bhr = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "bias_head_reg_conv.bin"))
                    if len(raw_bhr) > 0:
                        try: _ = self.head_reg_bias_conv.carregar_dados_bytes_bin(raw_bhr.copy()); any_ok = True
                        except _: pass
                except _:
                    pass
            except _:
                pass
            try:
                # tentar carregar tensores do bloco para compatibilidade
                try:
                    var raw_pw = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "bloco_peso_saida.bin"))
                    if len(raw_pw) > 0:
                        try: _ = self.bloco_peso_saida.carregar_dados_bytes_bin(raw_pw.copy()); any_ok = True
                        except _: pass
                except _:
                    pass
                try:
                    var raw_b = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "bloco_bias_saida.bin"))
                    if len(raw_b) > 0:
                        try: _ = self.bloco_bias_saida.carregar_dados_bytes_bin(raw_b.copy()); any_ok = True
                        except _: pass
                except _:
                    pass
                # kernels opcionalmente salvos como bloco_kernel_i.bin
                try:
                    var kidx = 0
                    while True:
                        try:
                            var raw_k = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "bloco_kernel_" + String(kidx) + ".bin"))
                            if len(raw_k) == 0:
                                break
                            # infer kernel shape from raw byte length when possible
                            var nvals = len(raw_k) // 4
                            var formato_k = List[Int]()
                            try:
                                var s = Int(math.floor(math.sqrt(Float64(nvals))))
                                if s * s == nvals and s > 0:
                                    formato_k.append(s); formato_k.append(s)
                                else:
                                    formato_k.append(nvals); formato_k.append(1)
                            except _:
                                formato_k.append(3); formato_k.append(3)
                            var kt = tensor_defs.Tensor(formato_k^, self.tipo_computacao)
                            try:
                                _ = kt.carregar_dados_bytes_bin(raw_k.copy())
                                self.bloco_kernels.append(kt^)
                                any_ok = True
                            except _:
                                pass
                            kidx = kidx + 1
                        except _:
                            break
                except _:
                    pass
            except _:
                pass

            # Also try to load legacy classification heads for compatibility
            try:
                try:
                    var raw_w = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "peso_cls.bin"))
                    if len(raw_w) > 0 and len(self.cabeca_classificacao_peso.formato) >= 1:
                        try: _ = self.cabeca_classificacao_peso.carregar_dados_bytes_bin(raw_w.copy()); any_ok = True
                        except _: pass
                except _:
                    pass
            except _:
                pass
            try:
                try:
                    var raw_b = arquivo_pkg.ler_arquivo_binario(os.path.join(model_dir, "bias_cls.bin"))
                    if len(raw_b) > 0 and len(self.cabeca_classificacao_bias.formato) >= 1:
                        try: _ = self.cabeca_classificacao_bias.carregar_dados_bytes_bin(raw_b.copy()); any_ok = True
                        except _: pass
                except _:
                    pass
            except _:
                pass

            return any_ok
        except _:
            return False

    # Redimensiona `img_pixels` para `target_size x target_size` preservando as caixas
    # fornecidas em `caixas` (lista de [x0,y0,x1,y1]) garantindo que a região
    # das anotações não seja perdida: faz um crop quadrado que contenha as caixas
    # (ou center-crop se não há caixas) e então redimensiona para `target_size`.
    # Retorna (img_redimensionada, caixas_ajustadas).
    fn redimensionar_para_tamanho_entrada(self, img_pixels: List[List[List[Float32]]], var caixas: List[List[Int]], var target_size: Int = 320) -> (List[List[List[Float32]]], List[List[Int]]):
        var h: Int = 0
        try:
            h = len(img_pixels)
        except _:
            h = 0
        var w: Int = 0
        try:
            if h > 0:
                w = len(img_pixels[0])
        except _:
            w = 0
        if w <= 0 or h <= 0:
            return (img_pixels.copy(), caixas.copy())

        var caixas_out: List[List[Int]] = List[List[Int]]()

        # determinar região alvo (crop) que preserve todas as caixas, se existirem
        var crop_x: Int = 0
        var crop_y: Int = 0
        var side: Int = 0
        if len(caixas) == 0:
            # center-crop quadrado
            side = min(w, h)
            crop_x = (w - side) // 2
            crop_y = (h - side) // 2
        else:
            var minx: Int = 1_000_000
            var miny: Int = 1_000_000
            var maxx: Int = -1_000_000
            var maxy: Int = -1_000_000
            for cb in caixas:
                try:
                    if cb[0] < minx: minx = cb[0]
                    if cb[1] < miny: miny = cb[1]
                    if cb[2] > maxx: maxx = cb[2]
                    if cb[3] > maxy: maxy = cb[3]
                except _:
                    continue
            var bbox_w: Int = maxx - minx
            var bbox_h: Int = maxy - miny
            if bbox_w < 0: bbox_w = 0
            if bbox_h < 0: bbox_h = 0
            side = bbox_w if bbox_w > bbox_h else bbox_h
            if side <= 0:
                side = min(w, h)
            # ensure side does not exceed image smallest dim
            var max_side = min(w, h)
            if side > max_side:
                side = max_side
            var center_x = Float32(minx + maxx) / 2.0
            var center_y = Float32(miny + maxy) / 2.0
            crop_x = Int(center_x - Float32(side) / 2.0)
            crop_y = Int(center_y - Float32(side) / 2.0)
            if crop_x < 0: crop_x = 0
            if crop_y < 0: crop_y = 0
            if crop_x + side > w: crop_x = w - side
            if crop_y + side > h: crop_y = h - side

        if side <= 0:
            side = min(w, h)
            crop_x = (w - side) // 2
            crop_y = (h - side) // 2

        var x1 = crop_x + side - 1
        var y1 = crop_y + side - 1
        # crop_and_resize_rgb usa x1,y1 inclusivos (consistente com uso no código)
        var resized = graficos_pkg.crop_and_resize_rgb(img_pixels, crop_x, crop_y, x1, y1, target_size, target_size)

        var scale: Float32 = 1.0
        try:
            scale = Float32(target_size) / Float32(side)
        except _:
            scale = 1.0

        for cb in caixas:
            try:
                var nx0 = Int(Float32(cb[0] - crop_x) * scale)
                var ny0 = Int(Float32(cb[1] - crop_y) * scale)
                var nx1 = Int(Float32(cb[2] - crop_x) * scale)
                var ny1 = Int(Float32(cb[3] - crop_y) * scale)
                if nx0 < 0: nx0 = 0
                if ny0 < 0: ny0 = 0
                if nx1 >= target_size: nx1 = target_size - 1
                if ny1 >= target_size: ny1 = target_size - 1
                var nb: List[Int] = List[Int]()
                nb.append(nx0); nb.append(ny0); nb.append(nx1); nb.append(ny1)
                caixas_out.append(nb^)
            except _:
                # em caso de erro, pular caixa
                continue

        return (resized.copy(), caixas_out^)

    fn inferir(mut self, img_pixels: List[List[List[Float32]]], var input_size: Int = -1, var max_per_image: Int = -1) -> List[List[Int]]:
        # Conv-FPN is now the canonical inference path; no fallback to patch-based flow.
        return self.inferir_convfpn(img_pixels, input_size, max_per_image)

    # Generate per-anchor predictions using the lightweight conv-FPN skeleton.
    # Returns: (cls_logits, reg_deltas, mean_vals, base_reg_list)
    fn gerar_predicoes_por_ancora_convfpn(mut self, img_pixels: List[List[List[Float32]]], anchors: List[List[Float32]], patch_size: Int) -> (List[Float32], List[List[Float32]], List[Float32], List[List[Float32]]):
        var in_size = self.parametros.input_size
        var img_matrix = img_pixels.copy()
        try:
            var resized_tuple = self.redimensionar_para_tamanho_entrada(img_pixels.copy(), List[List[Int]](), in_size)
            img_matrix = resized_tuple[0].copy()
        except _:
            img_matrix = img_pixels.copy()

        print("na geração de predições chegou aqui 00")

        # Build backbone feature maps (P3,P4,P5)
        var fmaps = self._backbone_forward(img_matrix, in_size)

        # Extra diagnostics: report structure of feature maps to isolate crashes
        try:
            try: print("[DBG] gerar_predicoes: fmaps_levels=" + String(len(fmaps)))
            except _: pass
            for lvl_idx in range(len(fmaps)):
                try:
                    var lvl = fmaps[lvl_idx]
                    var Hf = 0; var Wf = 0; var Cc = 0
                    try:
                        Hf = len(lvl)
                        if Hf > 0: Wf = len(lvl[0])
                        if Hf > 0 and Wf > 0 and len(lvl[0][0]) > 0:
                            Cc = len(lvl[0][0])
                    except _:
                        Hf = 0; Wf = 0; Cc = 0
                    try: print("[DBG] gerar_predicoes: fmap[level]=" + String(lvl_idx) + " Hf=" + String(Hf) + " Wf=" + String(Wf) + " C=" + String(Cc))
                    except _: pass
                except _:
                    try: print("[DBG] gerar_predicoes: fmap[level]=" + String(lvl_idx) + " UNAVAILABLE")
                    except _: pass
        except _:
            pass

        print("na geração de predições chegou aqui 01")

        # Generate anchors per-level (keeps backward compatibility with flat anchors elsewhere)
        var anchors_by_level = gerador_ancoras_pkg.gerar_ancoras_por_nivel(in_size, self.anchor_passos.copy(), self.anchor_escalas.copy(), self.anchor_multiplicadores.copy(), self.anchor_proporcoes.copy())

        # Diagnostic guards: detect obviously-broken shapes before heavy processing
        try:
            # anchors_by_level is List[level] of anchors lists
            var total_anchors = 0
            for lvl in anchors_by_level:
                total_anchors = total_anchors + len(lvl)
            try: print("[DEBUG] gerar_predicoes: total_anchors=", total_anchors)
            except _: pass
            # inspect head placeholder formatos
            var hcf = List[Int]()
            var hrf = List[Int]()
            try:
                try: hcf = self.head_cls_pesos_conv.formato.copy()
                except _: hcf = List[Int]()
                try: hrf = self.head_reg_pesos_conv.formato.copy()
                except _: hrf = List[Int]()
                try:
                    var s_hcf = String("[")
                    for i_h in range(len(hcf)):
                        if i_h > 0: s_hcf = s_hcf + String(",")
                        s_hcf = s_hcf + String(hcf[i_h])
                    s_hcf = s_hcf + String("]")
                    print("[DEBUG] head_cls_formato=" + s_hcf)
                except _:
                    pass
                try:
                    var s_hrf = String("[")
                    for i_h in range(len(hrf)):
                        if i_h > 0: s_hrf = s_hrf + String(",")
                        s_hrf = s_hrf + String(hrf[i_h])
                    s_hrf = s_hrf + String("]")
                    print("[DEBUG] head_reg_formato=" + s_hrf)
                except _:
                    pass
            except _:
                pass
            # quick sanity: if any formato dimension is absurd, bail out
            var MAX_DIM: Int = 1000000
            var bad = False
            try:
                for f in hcf:
                    if f <= 0 or f > MAX_DIM: bad = True; break
            except _:
                pass
            try:
                for f in hrf:
                    if f <= 0 or f > MAX_DIM: bad = True; break
            except _:
                pass
            if total_anchors > 500000 or bad:
                try: print("[ERROR] gerar_predicoes: aborting due to suspicious shapes/anchors", total_anchors)
                except _: pass
                return (List[Float32]()^, List[List[Float32]]()^, List[Float32]()^, List[List[Float32]]()^)
        except _:
            pass

        # Predict per-level using conv heads over FPN maps and return flattened outputs
        var preds_conv = self._fpn_heads_predict_por_nivel(fmaps, anchors_by_level)
        return preds_conv

    # Lightweight backbone: create three downsampled feature maps (simulated convs)
    fn _backbone_forward(mut self, img_matrix: List[List[List[Float32]]], in_size: Int) -> List[List[List[List[Float32]]]]:
        # Lightweight conv+pool backbone using existing CNN helpers
        # Produces P3,P4,P5 as List[level][y][x][c]. This is a small, deterministic
        # MobileNet-like skeleton that uses simple 3x3 convs + avgpool to create
        # multi-scale feature maps. It's intentionally small and deterministic so
        # it can be used for smoke tests and incremental migration.

        print("no _backbone_forward chegou aqui 00")

        var levels: List[List[List[List[Float32]]]] = List[List[List[List[Float32]]]]()
        var H: Int = 0
        var W: Int = 0
        try:
            H = len(img_matrix)
            if H > 0: W = len(img_matrix[0])
            if H == 0 or W == 0:
                return levels.copy()
        except _:
            return levels.copy()

        print("no _backbone_forward chegou aqui 01")

        # flatten grayscale image
        var flat: List[Float32] = List[Float32]()
        for y in range(H):
            for x in range(W):
                var v: Float32 = Float32(0.0)
                try:
                    var p = img_matrix[y][x].copy()
                    v = Float32(0.299) * p[0] + Float32(0.587) * p[1] + Float32(0.114) * p[2]
                except _:
                    v = Float32(0.0)
                flat.append(v)

        print("no _backbone_forward chegou aqui 02")

        # number of channels per fmap (kept small)
        var C: Int = 8
        var strides = List[Int](); strides.append(8); strides.append(16); strides.append(32)

        print("no _backbone_forward chegou aqui 03")

        # ensure we have kernel tensors available (3x3) in bloco_kernels
        try:
            var need = C
            var seed = 1337
            while len(self.bloco_kernels) < need:
                var formato_k = List[Int]()
                formato_k.append(3); formato_k.append(3)
                var k = tensor_defs.Tensor(formato_k^, self.tipo_computacao)
                for i in range(len(k.dados)):
                    seed = (seed * 1664525 + 1013904223 + i) % 2147483647
                    var u = Float32(seed) / Float32(2147483647)
                    k.dados[i] = (u * 2.0 - 1.0) * 0.15
                self.bloco_kernels.append(k^)
        except _:
            pass

        print("no _backbone_forward chegou aqui 04")

        # helper: sample pooled array into desired fh x fw grid
        fn _sample_grid(var pooled: List[Float32], var ph: Int, var pw: Int, var fh: Int, var fw: Int, var channel_idx: Int, var per_channel: Int) -> List[List[Float32]]:
            var out = List[List[Float32]]()
            if ph <= 0 or pw <= 0:
                for y in range(fh):
                    var row = List[Float32]()
                    for x in range(fw): row.append(0.0)
                    out.append(row^)
                return out^
            for y in range(fh):
                var row = List[Float32]()
                for x in range(fw):
                    var src_y = Int(Float32(y) * Float32(ph) / Float32(max(1, fh)))
                    var src_x = Int(Float32(x) * Float32(pw) / Float32(max(1, fw)))
                    if src_y < 0: src_y = 0
                    if src_x < 0: src_x = 0
                    if src_y >= ph: src_y = ph - 1
                    if src_x >= pw: src_x = pw - 1
                    var idx = src_y * pw + src_x
                    var val: Float32 = 0.0
                    try: val = pooled[idx]
                    except _: val = 0.0
                    row.append(val)
                out.append(row^)
            return out^

        #print("no _backbone_forward chegou aqui 05")

        # produce per-level maps
        for s_idx in range(len(strides)):
            #print("no _backbone_forward chegou aqui 05.01")
            var s = strides[s_idx]
            var fh = max(1, in_size // s)
            var fw = fh
            # build fmap for this level by running C conv kernels and pooling
            var fmap: List[List[List[Float32]]] = List[List[List[Float32]]]()
            for c_idx in range(C):
                #print("no _backbone_forward chegou aqui 05.02")
                try:
                    try:
                        print("[DEBUG] backbone: entering channel c_idx=" + String(c_idx) + " fh=" + String(fh) + " fw=" + String(fw) + " in_size=" + String(in_size) + " s=" + String(s))
                    except _:
                        pass
                    try:
                        print("[DEBUG] backbone: bloco_kernels_len=" + String(len(self.bloco_kernels)))
                    except _:
                        pass
                    var kern = self.bloco_kernels[c_idx].copy()
                    try:
                        var kform = List[Int]()
                        try: kform = kern.formato.copy()
                        except _: kform = List[Int]()
                        var s_kf = String("[")
                        for i_kf in range(len(kform)):
                            if i_kf > 0: s_kf = s_kf + String(",")
                            s_kf = s_kf + String(kform[i_kf])
                        s_kf = s_kf + String("]")
                        print("[DEBUG] backbone: kernel_formato=" + s_kf)
                    except _:
                        pass
                    var conv = cnn_impl._conv2d_valid_relu(flat.copy(), H, W, kern.copy(), self.tipo_computacao)
                    var conv_h = H - kern.formato[0] + 1
                    var conv_w = W - kern.formato[1] + 1
                    var pooled = cnn_impl._avgpool2x2_stride2(conv.copy(), conv_h, conv_w, self.tipo_computacao)
                    var ph = conv_h // 2
                    var pw = conv_w // 2
                    if ph <= 0: ph = 1
                    if pw <= 0: pw = 1
                    try:
                        print("[DEBUG] backbone: conv_h=" + String(conv_h) + " conv_w=" + String(conv_w) + " ph=" + String(ph) + " pw=" + String(pw) + " pooled_len=" + String(len(pooled)))
                    except _:
                        pass
                    var sampled = _sample_grid(pooled, ph, pw, fh, fw, c_idx, C).copy()
                    try:
                        var sampled_h = 0
                        var sampled_w = 0
                        try:
                            sampled_h = len(sampled)
                            if sampled_h > 0: sampled_w = len(sampled[0])
                        except _: sampled_h = 0; sampled_w = 0
                        print("[DEBUG] backbone: sampled_h=" + String(sampled_h) + " sampled_w=" + String(sampled_w) + " fh=" + String(fh) + " fw=" + String(fw))
                    except _:
                        pass
                    # merge sampled channel into fmap
                    for y in range(fh):
                        #print("no _backbone_forward chegou aqui 05.03")
                        #print("no _backbone_forward chegou aqui 05.03.01")
                        #print(y)
                        var fmap_len: Int = 0
                        try:
                            fmap_len = len(fmap)
                        except _:
                            fmap_len = 0
                        try:
                            print("[DEBUG] backbone: fmap_len=" + String(fmap_len))
                        except _:
                            pass

                        var row: List[List[Float32]] = List[List[Float32]]()
                        if y < fmap_len:
                            try:
                                row = fmap[y].copy()
                            except _:
                                row = List[List[Float32]]()
                                try: fmap.append(row^)
                                except _: pass
                        else:
                            row = List[List[Float32]]()
                            try: fmap.append(row^)
                            except _: pass

                        #print("no _backbone_forward chegou aqui 05.03.06")

                        for x in range(fw):
                            #print("no _backbone_forward chegou aqui 05.04")
                            # ensure cell exists
                            try:
                                var cell_exists = False
                                try:
                                    if len(fmap) > y and len(fmap[y]) > x:
                                        cell_exists = True
                                except _:
                                    cell_exists = False
                                if cell_exists:
                                    var cell = fmap[y][x].copy()
                                else:
                                    var cell = List[Float32]()
                                    for _ in range(c_idx): cell.append(0.0)
                                    try: fmap[y].append(cell^)
                                    except _: pass
                            except _:
                                var cell = List[Float32]()
                                for _ in range(c_idx): cell.append(0.0)
                                try: fmap[y].append(cell^)
                                except _: pass
                            # append this channel value (safe guard)
                            var sval: Float32 = 0.0
                            try:
                                sval = sampled[y][x]
                            except _:
                                sval = 0.0
                            try:
                                fmap[y][x].append(sval)
                            except _:
                                try:
                                    var cell = List[Float32]()
                                    for _ in range(c_idx): cell.append(0.0)
                                    cell.append(sval)
                                    fmap[y].append(cell^)
                                except _:
                                    pass

                        #print("no _backbone_forward chegou aqui 05.05")
                except _:
                    # on error, fill zeros for this channel

                    #print("no _backbone_forward chegou aqui 05.06")

                    for y in range(fh):
                        try:
                            var row = fmap[y].copy()
                        except _:
                            var row = List[List[Float32]]()
                            fmap.append(row^)
                        for x in range(fw):
                            try:
                                fmap[y][x].append(0.0)
                            except _:
                                var cell = List[Float32]()
                                for _ in range(c_idx): cell.append(0.0)
                                cell.append(0.0)
                                fmap[y].append(cell^)

            #print("no _backbone_forward chegou aqui 05.07")
            # ensure every cell has C channels
            for y in range(len(fmap)):
                #print("no _backbone_forward chegou aqui 05.08")
                for x in range(len(fmap[0])):
                    #print("no _backbone_forward chegou aqui 05.09")
                    var cell = fmap[y][x].copy()
                    if len(cell) < C:
                        var miss = C - len(cell)
                        for _ in range(miss): cell.append(0.0)
                #print("no _backbone_forward chegou aqui 05.10")
            #print("no _backbone_forward chegou aqui 05.11")
            levels.append(fmap^)

        #print("no _backbone_forward chegou aqui 06")

        # simple top-down fusion: upsample coarse maps and average
        try:
            # levels[2]=P5 -> levels[1]=P4
            var P5 = levels[2].copy()
            var P4 = levels[1].copy()
            var P3 = levels[0].copy()
            fn _upsample_avg(src: List[List[List[Float32]]], dst_h: Int, dst_w: Int) -> List[List[List[Float32]]]:
                var out = List[List[List[Float32]]]()
                var sh = len(src)
                var sw = 0
                try:
                    if sh > 0:
                        sw = len(src[0])
                except _:
                    sw = 0
                for y in range(dst_h):
                    var row = List[List[Float32]]()
                    for x in range(dst_w):
                        var sy = Int(Float32(y) * Float32(max(1, sh)) / Float32(max(1, dst_h)))
                        var sx = Int(Float32(x) * Float32(max(1, sw)) / Float32(max(1, dst_w)))
                        if sy < 0: sy = 0
                        if sx < 0: sx = 0
                        if sy >= sh: sy = sh - 1
                        if sx >= sw: sx = sw - 1
                        var val = src[sy][sx]
                        row.append(val.copy()^)
                    out.append(row^)
                return out^
            try:
                var up_p5 = _upsample_avg(P5, len(P4), len(P4[0]))
                for y in range(len(P4)):
                    for x in range(len(P4[0])):
                        var base = P4[y][x]
                        var addv = up_p5[y][x]
                        var sumv: List[Float32] = List[Float32]()
                        for i in range(min(len(base), len(addv))):
                            sumv.append((base[i] + addv[i]) * 0.5)
                        P4[y][x] = sumv^
                levels[1] = P4
            except _:
                pass
            try:
                var up_p4 = _upsample_avg(levels[1], len(P3), len(P3[0]))
                for y in range(len(P3)):
                    for x in range(len(P3[0])):
                        var base = P3[y][x]
                        var addv = up_p4[y][x]
                        var sumv: List[Float32] = List[Float32]()
                        for i in range(min(len(base), len(addv))):
                            sumv.append((base[i] + addv[i]) * 0.5)
                        P3[y][x] = sumv^
                levels[0] = P3
            except _:
                pass
        except _:
            pass

        return levels.copy()

    # Simple FPN heads predictor: for each anchor, pick a level and predict using lightweight linear heads
    fn _fpn_heads_predict_por_nivel(self, fmaps: List[List[List[List[Float32]]]], anchors_by_level: List[List[List[Float32]]]) -> (List[Float32], List[List[Float32]], List[Float32], List[List[Float32]]):
        try:
            try: print("[DBG] _fpn_heads_predict_por_nivel: enter")
            except _: pass
            try: print("[DBG] _fpn_heads_predict_por_nivel: fmaps_levels=" + String(len(fmaps)) + " anchors_levels=" + String(len(anchors_by_level)))
            except _: pass
        except _:
            pass
        var cls_out: List[Float32] = List[Float32]()
        var reg_out: List[List[Float32]] = List[List[Float32]]()
        var mean_out: List[Float32] = List[Float32]()
        var base_out: List[List[Float32]] = List[List[Float32]]()
        var feat: List[Float32] = List[Float32]()
        # Guard: if backbone produced no feature maps, return zeroed predictions
        try:
            if len(fmaps) == 0:
                try: print("[ERROR] _fpn_heads_predict_por_nivel: empty fmaps — returning zero outputs")
                except _: pass
                var total_a: Int = 0
                for lvl in anchors_by_level:
                    total_a = total_a + len(lvl)
                for _ in range(total_a):
                    cls_out.append(0.0)
                    var rrow = List[Float32]()
                    rrow.append(0.0); rrow.append(0.0); rrow.append(0.0); rrow.append(0.0)
                    reg_out.append(rrow^)
                    mean_out.append(0.0)
                    var brow = List[Float32]()
                    brow.append(0.0); brow.append(0.0); brow.append(0.0); brow.append(0.0)
                    base_out.append(brow^)
                return (cls_out^, reg_out^, mean_out^, base_out^)
        except _:
            pass
        try:
            var C = 8
            # prepare head weights from placeholders or defaults
            var hw_cls: List[Float32] = List[Float32]()
            var hw_reg: List[Float32] = List[Float32]()
            var has_pointwise_cls: Bool = False
            var has_pointwise_reg: Bool = False
            try:
                if len(self.head_cls_pesos_conv.formato) >= 2 and self.head_cls_pesos_conv.formato[0] >= C:
                    has_pointwise_cls = True
                elif len(self.head_cls_pesos_conv.dados) >= 1:
                    for i in range(C): hw_cls.append(self.head_cls_pesos_conv.dados[0])
                else:
                    for i in range(C): hw_cls.append(0.001)
            except _:
                for i in range(C): hw_cls.append(0.001)
            try:
                if len(self.head_reg_pesos_conv.formato) >= 2 and self.head_reg_pesos_conv.formato[0] >= C and self.head_reg_pesos_conv.formato[1] >= 4:
                    has_pointwise_reg = True
                elif len(self.head_reg_pesos_conv.dados) >= 4:
                    for j in range(4): hw_reg.append(self.head_reg_pesos_conv.dados[j])
                else:
                    for j in range(4): hw_reg.append(1.0)
            except _:
                for j in range(4): hw_reg.append(1.0)
            var strides = List[Int](); strides.append(8); strides.append(16); strides.append(32)
            # iterate per level so outputs stay in a deterministic flattened order
            for level_idx in range(len(anchors_by_level)):
                try:
                    try: print("[DBG] _fpn: processing level=" + String(level_idx))
                    except _: pass
                except _:
                    pass
                var fmap = List[List[List[Float32]]]()
                try:
                    try:
                        fmap = fmaps[level_idx]
                        try: print("[DBG] _fpn: fmap_len=" + String(len(fmap)) + " row0_len=" + (String(len(fmap[0])) if len(fmap) > 0 else String(0)))
                        except _: pass
                    except _:
                        try: print("[DBG] _fpn: fmap missing for level=" + String(level_idx))
                        except _: pass
                        fmap = List[List[List[Float32]]]()
                except _:
                    fmap = List[List[List[Float32]]]()
                var stride = strides[level_idx] if level_idx < len(strides) else 8 * (1 << level_idx)

                # Precompute per-level cls/reg maps if pointwise heads are available
                var pre_cls_map: List[List[Float32]] = List[List[Float32]]()
                var pre_reg_map: List[List[List[Float32]]] = List[List[List[Float32]]]()
                var precomputed: Bool = False
                try:
                    if (has_pointwise_cls or has_pointwise_reg) and len(fmap) > 0 and len(fmap[0]) > 0:
                        precomputed = True
                        var Hf = len(fmap); var Wf = len(fmap[0])
                        for yy in range(Hf):
                            var row_cls: List[Float32] = List[Float32]()
                            var row_reg: List[List[Float32]] = List[List[Float32]]()
                            for xx in range(Wf):
                                    var cell_feat = fmap[yy][xx].copy()
                                    # cls
                                    var cell_logit: Float32 = 0.0
                                    try:
                                        # spatial head support: formato [kh,kw,C,out_ch]
                                        if len(self.head_cls_pesos_conv.formato) >= 3 and self.head_cls_pesos_conv.formato[2] >= C:
                                            var kh = self.head_cls_pesos_conv.formato[0]
                                            var kw = self.head_cls_pesos_conv.formato[1]
                                            var half_h = kh // 2
                                            var half_w = kw // 2
                                            var outch = 1
                                            try: outch = self.head_cls_pesos_conv.formato[3]
                                            except _: outch = 1
                                            # compute single output channel (outch assumed 1 for cls)
                                            var acc: Float32 = 0.0
                                            for ky in range(kh):
                                                for kx in range(kw):
                                                    var sy = yy + ky - half_h
                                                    var sx = xx + kx - half_w
                                                    var neigh: List[Float32] = List[Float32]()
                                                    if sy >= 0 and sx >= 0 and sy < len(fmap) and sx < len(fmap[0]):
                                                        neigh = fmap[sy][sx].copy()
                                                    else:
                                                        for _ in range(C): neigh.append(0.0)
                                                    for c in range(min(len(neigh), C)):
                                                        var widx = ((ky * kw + kx) * C + c) * outch + 0
                                                        var wval: Float32 = 0.0
                                                        try: wval = self.head_cls_pesos_conv.dados[widx]
                                                        except _: wval = 0.0
                                                        acc = acc + neigh[c] * wval
                                            cell_logit = acc
                                            try: cell_logit = cell_logit + self.head_cls_bias_conv.dados[0]
                                            except _: pass
                                        elif has_pointwise_cls:
                                            for i in range(min(len(cell_feat), self.head_cls_pesos_conv.formato[0])):
                                                cell_logit = cell_logit + cell_feat[i] * self.head_cls_pesos_conv.dados[i * self.head_cls_pesos_conv.formato[1] + 0]
                                            try: cell_logit = cell_logit + self.head_cls_bias_conv.dados[0]
                                            except _: pass
                                        else:
                                            for i in range(min(len(cell_feat), len(hw_cls))):
                                                cell_logit = cell_logit + cell_feat[i] * hw_cls[i]
                                    except _:
                                        cell_logit = 0.0
                                    row_cls.append(cell_logit)

                                    # reg (4 deltas)
                                    var deltas_cell: List[Float32] = List[Float32]()
                                    try:
                                        if len(self.head_reg_pesos_conv.formato) >= 3 and self.head_reg_pesos_conv.formato[2] >= C:
                                            var kh2 = self.head_reg_pesos_conv.formato[0]
                                            var kw2 = self.head_reg_pesos_conv.formato[1]
                                            var half_h2 = kh2 // 2
                                            var half_w2 = kw2 // 2
                                            for j in range(4):
                                                var acc2: Float32 = 0.0
                                                for ky in range(kh2):
                                                    for kx in range(kw2):
                                                        var sy2 = yy + ky - half_h2
                                                        var sx2 = xx + kx - half_w2
                                                        var neigh2: List[Float32] = List[Float32]()
                                                        if sy2 >= 0 and sx2 >= 0 and sy2 < len(fmap) and sx2 < len(fmap[0]):
                                                            neigh2 = fmap[sy2][sx2].copy()
                                                        else:
                                                            for _ in range(C): neigh2.append(0.0)
                                                        for c in range(min(len(neigh2), C)):
                                                            var widx2 = ((ky * kw2 + kx) * C + c) * self.head_reg_pesos_conv.formato[3] + j
                                                            var wval2: Float32 = 0.0
                                                            try: wval2 = self.head_reg_pesos_conv.dados[widx2]
                                                            except _: wval2 = 0.0
                                                            acc2 = acc2 + neigh2[c] * wval2
                                                try: acc2 = acc2 + self.head_reg_bias_conv.dados[j]
                                                except _: pass
                                                deltas_cell.append(acc2)
                                        elif has_pointwise_reg:
                                            for j in range(4):
                                                var dv: Float32 = 0.0
                                                for i in range(min(len(cell_feat), self.head_reg_pesos_conv.formato[0])):
                                                    dv = dv + cell_feat[i] * self.head_reg_pesos_conv.dados[i * self.head_reg_pesos_conv.formato[1] + j]
                                                try: dv = dv + self.head_reg_bias_conv.dados[j]
                                                except _: pass
                                                deltas_cell.append(dv)
                                        else:
                                            var mval_tmp: Float32 = 0.0
                                            for v in cell_feat: mval_tmp = mval_tmp + v
                                            if len(cell_feat) > 0: mval_tmp = mval_tmp / Float32(len(cell_feat))
                                            for j in range(4): deltas_cell.append((mval_tmp - 0.5) * 0.1 * hw_reg[j])
                                    except _:
                                        for j in range(4): deltas_cell.append(0.0)
                                    row_reg.append(deltas_cell^)
                            pre_cls_map.append(row_cls^)
                            pre_reg_map.append(row_reg^)
                except _:
                    precomputed = False

                # iterate anchors and sample the precomputed maps (or compute per-anchor fallback)
                for a in anchors_by_level[level_idx]:
                    var cx: Float32 = 0.0
                    var cy: Float32 = 0.0
                    try:
                        cx = a[0]; cy = a[1]
                    except _:
                        pass
                    var fx = 0; var fy = 0
                    try:
                        fx = Int(cx) // stride
                        fy = Int(cy) // stride
                    except _:
                        fx = 0; fy = 0
                    if fy < 0: fy = 0
                    if fx < 0: fx = 0

                    var feat: List[Float32] = List[Float32]()
                    if len(fmap) == 0 or len(fmap[0]) == 0:
                        for i in range(C): feat.append(0.0)
                    else:
                        if fy >= len(fmap): fy = len(fmap) - 1
                        if fx >= len(fmap[0]): fx = len(fmap[0]) - 1
                        feat = fmap[fy][fx].copy()

                    var mval: Float32 = 0.0
                    for v in feat: mval = mval + v
                    if len(feat) > 0:
                        mval = mval / Float32(len(feat))

                    var logit: Float32 = 0.0
                    var deltas: List[Float32] = List[Float32]()

                    if precomputed:
                        try:
                            logit = pre_cls_map[fy][fx]
                        except _:
                            logit = 0.0
                        try:
                            deltas = pre_reg_map[fy][fx].copy()
                        except _:
                            for j in range(4): deltas.append(0.0)
                    else:
                        # support spatial head application at a given fmap location
                        try:
                            if len(self.head_cls_pesos_conv.formato) >= 3 and self.head_cls_pesos_conv.formato[2] >= C:
                                var kh = self.head_cls_pesos_conv.formato[0]
                                var kw = self.head_cls_pesos_conv.formato[1]
                                var half_h = kh // 2
                                var half_w = kw // 2
                                var acc: Float32 = 0.0
                                for ky in range(kh):
                                    for kx in range(kw):
                                        var sy = fy + ky - half_h
                                        var sx = fx + kx - half_w
                                        var neigh: List[Float32] = List[Float32]()
                                        if sy >= 0 and sx >= 0 and sy < len(fmap) and sx < len(fmap[0]):
                                            neigh = fmap[sy][sx].copy()
                                        else:
                                            for _ in range(C): neigh.append(0.0)
                                        for c in range(min(len(neigh), C)):
                                            var widx = ((ky * kw + kx) * C + c) * 1 + 0
                                            var wval: Float32 = 0.0
                                            try: wval = self.head_cls_pesos_conv.dados[widx]
                                            except _: wval = 0.0
                                            acc = acc + neigh[c] * wval
                                logit = acc
                                try: logit = logit + self.head_cls_bias_conv.dados[0]
                                except _: pass
                            elif has_pointwise_cls:
                                for i in range(min(len(feat), self.head_cls_pesos_conv.formato[0])):
                                    logit = logit + feat[i] * self.head_cls_pesos_conv.dados[i * self.head_cls_pesos_conv.formato[1] + 0]
                                try: logit = logit + self.head_cls_bias_conv.dados[0]
                                except _: pass
                            else:
                                for i in range(min(len(feat), len(hw_cls))):
                                    logit = logit + feat[i] * hw_cls[i]
                        except _:
                            logit = 0.0

                        try:
                            if len(self.head_reg_pesos_conv.formato) >= 3 and self.head_reg_pesos_conv.formato[2] >= C:
                                var kh2 = self.head_reg_pesos_conv.formato[0]
                                var kw2 = self.head_reg_pesos_conv.formato[1]
                                var half_h2 = kh2 // 2
                                var half_w2 = kw2 // 2
                                for j in range(4):
                                    var acc2: Float32 = 0.0
                                    for ky in range(kh2):
                                        for kx in range(kw2):
                                            var sy2 = fy + ky - half_h2
                                            var sx2 = fx + kx - half_w2
                                            var neigh2: List[Float32] = List[Float32]()
                                            if sy2 >= 0 and sx2 >= 0 and sy2 < len(fmap) and sx2 < len(fmap[0]):
                                                neigh2 = fmap[sy2][sx2].copy()
                                            else:
                                                for _ in range(C): neigh2.append(0.0)
                                            for c in range(min(len(neigh2), C)):
                                                var widx2 = ((ky * kw2 + kx) * C + c) * self.head_reg_pesos_conv.formato[3] + j
                                                var wval2: Float32 = 0.0
                                                try: wval2 = self.head_reg_pesos_conv.dados[widx2]
                                                except _: wval2 = 0.0
                                                acc2 = acc2 + neigh2[c] * wval2
                                    try: acc2 = acc2 + self.head_reg_bias_conv.dados[j]
                                    except _: pass
                                    deltas.append(acc2)
                            elif has_pointwise_reg:
                                for j in range(4):
                                    var dv: Float32 = 0.0
                                    for i in range(min(len(feat), self.head_reg_pesos_conv.formato[0])):
                                        dv = dv + feat[i] * self.head_reg_pesos_conv.dados[i * self.head_reg_pesos_conv.formato[1] + j]
                                    try: dv = dv + self.head_reg_bias_conv.dados[j]
                                    except _: pass
                                    deltas.append(dv)
                            else:
                                for j in range(4):
                                    deltas.append((mval - 0.5) * 0.1 * hw_reg[j])
                        except _:
                            for j in range(4): deltas.append(0.0)

                    var base: List[Float32] = List[Float32]()
                    for j in range(4): base.append((mval - 0.5) * 0.1)

                    cls_out.append(logit)
                    reg_out.append(deltas^)
                    mean_out.append(mval)
                    base_out.append(base^)
        except _:
            pass
        return (cls_out^, reg_out^, mean_out^, base_out^)

    fn treinar(mut self, var dataset_dir: String, var altura: Int = 640, var largura: Int = 640,
               var patch_size: Int = 64, var epocas: Int = 5, var taxa_aprendizado: Float32 = 0.05,
               var batch_size: Int = 8, var batch_size_fim: Int = 128, var early_stop: Bool = True, var allowed_classes: List[String] = List[String]()) -> String:
        return trainer.treinar_retina_convfpn(self, dataset_dir, altura, largura, patch_size, epocas, taxa_aprendizado, batch_size, batch_size_fim, early_stop, allowed_classes)

    # A cabeça de regressão (pesos e bias) é armazenada dentro do `bloco_cnn` como
    # `bloco_cnn.peso_saida` e `bloco_cnn.bias_saida`. Esta função retorna uma
    # tupla (peso, bias) pronta para uso e evita duplicar estado dentro do wrapper.
    fn obter_cabeca_regressao(self) -> (tensor_defs.Tensor, tensor_defs.Tensor):
        # Return regression head tensors derived from conv-head placeholders.
        try:
            var n = len(self.head_reg_pesos_conv.dados)
            if n <= 0:
                var empty_w = tensor_defs.Tensor(List[Int](), "cpu")
                var empty_b = tensor_defs.Tensor(List[Int](), "cpu")
                return (empty_w, empty_b)
            var shape_w = List[Int](); shape_w.append(n); shape_w.append(1)
            var tipo = "cpu"
            try:
                tipo = self.backbone_pesos.tipo_computacao
            except _:
                tipo = "cpu"
            var w = tensor_defs.Tensor(shape_w^, tipo)
            for i in range(n):
                w.dados[i] = self.head_reg_pesos_conv.dados[i]
            var shape_b = List[Int](); shape_b.append(1); shape_b.append(n)
            var b = tensor_defs.Tensor(shape_b^, tipo)
            for j in range(n):
                b.dados[j] = 0.0
            return (w, b)
        except _:
            var empty_w = tensor_defs.Tensor(List[Int](), "cpu")
            var empty_b = tensor_defs.Tensor(List[Int](), "cpu")
            return (empty_w, empty_b)

    # Extrai features do patch RGB (patch_size x patch_size) usando tensores internos.
    fn extrair_features_patch(self, var patch_rgb: List[List[List[Float32]]]) -> tensor_defs.Tensor:
        try:
            var ps = self.parametros.patch_size
            var in_shape = List[Int](); in_shape.append(1); in_shape.append(ps * ps)
            var tensor_in = tensor_defs.Tensor(in_shape^, self.tipo_computacao)
            for yy in range(ps):
                for xx in range(ps):
                    var gray: Float32 = Float32(0.0)
                    try:
                        var pix = patch_rgb[yy][xx].copy()
                        gray = Float32(0.299) * pix[0] + Float32(0.587) * pix[1] + Float32(0.114) * pix[2]
                    except _:
                        gray = Float32(0.0)
                    tensor_in.dados[yy * ps + xx] = gray

            # Inline feature extraction using cnn_impl helpers to avoid constructing
            # temporary BlocoCNN instances (ownership/implicit-copy issues).
            var H = ps; var W = ps
            var kh = 3; var kw = 3
            try:
                kh = self.parametros.kernel_h
                kw = self.parametros.kernel_w
            except _:
                kh = 3; kw = 3

            var conv_h = H - kh + 1
            var conv_w = W - kw + 1
            var pool_h = max(1, conv_h // 2)
            var pool_w = max(1, conv_w // 2)
            var feat_dim = self.parametros.num_filtros * pool_h * pool_w

            var formato = List[Int]()
            formato.append(1)
            formato.append(feat_dim)
            var out = tensor_defs.Tensor(formato^, self.tipo_computacao)

            # build img list from tensor_in for cnn_impl conv helper
            var img_list = List[Float32]()
            for i in range(len(tensor_in.dados)):
                img_list.append(tensor_in.dados[i])

            var off = 0
            for f in range(self.parametros.num_filtros):
                try:
                    var conv = dispatcher_tensor.conv2d_valid_relu_dispatch(img_list.copy(), H, W, self.bloco_kernels[f].dados.copy(), kh, kw, self.tipo_computacao)
                    var pooled = dispatcher_tensor.avgpool2x2_stride2_dispatch(conv.copy(), conv_h, conv_w, self.tipo_computacao)
                    for i in range(len(pooled)):
                        out.dados[off + i] = pooled[i]
                    off = off + len(pooled)
                except _:
                    # on error, fill zeros for this filter's features
                    var fill_n = pool_h * pool_w
                    for i in range(fill_n):
                        out.dados[off + i] = 0.0
                    off = off + fill_n

            return out^
        except _:
            var _empty = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
            return _empty^

    # CONFIGURAÇÃO: habilita e inicializa placeholders para o pipeline conv-FPN
    # - `backbone_tipo`: string que descreve o backbone desejado (ex.: "mobilenet_v2" ou "resnet50")
    # - Esta função prepara tensores placeholder, habilita `usar_conv_fpn=True` e
    #   inicializa tensores mínimos para tornar o pipeline conv‑FPN executável.
    #   A implementação atual fornece um backbone leve (esqueleto baseado em
    #   pooling) e cabeças simples para testes e smoke runs; esse esqueleto deve
    #   ser substituído por backbones reais (MobileNet/ResNet) e por uma FPN
    #   completa para produção.
    fn configurar_conv_fpn(mut self, var backbone_tipo_in: String = "mobilenet_v2", var head_kernel_size: Int = 1) -> Bool:
        try:
            var C: Int = 8
            self.usar_conv_fpn = True
            self.backbone_tipo = backbone_tipo_in^
            # pesos vazios; shape real deve ser determinado pela implementação do backbone/FPN
            self.backbone_pesos = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
            self.fpn_pesos = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
            # cabeças conv (pointwise 1x1) placeholders for weights/biases (cls/reg)
            # Initialize per-channel weights so heads can be applied as dot-products
            try:
                var C = 8
                # If user requests spatial heads (kernel>1), initialize tensors with
                # formato [kh, kw, C, out_channels] for cls and [kh, kw, C, 4] for reg.
                if head_kernel_size > 1:
                    var kh = head_kernel_size; var kw = head_kernel_size
                    var shape_cls = List[Int](); shape_cls.append(kh); shape_cls.append(kw); shape_cls.append(C); shape_cls.append(1)
                    self.head_cls_pesos_conv = tensor_defs.Tensor(shape_cls^, self.tipo_computacao)
                    var shape_cb = List[Int](); shape_cb.append(1); shape_cb.append(1)
                    self.head_cls_bias_conv = tensor_defs.Tensor(shape_cb^, self.tipo_computacao)
                    # small random init
                    var seed = 1234
                    for i in range(len(self.head_cls_pesos_conv.dados)):
                        seed = (seed * 1664525 + 1013904223 + i) % 2147483647
                        var u = Float32(seed) / Float32(2147483647)
                        self.head_cls_pesos_conv.dados[i] = (u * 2.0 - 1.0) * 0.01
                    self.head_cls_bias_conv.dados[0] = 0.0
                else:
                    var shape_cls = List[Int](); shape_cls.append(8); shape_cls.append(1)
                    self.head_cls_pesos_conv = tensor_defs.Tensor(shape_cls^, self.tipo_computacao)
                    var shape_cb = List[Int](); shape_cb.append(1); shape_cb.append(1)
                    self.head_cls_bias_conv = tensor_defs.Tensor(shape_cb^, self.tipo_computacao)
                    # default small weights
                    for i in range(len(self.head_cls_pesos_conv.dados)):
                        self.head_cls_pesos_conv.dados[i] = 0.001
                    self.head_cls_bias_conv.dados[0] = 0.0
            except _:
                self.head_cls_pesos_conv = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
            try:
                if head_kernel_size > 1:
                    var kh = head_kernel_size; var kw = head_kernel_size
                    var shape_reg = List[Int](); shape_reg.append(kh); shape_reg.append(kw); shape_reg.append(C); shape_reg.append(4)
                    self.head_reg_pesos_conv = tensor_defs.Tensor(shape_reg^, self.tipo_computacao)
                    var shape_rb = List[Int](); shape_rb.append(1); shape_rb.append(4)
                    self.head_reg_bias_conv = tensor_defs.Tensor(shape_rb^, self.tipo_computacao)
                    var seed2 = 4321
                    for i in range(len(self.head_reg_pesos_conv.dados)):
                        seed2 = (seed2 * 1664525 + 1013904223 + i) % 2147483647
                        var u2 = Float32(seed2) / Float32(2147483647)
                        self.head_reg_pesos_conv.dados[i] = (u2 * 2.0 - 1.0) * 0.01
                    for j in range(len(self.head_reg_bias_conv.dados)):
                        self.head_reg_bias_conv.dados[j] = 0.0
                else:
                    var shape_reg = List[Int](); shape_reg.append(8); shape_reg.append(4)
                    self.head_reg_pesos_conv = tensor_defs.Tensor(shape_reg^, self.tipo_computacao)
                    var shape_rb = List[Int](); shape_rb.append(1); shape_rb.append(4)
                    self.head_reg_bias_conv = tensor_defs.Tensor(shape_rb^, self.tipo_computacao)
                    for i in range(len(self.head_reg_pesos_conv.dados)):
                        self.head_reg_pesos_conv.dados[i] = 0.01
                    for j in range(len(self.head_reg_bias_conv.dados)):
                        self.head_reg_bias_conv.dados[j] = 0.0
            except _:
                self.head_reg_pesos_conv = tensor_defs.Tensor(List[Int](), self.tipo_computacao)
            try:
                print("[DEBUG] configurar_conv_fpn: habilitado pipeline conv-FPN; backbone=", self.backbone_tipo)
            except _:
                pass
            # Diagnostic: print head formatos immediately after initialization
            try:
                var s1 = String("[DEBUG] configurar_conv_fpn: head_cls_formato=")
                try:
                    var tmp = self.head_cls_pesos_conv.formato.copy()
                    var s_tmp = String("[")
                    for ii in range(len(tmp)):
                        if ii > 0: s_tmp = s_tmp + String(",")
                        s_tmp = s_tmp + String(tmp[ii])
                    s_tmp = s_tmp + String("]")
                    s1 = s1 + s_tmp
                except _:
                    s1 = s1 + String("[]")
                print(s1)
            except _:
                pass
            try:
                var s2 = String("[DEBUG] configurar_conv_fpn: head_reg_formato=")
                try:
                    var tmp2 = self.head_reg_pesos_conv.formato.copy()
                    var s_tmp2 = String("[")
                    for jj in range(len(tmp2)):
                        if jj > 0: s_tmp2 = s_tmp2 + String(",")
                        s_tmp2 = s_tmp2 + String(tmp2[jj])
                    s_tmp2 = s_tmp2 + String("]")
                    s2 = s2 + s_tmp2
                except _:
                    s2 = s2 + String("[]")
                print(s2)
            except _:
                pass
            return True
        except _:
            return False

    # INFERÊNCIA (conv‑FPN — via canônica, esqueleto executável)
    # Descrição do pipeline atualmente suportado pelo código:
    # 1) Preprocessamento: redimensionar/crop para `input_size` (função `redimensionar_para_tamanho_entrada`).
    # 2) Backbone (esqueleto atual): extrai mapas de características em múltiplos níveis (P3,P4,P5).
    # 3) FPN (esqueleto): combina/usa os mapas multi‑escala (atualmente pass‑through/pooling).
    # 4) Cabeças conv por nível (esqueleto): aplicam pequenos filtros/pesos que produzem
    #    - logits de classificação por localização/âncora
    #    - deltas de regressão por localização/âncora (tx,ty,tw,th)
    # 5) Decodificação + Pós‑processamento: converte deltas + anchors → boxes e aplica filtragem/NMS.
    #
    # Observações importantes:
    # - O caminho conv‑FPN é a via canônica de inferência e treino usados atualmente.
    # - O código fornece um esqueleto executável para desenvolvimento e smoke tests,
    #   mas deve ser substituído por implementações reais de backbone/FPN/cabeças
    #   para uso em produção (ex.: MobileNetV2/ResNet50 + FPN completo).
    # - O `BlocoCNN` (patch‑based) é mantido apenas por compatibilidade com checkpoints
    #   antigos e pode ser usado como fallback temporário, porém a preferência é usar
    #   o fluxo conv‑FPN.
    fn inferir_convfpn(mut self, img_pixels: List[List[List[Float32]]], var input_size: Int = -1, var max_per_image: Int = -1) -> List[List[Int]]:
        # Simple executable skeleton: generate per-anchor predictions using
        # small patch summaries and placeholder head weights. This is NOT a
        # real conv-FPN implementation but provides a runnable path so the
        # training loop can exercise the new pipeline without `BlocoCNN`.
        var in_size = input_size if input_size > 0 else self.parametros.input_size
        var anchors = gerador_ancoras_pkg.gerar_ancoras(in_size)
        var patched_img = img_pixels
        try:
            var resized_tuple = self.redimensionar_para_tamanho_entrada(img_pixels.copy(), List[List[Int]](), in_size)
            patched_img = resized_tuple[0]
        except _:
            patched_img = img_pixels

        var preds = self.gerar_predicoes_por_ancora_convfpn(patched_img, anchors, self.parametros.patch_size)
        
        var cls_logits = preds[0]
        var reg_deltas = preds[1]
        var cls_scores: List[Float32] = List[Float32]()

        

        for i in range(len(cls_logits)):
            var l = cls_logits[i]
            var s: Float32 = 0.0
            try:
                if l >= 50.0:
                    s = 1.0
                elif l <= -50.0:
                    s = 0.0
                else:
                    s = 1.0 / (1.0 + Float32(math.exp(-Float64(l))))
            except _:
                s = 0.0
            cls_scores.append(s)

        # decode identical to regular inferir after dividing by weights
        var boxes: List[List[Float32]] = List[List[Float32]]()
        for i in range(len(anchors)):
            var a = anchors[i].copy()
            var dx = reg_deltas[i][0]; var dy = reg_deltas[i][1]; var dw = reg_deltas[i][2]; var dh = reg_deltas[i][3]
            try:
                if len(self.bbox_reg_weights) >= 4:
                    dx = dx / self.bbox_reg_weights[0]
                    dy = dy / self.bbox_reg_weights[1]
                    dw = dw / self.bbox_reg_weights[2]
                    dh = dh / self.bbox_reg_weights[3]
            except _:
                pass
            if dx > 3.0: dx = 3.0
            if dx < -3.0: dx = -3.0
            if dy > 3.0: dy = 3.0
            if dy < -3.0: dy = -3.0
            if dw != dw: dw = 0.0
            if dh != dh: dh = 0.0
            var cx = a[0] + dx * a[2]
            var cy = a[1] + dy * a[3]
            if dw > 4.0: dw = 4.0
            if dw < -4.0: dw = -4.0
            if dh > 4.0: dh = 4.0
            if dh < -4.0: dh = -4.0
            # finalize decoded box for this anchor
            var w = a[2] * Float32(math.exp(Float64(dw)))
            var h = a[3] * Float32(math.exp(Float64(dh)))
            var x0 = cx - w/2.0
            var y0 = cy - h/2.0
            var x1 = cx + w/2.0
            var y1 = cy + h/2.0
            var outb = List[Float32]()
            outb.append(x0); outb.append(y0); outb.append(x1); outb.append(y1)
            boxes.append(outb^)

        # Simple postprocess: convert to integer boxes and filter by score
        var results: List[List[Int]] = List[List[Int]]()
        for i in range(len(boxes)):
            try:
                if cls_scores[i] <= 0.01:
                    continue
            except _:
                pass
            var b = boxes[i]
            var ib = List[Int]()
            var x0v = b[0]
            if x0v < 0.0: x0v = 0.0
            var y0v = b[1]
            if y0v < 0.0: y0v = 0.0
            var x1v = b[2]
            if x1v < 0.0: x1v = 0.0
            var y1v = b[3]
            if y1v < 0.0: y1v = 0.0
            ib.append(Int(x0v)); ib.append(Int(y0v)); ib.append(Int(x1v)); ib.append(Int(y1v))
            results.append(ib^)
            if max_per_image > 0 and len(results) >= max_per_image:
                break

        return results

# Module-level wrapper to allow callers to invoke the instance method
# without relying on method dispatch/visibility rules in the caller.
fn gerar_predicoes_por_ancora_convfpn_module(mut detector: RetinaFace, img_pixels: List[List[List[Float32]]], anchors: List[List[Float32]], patch_size: Int) -> (List[Float32], List[List[Float32]], List[Float32], List[List[Float32]]):
    #print("na geração de predições convfpn module chegou aqui 00")
    return detector.gerar_predicoes_por_ancora_convfpn(img_pixels, anchors, patch_size)
