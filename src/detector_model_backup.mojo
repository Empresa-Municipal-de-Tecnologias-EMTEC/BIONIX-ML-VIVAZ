import bionix_ml.camadas.cnn as cnn_pkg
import bionix_ml.computacao.dispatcher_tensor as dispatcher
import bionix_ml.nucleo.Tensor as tensor_defs
import bionix_ml.dados as dados_pkg
import bionix_ml.graficos as graficos_pkg
import math
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import bionix_ml.camadas.cnn.cnn as cnn_impl
import bionix_ml.autograd.tipos_mlp as tipos_mlp
import bionix_ml.perdas.bce as bce_perdas
import bionix_ml.computacao.sessao as sessao_driver
import bionix_ml.computacao.storage_sessao as storage_sessao
import os
import adaptadores.detectar_face as detect_pkg
import bionix_ml.dados.bmp as bmpmod
import bionix_ml.uteis as uteis
import io_modelo

import bionix_ml.uteis.arquivo as arquivo_io
import os

# Backup copy of detector_model.mojo before refactor
