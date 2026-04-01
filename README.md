# BIONIX-ML-VIVAZ

<p align="center">
	<img src="ICONE.png" alt="Ícone do BIONIX-ML-VIVAZ" width="160">
</p>

Modelos e APIs para detecção e reconhecimento facial construídos sobre o framework **BIONIX-ML**.

O repositório contém dois sistemas:

- **Detector RetinaFace** — localiza faces em imagens BMP, produz bounding boxes.
- **Reconhecedor ArcFace** — extrai embeddings de face de 128 dimensões e identifica ou verifica pessoas.

Ambos são expostos de três formas:
- **API HTTP** (`api/run_api_server.mojo`) — servidor TCP/HTTP sequencial com 8 endpoints.
- **Módulo WebAssembly** (`wasm/arcface_wasm.c`) — inferência ArcFace compilada com Emscripten para rodar direto no browser.
- **Demo browser** (`build/index.html`) — página standalone com câmera em tempo-real, modo live e verificação de par.

Importante: o repositório `BIONIX-ML` deve estar clonado como diretório irmão.

Repositório do framework principal: https://github.com/Empresa-Municipal-de-Tecnologias-EMTEC/BIONIX-ML

---

## Pré-requisitos

| Ferramenta | Versão mínima | Finalidade |
|---|---|---|
| [pixi](https://prefix.dev/) | qualquer | gerenciador de ambiente / tasks |
| Mojo (via pixi) | 0.25.6 | compilar todos os módulos `.mojo` |
| Python 3.8+ | — | converter imagens do dataset |
| [Pillow](https://pypi.org/project/Pillow/) | — | `convert_images_to_bmp.py` |
| [Emscripten](https://emscripten.org/docs/getting_started/downloads.html) | 3.x | compilar o módulo WASM (opcional) |

O pixi instala o Mojo automaticamente. Tudo mais é opcional conforme os módulos que serão usados.

---

## Layout de diretórios

```
parent/
├─ BIONIX-ML/          ← framework (deve existir como irmão)
└─ BIONIX-ML-VIVAZ/
   ├─ src/
   │   ├─ run_retina_train.mojo   ← treino do detector
   │   ├─ run_retina_infer.mojo   ← inferência do detector
   │   ├─ run_arcface_train.mojo  ← treino do reconhecedor
   │   ├─ run_arcface_infer.mojo  ← inferência do reconhecedor
   │   ├─ reconhecedor/           ← pacote ArcFace
   │   ├─ retina/                 ← pacote RetinaFace
   │   ├─ api/                    ← servidor HTTP
   │   ├─ wasm/                   ← módulo WebAssembly (C + Makefile)
   │   ├─ DATASET/                ← dataset local (ignorado pelo git)
   │   ├─ MODELO/                 ← checkpoints (ignorado pelo git)
   │   └─ build/                  ← artefatos compilados + index.html
   └─ convert_images_to_bmp.py
```

---

## Configuração do ambiente

```bash
# 1. Clone os dois repositórios lado a lado
git clone <url>/BIONIX-ML
git clone <url>/BIONIX-ML-VIVAZ

# 2. Instale o pixi (se ainda não tiver)
curl -fsSL https://pixi.sh/install.sh | bash

# 3. Entre na pasta src do projeto e instale o ambiente Mojo
cd BIONIX-ML-VIVAZ/src
pixi install
```

---

## Detector de faces (RetinaFace)

### Treinar

```bash
# Interpretado (mais rápido para iterar)
cd src
pixi run run_retina_train

# Compilado com debug (build + executa)
pixi run run_retina_train_debug

# Só compilar (separado da execução)
pixi run run_retina_train_debug_build
./run_retina_train
```

O modelo é salvo em `MODELO/retina_modelo/`.

### Inferir

```bash
pixi run run_retina_infer
```

---

## Reconhecedor de faces (ArcFace)

### Treinar

```bash
cd src

# Interpretado
pixi run run_arcface_train

# Compilado + debug
pixi run run_arcface_train_debug

# Só compilar
pixi run run_arcface_train_debug_build
./run_arcface_train
```

Os checkpoints são salvos em `MODELO/arcface_modelo/`:

```
arcface_state.txt        ← epoch, lr, patch_size, embed_dim, num_classes
kernels.bin              ← pesos da camada CNN
proj_peso.bin            ← matriz de projeção (D→128)
proj_bias.bin
cls_peso.bin             ← cabeça de classificação
cls_bias.bin
cnn_peso_saida.tensor.txt
```

### Inferir / identificar

```bash
# Interpretado
pixi run run_arcface_infer

# Compilado + debug
pixi run run_arcface_infer_debug
```

O script constrói uma galeria a partir de `DATASET/` e identifica faces chamando `bionix_identify_rgba`.

---

## API HTTP

O servidor expõe 8 endpoints na porta 8080 (padrão). Aceita imagens **BMP** no corpo das requisições.

### Endpoints

| Método | Caminho | Corpo | Resposta |
|---|---|---|---|
| GET | `/health` | — | `{"status":"ok","version":"1.0.0"}` |
| POST | `/detectar` | BMP bytes | `{"boxes":[[x0,y0,x1,y1],...]}` |
| POST | `/identificar` | BMP bytes | `{"identidade":"nome","score":0.95,"box":[...]}` |
| POST | `/verificar_par` | `uint32_LE(tam_A)` + BMP_A + BMP_B | `{"mesma_pessoa":true,"score":0.92}` |
| POST | `/embedding` | BMP bytes | `{"embedding":[...128 floats]}` |
| GET | `/galeria` | — | `{"identidades":["A","B",...],"total":46}` |
| POST | `/galeria/construir` | — | `{"ok":true,"classes":46}` |
| POST | `/galeria/adicionar` | `byte(name_len)` + nome + BMP bytes | `{"ok":true,"galeria_size":47}` |

### Compilar e executar

```bash
cd src

# Compilar versão release (move também os pesos de MODELO/ para build/)
pixi run build_api

# Compilar versão debug
pixi run build_api_debug

# Executar (porta 8080, dataset em DATASET/)
pixi run run_api

# Ou diretamente com argumentos customizados
./build/bionix_api <porta> <caminho_dataset>
```

### Exemplos curl

```bash
# Health check
curl http://localhost:8080/health

# Detectar faces em uma imagem
curl -X POST --data-binary @foto.bmp http://localhost:8080/detectar

# Identificar quem está na foto
curl -X POST --data-binary @foto.bmp http://localhost:8080/identificar

# Verificar se duas fotos são da mesma pessoa
# (corpo: 4 bytes LE com tamanho da foto A, depois foto A, depois foto B)
python3 -c "
import struct, sys
a = open('face_a.bmp','rb').read()
b = open('face_b.bmp','rb').read()
sys.stdout.buffer.write(struct.pack('<I', len(a)) + a + b)
" | curl -X POST --data-binary @- http://localhost:8080/verificar_par
```

---

## Módulo WebAssembly

O módulo `wasm/arcface_wasm.c` implementa o mesmo pipeline de inferência ArcFace em C puro para rodar no browser via Emscripten.

### Pré-requisito: Emscripten

```bash
# Via apt (Ubuntu/Debian)
sudo apt install emscripten

# Ou via emsdk
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest
source ./emsdk_env.sh
```

### Exportar pesos (bundle binário)

O bundle concentra todos os pesos do modelo ArcFace em um único arquivo para download pelo browser.
**Requer modelo treinado em `MODELO/arcface_modelo/`.**

```bash
cd src
pixi run build_wasm_bundle
# Gera em build/:
#   arcface_bundle.bin   ← pesos CNN + projeção + classificação (~3.8 MB)
#   arcface_gallery.bin  ← embeddings da galeria
#   arcface_gallery.json ← nomes das identidades
```

### Compilar o módulo WASM

```bash
pixi run build_wasm        # release (otimizado)
pixi run build_wasm_debug  # debug com asserts Emscripten
# Gera em build/:
#   arcface.js    ← glue JavaScript do Emscripten
#   arcface.wasm  ← binário WebAssembly
```

### Demo browser

Após gerar os artefatos acima, sirva a pasta `build/` com qualquer servidor HTTP estático:

```bash
cd src/build

# Python (mais simples)
python3 -m http.server 9000

# Node
npx serve .
```

Abra `http://localhost:9000/index.html`. A página:
- Carrega o modelo WASM e pesos automaticamente.
- Abre a câmera e identifica faces em tempo-real (modo live) ou a cada clique em **Identificar**.
- Permite verificar se dois arquivos de imagem são da mesma pessoa (**Verificar Par**).

---

## Distribuição completa

Para gerar todos os artefatos de uma vez (bundle + WASM + API):

```bash
cd src
pixi run dist
```

Conteúdo resultante de `build/`:

```
build/
├─ bionix_api          ← binário da API HTTP
├─ arcface.js          ← glue Emscripten
├─ arcface.wasm        ← módulo WebAssembly
├─ arcface_bundle.bin  ← pesos do modelo (WASM)
├─ arcface_gallery.bin ← galeria de embeddings
├─ arcface_gallery.json
├─ index.html          ← demo browser
└─ MODELO/             ← pesos para a API nativa
   ├─ arcface_modelo/
   └─ retina_modelo/
```

---

## Dataset

Para treinar e avaliar o sistema de reconhecimento facial usaremos o dataset VGGFace2 disponível em:

https://www.kaggle.com/datasets/hearfool/vggface2

Baixe e extraia o conteúdo do dataset para `src/DATASET` (ou para o caminho configurado). O diretório `src/DATASET` está ignorado pelo git.

## Conversão de imagens

Existe um script na raiz do repositório chamado `convert_images_to_bmp.py` que percorre recursivamente o dataset e converte imagens para o formato BMP.

Pré-requisitos:

- Instale Python 3.8 ou superior (recomendado: Python 3.10+).
- Instale a biblioteca Pillow:

```
pip install pillow
```

Uso (execute a partir da raiz do repositório, onde está `convert_images_to_bmp.py`):

```
python convert_images_to_bmp.py                # usa src/DATASET, apaga originais após conversão
python convert_images_to_bmp.py --dataset DATASET --overwrite
python convert_images_to_bmp.py --dry-run     # mostra ações sem gravar (não apaga arquivos)
```

Opções úteis do script:

- `--dataset/-d`: caminho do dataset (padrão `src/DATASET`)
- `--overwrite/-o`: sobrescrever arquivos BMP existentes
- O script remove os arquivos originais após uma conversão bem-sucedida por padrão.
- `--dry-run`: simular ações sem gravar arquivos

Em WSL / Linux você pode criar um ambiente virtual para isolar dependências:

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pillow
python convert_images_to_bmp.py
```

Observação: o diretório `src/DATASET` já está listado em `.gitignore`.

