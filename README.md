# BIONIX-ML-VIVAZ (.NET)

<p align="center">
	<img src="ICONE.png" alt="Ícone do BIONIX-ML-VIVAZ" width="160">
</p>

Conjunto de projetos .NET que consomem `Bionix.ML` para detecção e reconhecimento facial, anotação de datasets e uma API HTTP.

Visão geral

Este repositório engloba modelos e integracões para detecção e reconhecimento facial, além de ferramentas auxiliares:

- `src/DetectorModel/` — adaptadores/serviços de detecção (RetinaFace-stub)
- `src/IdentificadorModel/` — extrai embeddings e realiza identificação (ArcFace-stub)
- `src/Vivaz.Api/` — ASP.NET Core Web API com endpoints de detecção e identificação
- `src/AnotadorDeDataSet/` — ferramenta desktop para anotação de dataset

Layout e artefatos

Organização típica:

```
src/
	DetectorModel/
	IdentificadorModel/
	Vivaz.Api/
	AnotadorDeDataSet/
	DATASET/    ← local, não versionado
	MODELO/     ← checkpoints, não versionado
```

Como compilar e executar

1. Restaurar dependências e build:

```
dotnet restore
dotnet build -c Release
```

2. Executar a API (desenvolvimento):

```
dotnet run --project src/Vivaz.Api -c Release
```

3. Executar o anotador de dataset (WinForms):

```
dotnet run --project src/AnotadorDeDataSet -f net8.0-windows
```

API e endpoints

A API expõe endpoints para health, detecção e identificação. Exemplo (resumo):

- `GET /api/face/health` — health check
- `POST /api/face/detect` — recebe BMP e retorna bounding boxes
- `POST /api/face/compare` — recebe duas imagens e retorna score/verificação
- `POST /api/face/embedding` — retorna embedding de 128 floats

Treinar do zero (IdentificadorLeve / DetectorLeve)
-----------------------------------------------
Os runners suportam iniciar do zero e retomar a partir de checkpoints salvos em `PESOS/`.

- Treinar `IdentificadorLeve` do zero (PowerShell):

```
# remover checkpoints antigos (opcional)
if (Test-Path PESOS\IDENTIFICADOR_LEVE) { Remove-Item -Recurse PESOS\IDENTIFICADOR_LEVE }
$env:RESUME='0'; $env:INITIAL_LR='0.005'; $env:EPOCHS='1000'; $env:LOSS_THRESHOLD='0.0005'; $env:SUPPRESS_OUTPUTS='1';
dotnet run --project src/IdentificadorLeveModel.Runner/IdentificadorLeveModel.Runner.csproj -c Debug
```

- Treinar `DetectorLeve` do zero (PowerShell):

```
# quick test (saves image and exits)
$env:QUICK_TEST_SAVE='1'; dotnet run --project src/DetectorLeveModel.Runner/DetectorLeveModel.Runner.csproj -c Debug -- --mode train

# full train (resume support handled via env RESUME)
$env:RESUME='0'; $env:EPOCHS='1000'; dotnet run --project src/DetectorLeveModel.Runner/DetectorLeveModel.Runner.csproj -c Debug -- --mode train
```

Retomar treinos
----------------
Para retomar a partir de um checkpoint coloque os arquivos de checkpoint em `PESOS/<NOME>` e execute com `RESUME=1`:

```
$env:RESUME='1'; dotnet run --project src/IdentificadorLeveModel.Runner/IdentificadorLeveModel.Runner.csproj -c Debug
```

WASM / API — como usar
-----------------------
O projeto inclui duas formas de expor inferência WASM via a API:

- Endpoints tradicionais (API):
	- `POST /api/face/detect` — envia um arquivo (`form` `file`) retorna PNG crop ou JSON detections
	- `POST /api/face/detectjson` — retorna JSON de detections

- Endpoints WASM-backed (o runtime usa `Vivaz.WASM` internamente):
	- `POST /api/face/wasm/detectjson` — aceita `form` `file`, retorna JSON
	- `POST /api/face/wasm/detectcrop` — aceita `form` `file`, retorna PNG crop
	- `POST /api/face/wasm/embed` — aceita `form` `file`, retorna `{ embedding: [...] }`
	- `POST /api/face/wasm/compare` — aceita dois arquivos no `form` (ordem: primeiro, segundo), retorna `{ percent, same }`

Exemplos com `curl` (API na porta 5000):

```bash
curl -X POST -F "file=@face.jpg" http://localhost:5000/api/face/wasm/detectjson
curl -X POST -F "file=@face.jpg" http://localhost:5000/api/face/wasm/detectcrop --output crop.png
curl -X POST -F "file=@a.jpg" -F "file=@b.jpg" http://localhost:5000/api/face/wasm/compare
```

Embedding weights into `Vivaz.WASM` (offline WASM)
-------------------------------------------------
If you want `Vivaz.WASM` to include model weights as embedded resources (so the API can use WASM without reading external `PESOS`), copy the target weights folder into `src/Vivaz.WASM/PESOS/<FOLDER>` before building the `Vivaz.WASM` project. Example:

```
# copy current trained weights into the WASM project (one-time)
cp -r PESOS/IDENTIFICADOR_LEVE src/Vivaz.WASM/PESOS/IDENTIFICADOR_LEVE
dotnet build src/Vivaz.WASM/Vivaz.WASM.csproj -c Release
```

Docker (build + run)
--------------------
This repository includes Dockerfiles and a `docker-compose.yml` to run the API and Demo.

From the repository root (`BIONIX-ML-VIVAZ`):

```powershell
docker compose up --build
```

This starts two services by default:

- `vivaz_api` mapped to container port `80` → host `5001` in compose (configurable in `docker-compose.yml`)
- `vivaz_demo` mapped to container port `80` → host `5000`

Notes:
- To embed WASM weights inside the `Vivaz.WASM` assembly before building the images, copy the `PESOS/<FOLDER>` into `src/Vivaz.WASM/PESOS` then run `docker compose build` so the embedded resources are included in the build stage.
- The `Vivaz.WASM` project is a library and does not run standalone — the Dockerfile for WASM is build-only to produce artifacts.


Modelos e dataset

Coloque checkpoints em `src/MODELO/` e imagens em `src/DATASET/` (ambas pastas não versionadas). Este repositório inclui suporte para o dataset CelebA com landmarks. Estrutura esperada:

- `DATASET/faces_com_landmarks/list_landmarks_align_celeba.csv` — anotações de landmarks (CSV)
- `DATASET/faces_com_landmarks/list_bbox_celeba.csv` — bounding boxes (CSV, opcional)
- `DATASET/faces_com_landmarks/img_align_celeba/img_align_celeba/` — imagens JPG

O `DataLoader` em `src/DetectorModel/dados/DataLoader.cs` detecta automaticamente o formato CelebA (CSV) e carrega landmarks e bounding boxes quando presentes. Para usar outro dataset (por exemplo WIDER), mantenha o formato original de anotações e o loader fará fallback.

Para produção, exporte os pesos para um formato portátil (por exemplo ONNX) se for necessário interoperar com outros runtimes.

Retomar treino (checkpoints)
-----------------------------
O runner salva checkpoints completos em `PESOS/DETECTOR` e suporta retomada exata do treino.

Formato de checkpoint (arquivo):

- `*.bin` — pesos (ex.: `head_p3_cls.bin`, `stage2_0_conv1.bin`)
- `*.bias.bin` — biases correspondentes (ex.: `head_p3_cls.bias.bin`)
- `*.grad.bin` — gradientes (quando presentes)
- `opt_slot_*.bin` — arquivos de slots do otimizador (velocidades/momentums)
- `opt_meta.json` — metadados do otimizador (slots, lr, momentum, timestamp)
- `meta.json` — metadados do checkpoint contendo pelo menos: `epoch`, `lr`, `timestamp`, `rngSeed`, `processedSamples`

Para retomar automaticamente, execute o runner com a flag `--resume` ou defina a variável de ambiente `RESUME=1`:

```
dotnet run --project src/Bionix.ML.Vivaz.Runner -c Debug -- --resume
```

O fluxo de retomada realiza:

- leitura de `meta.json` para recuperar número da época, `rngSeed` e `processedSamples` (offset de amostras processadas);
- leitura de todos os pesos salvos (`*.bin` / `*.bias.bin`) e cópia para os tensores do modelo quando as formas coincidirem;
- carregamento dos slots do otimizador (`opt_slot_*.bin`) se existirem — o loader valida o `opt_meta.json` e ignora slots cuja forma não coincida com o modelo atual;

Observações e boas práticas:

- Para retomada exata, certifique-se de que `opt_slot_*.bin` e `opt_meta.json` estejam presentes e que o `meta.json` contenha os campos esperados (`rngSeed` e `processedSamples`).
- Checkpoints são escritos de forma atômica (escrito em diretório temporário e substituído), reduzindo o risco de arquivos corrompidos.
- O `meta.json.processedSamples` permite pular as N primeiras amostras ao reiniciar para continuar exatamente de onde parou.


Conversão de imagens

Existe um utilitário Python para conversão/normalização de imagens (se aplicável). Instale `Pillow` se for usar scripts de conversão.

Observações

Os projetos expõem pontos de partida (stubs) para detecção e reconhecimento. É necessário integrar o carregamento de checkpoints e a lógica de inferência para tornar os endpoints produtivos. Para cenários cross-platform considere migrar utilitários de imagem para ImageSharp/SkiaSharp.

Ajuda e contribuições
Abra issues descrevendo o cenário (incluir versão .NET, SO, passos para reproduzir). Pull requests com exemplos funcionais e instruções claras são bem-vindos.
