# BIONIX-ML-VIVAZ

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
O repositório expõe endpoints tradicionais e endpoints "WASM-backed" — estes últimos usam a biblioteca `Vivaz.WASM` internamente no servidor para executar inferência.

Principais endpoints (resumo):

- `POST /api/face/detect` — envia um arquivo (`form` campo `file`), retorna uma imagem PNG recortada ou JSON com possíveis detecções.
- `POST /api/face/detectjson` — envia um arquivo (`form` campo `file`), retorna JSON com todas as detecções.
- `POST /api/face/wasm/detectjson` — mesma funcionalidade que `detectjson`, porém executa através do runtime WASM (`Vivaz.WASM`) no servidor.
- `POST /api/face/wasm/detectcrop` — envia `file`, retorna PNG com o crop consensual detectado (ou 404 se não encontrar).
- `POST /api/face/wasm/embed` — envia `file`, retorna JSON `{ "embedding": [ ... ] }` com o vetor de embedding.
- `POST /api/face/wasm/compare` — envia dois arquivos no `form` (primeiro e segundo), retorna JSON `{ "percent": <num>, "same": <bool> }`.

Exemplos rápidos com `curl` (API local padrão):

```bash
curl -X POST -F "file=@face.jpg" http://localhost:5000/api/face/wasm/detectjson
curl -X POST -F "file=@face.jpg" http://localhost:5000/api/face/wasm/detectcrop --output crop.png
curl -X POST -F "file=@a.jpg" -F "file=@b.jpg" http://localhost:5000/api/face/wasm/compare
```

Como usar `Vivaz.WASM` a partir do navegador
-------------------------------------------
O servidor oferece endpoints WASM-backed que podem ser consumidos diretamente pelo navegador via `fetch` / `FormData`.

Modelo de integração (página de exemplo já incluída em `src/Vivaz.Demonstracao/wwwroot`):

- Estrutura HTML mínima usada nas demos:

```html
<video id="video" autoplay playsinline width="320" height="240"></video>
<canvas id="canvas" style="display:none"></canvas>
<button id="capA">Capture A</button>
<button id="capB">Capture B</button>
<button id="compare">Compare</button>
<div>A: <img id="imgA" width="160"></div>
<div>B: <img id="imgB" width="160"></div>
<div id="result"></div>
<script src="/js/compare.js"></script>
<script>
	window.demoCompareConfig = {
		embedEndpoint: '/api/face/wasm/embed',
		compareEndpoint: '/api/face/wasm/compare',
		thresholdPercent: 70
	};
</script>
```

- A lógica comum está em `src/Vivaz.Demonstracao/wwwroot/js/compare.js`. Ela:
	- inicializa a câmera via `navigator.mediaDevices.getUserMedia`,
	- captura imagens para um `canvas`,
	- envia imagens ao endpoint de embedding (`embedEndpoint`) para obter os vetores,
	- faz comparação localmente (produto interno / normalização) quando os embeddings estão disponíveis, ou recorre ao endpoint de comparação do servidor (`compareEndpoint`) como fallback.

Modo cliente (WebAssembly)
--------------------------
É possível executar `Vivaz.WASM` inteiramente no navegador compilando a biblioteca para WebAssembly e servindo os artefatos gerados junto à página de demonstração. O fluxo recomendado:

- Compile `Vivaz.WASM` para um runtime WebAssembly (ex.: `browser-wasm`) usando o SDK apropriado ou ferramenta de empacotamento. Um exemplo genérico (ajuste conforme seu toolchain):

```bash
# publique artefatos wasm em uma pasta local
dotnet publish src/Vivaz.WASM/Vivaz.WASM.csproj -c Release -r browser-wasm -o src/Vivaz.WASM/wasm_publish --no-self-contained
```

Para instruções passo-a-passo sobre como publicar e disponibilizar os artefatos WASM para a demo (inclui comandos exatos e recomendações sobre MIME/types e Git LFS), veja também: [Instruções para publicar/usar WASM](src/Vivaz.WASM/INSTRUCOES.md).

Importante — pesos e URL da API
--------------------------------
O runtime cliente (WASM) precisa acessar os arquivos de pesos. Por convenção o `Vivaz.Api` serve os pesos em um endpoint estático `GET /pesos/<NOME>.zip` — por exemplo `http://localhost:5000/pesos/CLASSIFICADOR_DETECTOR_LEVE.zip`.

Antes de iniciar a API de demonstração copie as pastas de pesos para o diretório público do projeto API (ex.: `src/Vivaz.Api/wwwroot/pesos/`). Exemplo (PowerShell):

```powershell
# copia pesos para a pasta pública do servidor API
Copy-Item -Recurse PESOS\CLASSIFICADOR_DETECTOR_LEVE src\Vivaz.Api\wwwroot\pesos\CLASSIFICADOR_DETECTOR_LEVE
# Opcional: empacote como zip para que o WASM baixe um único arquivo por convenção
Compress-Archive -Path src\Vivaz.Api\wwwroot\pesos\CLASSIFICADOR_DETECTOR_LEVE -DestinationPath src\Vivaz.Api\wwwroot\pesos\CLASSIFICADOR_DETECTOR_LEVE.zip -Force
```

Ao iniciar o demo/aplicação cliente, defina a variável de ambiente `VIVAZ_API_URL` apontando para a base da sua API (ex.: `http://localhost:5000`). O `Vivaz.WASM` tentará baixar `VIVAZ_API_URL/pesos/CLASSIFICADOR_DETECTOR_LEVE.zip` ou `.../IDENTIFICADOR_LEVE.zip` caso não encontre pesos embutidos ou locais.

Instalar ferramentas WASM no .NET
--------------------------------
Para compilar para `browser-wasm` instale o workload `wasm-tools` no .NET SDK (uma vez por máquina):

```powershell
dotnet workload install wasm-tools
```


- Copie os arquivos resultantes (`vivaz.wasm`, `vivaz.js`, `_framework` etc. dependendo do toolchain) para a pasta de arquivos estáticos da demonstração: `src/Vivaz.Demonstracao/wwwroot/wasm/`.
- O `Vivaz.Demonstracao` (demo) agora tenta carregar automaticamente um loader JS em `/wasm/` e, se encontrar uma implementação cliente, usará o runtime WebAssembly local para gerar embeddings e fazer comparações sem roundtrip ao servidor.

Arquivos de suporte e fallback
-----------------------------
O demo inclui um pequeno loader/fallback em `src/Vivaz.Demonstracao/wwwroot/js/vivaz-wasm-loader.js` que tenta carregar artefatos em `/wasm/` e expõe a API `window.vivazWasm`:

- `window.vivazWasm.ready` — Promise que resolve quando o loader está pronto.
- `window.vivazWasm.embedFromBlob(blob)` — retorna `{ embedding: [...] }` quando suportado.
- `window.vivazWasm.compareBlobs(a,b)` — retorna `{ percent, same }` quando suportado.

Se o loader não encontrar artefatos wasm, ele faz fallback às chamadas HTTP para os endpoints WASM-backed do servidor (`/api/face/wasm/embed`, `/api/face/wasm/compare`).

Servindo o WASM com Docker (demo)
---------------------------------
O `Dockerfile.demo` foi atualizado para copiar automaticamente qualquer artefato presente em `src/Vivaz.WASM/wasm_publish/` para `wwwroot/wasm/` do demo durante a construção da imagem. Assim, para publicar a demo com o WASM embutido:

1. Publique os artefatos WASM em `src/Vivaz.WASM/wasm_publish/` (veja comando acima).
2. Construa a imagem demo normalmente:

```bash
docker build -f Dockerfile.demo -t vivaz_demo:wasm .
docker run -p 80:80 vivaz_demo:wasm
```

Observações
-----------
- O processo exato de compilação para WebAssembly depende do SDK/toolchain que você escolher (por exemplo: Blazor WebAssembly, dotnet-wasm toolchain, ou um empacotador AOT). Ajuste as instruções acima conforme necessário.
- Não houve alteração no projeto `Vivaz.WASM` além de instruções — você pode produzir o WASM usando a sua ferramenta preferida. O repositório agora inclui suporte no demo para servir e usar esses artefatos no cliente.

Portanto, para integrar ao seu site, reutilize a estrutura de HTML e o `compare.js` (ou implemente chamadas `fetch` equivalentes que enviem `FormData` com o arquivo no campo `file`).

Incluir pesos embutidos no `Vivaz.WASM`
--------------------------------------
Se quiser que o `Vivaz.WASM` inclua pesos no binário (recursos incorporados) — de forma que a API não precise ler a pasta `PESOS` em disco — copie a pasta de pesos para `src/Vivaz.WASM/PESOS/<NOME>` antes de compilar o projeto `Vivaz.WASM`. Exemplo (PowerShell):

```powershell
# copiar pesos treinados para o projeto WASM (uma vez)
Copy-Item -Recurse PESOS\IDENTIFICADOR_LEVE src\Vivaz.WASM\PESOS\IDENTIFICADOR_LEVE
dotnet build src\Vivaz.WASM\Vivaz.WASM.csproj -c Release
```

O `Vivaz.WASM` tenta localizar recursos incorporados com nomes que contenham `CLASSIFICADOR_DETECTOR_LEVE` e `IDENTIFICADOR_LEVE`; quando detectados, ele extrai e usa esses arquivos temporariamente em runtime.

Docker (build + run)
--------------------
Este repositório inclui `Dockerfile`s e um `docker-compose.yml` para executar a API e a demonstração.

Do diretório raiz (`BIONIX-ML-VIVAZ`):

```powershell
docker compose up --build
```

Por padrão o compose sobe dois serviços úteis para desenvolvimento:

- `vivaz_api` (API) — configurável via `docker-compose.yml` (mapeamento de portas padrão no compose pode expor a API em `localhost:5001`).
- `vivaz_demo` (página de demonstração) — geralmente exposta em `localhost:5000`.

Observações:
- Para embutir pesos no binário `Vivaz.WASM` antes de construir as imagens, copie `PESOS/<FOLDER>` para `src/Vivaz.WASM/PESOS` e execute `docker compose build` para que os recursos incorporados sejam empacotados na imagem.
- O projeto `Vivaz.WASM` é uma biblioteca (.dll) usada pelo servidor; sua Dockerfile típica é de build-only para produzir o artefato que o `Vivaz.Api` consome.


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

Ajuda e contribuições
Abra issues descrevendo o cenário (incluir versão .NET, SO, passos para reproduzir). Pull requests com exemplos funcionais e instruções claras são bem-vindos.
