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

Modelos e dataset

Coloque checkpoints em `src/MODELO/` e imagens em `src/DATASET/` (ambas pastas não versionadas). Este repositório inclui suporte para o dataset CelebA com landmarks. Estrutura esperada:

- `DATASET/faces_com_landmarks/list_landmarks_align_celeba.csv` — anotações de landmarks (CSV)
- `DATASET/faces_com_landmarks/list_bbox_celeba.csv` — bounding boxes (CSV, opcional)
- `DATASET/faces_com_landmarks/img_align_celeba/img_align_celeba/` — imagens JPG

O `DataLoader` em `src/DetectorModel/dados/DataLoader.cs` detecta automaticamente o formato CelebA (CSV) e carrega landmarks e bounding boxes quando presentes. Para usar outro dataset (por exemplo WIDER), mantenha o formato original de anotações e o loader fará fallback.

Para produção, exporte os pesos para um formato portátil (por exemplo ONNX) se for necessário interoperar com outros runtimes.

Conversão de imagens

Existe um utilitário Python para conversão/normalização de imagens (se aplicável). Instale `Pillow` se for usar scripts de conversão.

Observações

Os projetos expõem pontos de partida (stubs) para detecção e reconhecimento. É necessário integrar o carregamento de checkpoints e a lógica de inferência para tornar os endpoints produtivos. Para cenários cross-platform considere migrar utilitários de imagem para ImageSharp/SkiaSharp.

Ajuda e contribuições
Abra issues descrevendo o cenário (incluir versão .NET, SO, passos para reproduzir). Pull requests com exemplos funcionais e instruções claras são bem-vindos.
