# BIONIX-ML-VIVAZ (.NET)

Conjunto de projetos .NET que consomem `Bionix.ML` para prover detecção e reconhecimento facial, além de ferramentas de anotação de dataset e uma API HTTP mínima.

Resumo dos projetos
- `src/DetectorModel/` — biblioteca com adaptadores/serviços de detecção (RetinaFace stubs).
- `src/IdentificadorModel/` — biblioteca para cálculo de embeddings e identificação (ArcFace stubs).
- `src/Vivaz.Api/` — ASP.NET Core Web API com endpoints de detecção/identificação (`FaceController`).
- `src/AnotadorDeDataSet/` — WinForms (DatasetAnnotator) portado para .NET (target `net8.0-windows`).

Como compilar toda a solução

dotnet restore
dotnet build -c Release

Executar a API (desenvolvimento)

dotnet run --project src/Vivaz.Api -c Release

Endpoints principais (resumo)
- `GET /api/face/health` — health check
- `POST /api/face/detect` — recebe BMP e retorna bounding boxes
- `POST /api/face/compare` — recebe duas imagens e retorna score/verificação

Executar o anotador de dataset (WinForms)

dotnet run --project src/AnotadorDeDataSet -f net8.0-windows

Modelos e dataset
- Coloque checkpoints e artefatos de modelos em `src/MODELO/` e imagens convertidas em `src/DATASET/` (pastas não versionadas).
- Os formatos de pesos dependem do pipeline original (ver `BIONIX-ML`); para uso em produção considere exportar para ONNX ou outro formato padrão.

Observações e recomendações
- A API e os modelos neste repositório são stubs de integração: os endpoints foram adicionados como pontos de partida — implemente o carregamento de pesos e a lógica de inferência conforme seu formato de modelo.
- Para imagens, usamos utilitários base em `System.Drawing`. Em cenários cross-platform considere ImageSharp/SkiaSharp.
- Quando os repositórios forem separados em pacotes distintos, substitua `ProjectReference` por pacotes NuGet, submódulos, ou uma estratégia de CI que publique artefatos.

Ajuda e contribuições
Abra issues descrevendo o cenário (incluir versão .NET, SO, passos para reproduzir). Pull requests com exemplos funcionais e instruções claras são bem-vindos.
