#!/bin/bash
# publish_wasm.sh - Publicação automatizada do Vivaz.WASM para a demonstração
set -e

# Configurações de Caminhos
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WASM_PROJ="$PROJECT_ROOT/src/Vivaz.WASM/Vivaz.WASM.csproj"
PUBLISH_DIR="$PROJECT_ROOT/artifacts/vivaz_wasm_publish"
DEMO_WWWROOT="$PROJECT_ROOT/src/Vivaz.Demonstracao/wwwroot"
TARGET_DIR="$DEMO_WWWROOT/vivaz-wasm"
PESOS_SRC="$PROJECT_ROOT/PESOS"
PESOS_DEST="$TARGET_DIR/PESOS"
PESOS_WWWROOT_DEST="$DEMO_WWWROOT/PESOS"

echo "--- Iniciando publicação do Vivaz.WASM ---"
echo "Projeto: $WASM_PROJ"
echo "Destino: $TARGET_DIR"

# 1. Garantir que o workload wasm-tools está instalado
if ! dotnet workload list | grep -q "wasm-tools"; then
    echo "Instalando workload wasm-tools..."
    sudo dotnet workload install wasm-tools
fi

# 2. Limpar diretórios antigos
rm -rf "$PUBLISH_DIR"
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

# 3. Publicar para browser-wasm
echo "Executando dotnet publish..."
dotnet publish "$WASM_PROJ" -c Release -r browser-wasm -o "$PUBLISH_DIR"

# 4. Copiar arquivos do AppBundle para o destino da demonstração
# O dotnet publish para browser-wasm gera o AppBundle no diretório de saída
APPBUNDLE_DIR="$PROJECT_ROOT/src/Vivaz.WASM/bin/Release/net8.0/browser-wasm/AppBundle"

if [ -d "$APPBUNDLE_DIR" ]; then
    echo "Copiando AppBundle para $TARGET_DIR..."
    cp -r "$APPBUNDLE_DIR/"* "$TARGET_DIR/"
else
    echo "ERRO: AppBundle não encontrado em $APPBUNDLE_DIR"
    exit 1
fi

# 5. Copiar pesos para o sistema de arquivos virtual do WASM e para servir via HTTP
echo "Copiando pesos para $PESOS_DEST..."
mkdir -p "$PESOS_DEST"
if [ -d "$PESOS_SRC" ]; then
    cp -r "$PESOS_SRC/"* "$PESOS_DEST/"
    
    echo "Copiando pesos também para $PESOS_WWWROOT_DEST (servir em /PESOS)..."
    mkdir -p "$PESOS_WWWROOT_DEST"
    cp -r "$PESOS_SRC/"* "$PESOS_WWWROOT_DEST/"
else
    echo "AVISO: Pasta PESOS não encontrada em $PESOS_SRC. Pulando cópia de pesos."
fi

# 5b. Garantir que exista um blazor.boot.json mínimo contendo os arquivos em PESOS
BOOT_JSON="$TARGET_DIR/blazor.boot.json"
if [ ! -f "$BOOT_JSON" ]; then
    echo "Gerando blazor.boot.json mínimo com assets de PESOS em $BOOT_JSON..."
    # montar lista de arquivos em PESOS relativa a vivaz-wasm
    ASSETS_JSON=""
    if [ -d "$PESOS_SRC" ]; then
        while IFS= read -r -d '' f; do
            rel=${f#"$PESOS_SRC/"}
            rel=${rel//\//\/}
            ASSETS_JSON+="        \"PESOS/$rel\": \"\",\n"
        done < <(find "$PESOS_SRC" -type f -print0)
        # remover vírgula final
        ASSETS_JSON="${ASSETS_JSON%,\n}"
    fi

    cat > "$BOOT_JSON" <<EOF
{
    "mainAssemblyName": "Vivaz.WASM.dll",
    "resources": {
        "assembly": {
            "Vivaz.WASM.dll": "",
            "Bionix.ML.dll": "",
            "DetectorLeveBModel.dll": "",
            "DetectorModel.dll": "",
            "IdentificadorLeveModel.dll": "",
            "ILGPU.dll": "",
            "SixLabors.ImageSharp.dll": ""
        },
        "runtime": {
            "icudt.dat": ""
        },
        "wasmNative": {
            "dotnet.native.wasm": ""
        },
        "jsModuleNative": {
            "dotnet.native.js": ""
        },
        "jsModuleRuntime": {
            "dotnet.runtime.js": ""
        }
    },
    "assets": {
$ASSETS_JSON
    }
}
EOF
fi

# 6. Garantir que vivaz.js e vivaz-wasm-loader.js existam (os arquivos fonte já estão no repo)
echo "Verificando scripts de carregamento no wwwroot..."
if [ ! -f "$DEMO_WWWROOT/vivaz.js" ]; then
    echo "AVISO: vivaz.js não encontrado em $DEMO_WWWROOT"
fi

echo "--- Publicação concluída com sucesso! ---"
echo "Arquivos disponíveis em: $TARGET_DIR"
echo "Acesse a demo em: /compare_wasm.html"
