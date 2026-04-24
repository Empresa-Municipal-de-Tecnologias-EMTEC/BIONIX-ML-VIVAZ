#!/bin/bash
set -e

# Configurações
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WASM_PROJ="$PROJECT_ROOT/src/Vivaz.WASM/Vivaz.WASM.csproj"
PUBLISH_DIR="$PROJECT_ROOT/artifacts/vivaz_wasm_publish"
DEMO_WWWROOT="$PROJECT_ROOT/src/Vivaz.Demonstracao/wwwroot"
TARGET_DIR="$DEMO_WWWROOT/vivaz-wasm"
PESOS_SRC="$PROJECT_ROOT/PESOS"
PESOS_DEST="$TARGET_DIR/PESOS"

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
APPBUNDLE_DIR="$PROJECT_ROOT/src/Vivaz.WASM/bin/Release/net8.0/browser-wasm/AppBundle"

if [ -d "$APPBUNDLE_DIR" ]; then
    echo "Copiando AppBundle para $TARGET_DIR..."
    cp -r "$APPBUNDLE_DIR/"* "$TARGET_DIR/"
else
    echo "ERRO: AppBundle não encontrado em $APPBUNDLE_DIR"
    exit 1
fi

# 5. Copiar pesos para o sistema de arquivos virtual do WASM
echo "Copiando pesos para $PESOS_DEST..."
mkdir -p "$PESOS_DEST"
cp -r "$PESOS_SRC/"* "$PESOS_DEST/"

# 6. Criar arquivo de ajuda para o loader JS
cat <<EOF > "$TARGET_DIR/vivaz-loader-helper.js"
// Helper para carregar o Vivaz.WASM usando a nova API do .NET 8
import { dotnet } from './_framework/dotnet.js';

let isRuntimeReady = false;
let exports = null;

export async function initVivaz() {
    if (isRuntimeReady) return exports;
    
    const { getAssemblyExports, getConfig } = await dotnet
        .withDiagnosticTracing(false)
        .withApplicationArgumentsFromQuery()
        .create();

    const config = getConfig();
    exports = await getAssemblyExports(config.mainAssemblyName);
    isRuntimeReady = true;
    return exports;
}
EOF

echo "--- Publicação concluída com sucesso! ---"
echo "Arquivos disponíveis em: $TARGET_DIR"
