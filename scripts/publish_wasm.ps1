# PowerShell conversion of publish_wasm.sh
# Usage: run from repository root or call directly with PowerShell/PowerShell Core

$ErrorActionPreference = 'Stop'

# Paths
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$WASM_PROJ = Join-Path $PROJECT_ROOT 'src\Vivaz.WASM\Vivaz.WASM.csproj'
$PUBLISH_DIR = Join-Path $PROJECT_ROOT 'artifacts\vivaz_wasm_publish'
$DEMO_WWWROOT = Join-Path $PROJECT_ROOT 'src\Vivaz.Demonstracao\wwwroot'
$TARGET_DIR = Join-Path $DEMO_WWWROOT 'vivaz-wasm'
$PESOS_SRC = Join-Path $PROJECT_ROOT 'PESOS'
$PESOS_DEST = Join-Path $TARGET_DIR 'PESOS'
$PESOS_WWWROOT_DEST = Join-Path $DEMO_WWWROOT 'PESOS'

Write-Host "--- Iniciando publicação do Vivaz.WASM ---"
Write-Host "Projeto: $WASM_PROJ"
Write-Host "Destino: $TARGET_DIR"

# 1. Garantir que o workload wasm-tools está instalado
try {
    $workloads = dotnet workload list 2>&1
} catch {
    Write-Warning "Não foi possível executar 'dotnet workload list': $_"
    $workloads = ''
}

if ($workloads -notmatch 'wasm-tools') {
    Write-Host "Instalando workload wasm-tools..."
    try {
        dotnet workload install wasm-tools
    } catch {
        Write-Warning "Falha ao instalar wasm-tools: $_"
        Write-Host "Se preciso, execute manualmente: dotnet workload install wasm-tools (como Administrador)."
    }
}

# 2. Limpar diretórios antigos
if (Test-Path $PUBLISH_DIR) { Remove-Item $PUBLISH_DIR -Recurse -Force }
if (Test-Path $TARGET_DIR) { Remove-Item $TARGET_DIR -Recurse -Force }

New-Item -ItemType Directory -Path $TARGET_DIR -Force | Out-Null

# 3. Publicar para browser-wasm
Write-Host "Executando dotnet publish..."
dotnet publish $WASM_PROJ -c Release -r browser-wasm -o $PUBLISH_DIR

# 4. Copiar arquivos do AppBundle para o destino da demonstração
$APPBUNDLE_DIR = Join-Path $PROJECT_ROOT 'src\Vivaz.WASM\bin\Release\net8.0\browser-wasm\AppBundle'

if (Test-Path $APPBUNDLE_DIR) {
    Write-Host "Copiando AppBundle para $TARGET_DIR..."
    Copy-Item -Path (Join-Path $APPBUNDLE_DIR '*') -Destination $TARGET_DIR -Recurse -Force
} else {
    Write-Error "ERRO: AppBundle não encontrado em $APPBUNDLE_DIR"
    exit 1
}

# 4b. Garantir que exista um blazor.boot.json mínimo no root do vivaz-wasm
$bootJsonPath = Join-Path $TARGET_DIR 'blazor.boot.json'
if (-not (Test-Path $bootJsonPath)) {
        Write-Host "Gerando blazor.boot.json mínimo em $bootJsonPath..."

        # Coletar arquivos de PESOS e montar entradas JSON relativas (servidas em /vivaz-wasm/PESOS/...)
        $pesoFiles = @()
        if (Test-Path $PESOS_SRC) {
                Get-ChildItem -Path $PESOS_SRC -Recurse -File | ForEach-Object {
                    $rel = $_.FullName.Substring($PESOS_SRC.Length+1) -replace '\\','/'
                    $pesoFiles += ('"PESOS/{0}": ""' -f $rel)
                }
        }

        $assetsBlock = ""
        if ($pesoFiles.Count -gt 0) {
            $assetsBlock = '        ,"assets": {' + "`n" + ($pesoFiles -join ",`n") + "`n        }"
        }

        $bootJson = @"
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
    }
$assetsBlock
}
"@

        $bootJson | Out-File -FilePath $bootJsonPath -Encoding UTF8
}

# 5. Copiar pesos para o sistema de arquivos virtual do WASM
Write-Host "Copiando pesos para $PESOS_DEST..."
New-Item -ItemType Directory -Path $PESOS_DEST -Force | Out-Null
Copy-Item -Path (Join-Path $PESOS_SRC '*') -Destination $PESOS_DEST -Recurse -Force

Write-Host "Copiando pesos também para $PESOS_WWWROOT_DEST (servir em /PESOS)..."
New-Item -ItemType Directory -Path $PESOS_WWWROOT_DEST -Force | Out-Null
Copy-Item -Path (Join-Path $PESOS_SRC '*') -Destination $PESOS_WWWROOT_DEST -Recurse -Force

# 6. Criar arquivo de ajuda para o loader JS
$helperPath = Join-Path $TARGET_DIR 'vivaz-loader-helper.js'
# Do not overwrite an existing custom helper; only create a default if missing
if (-not (Test-Path $helperPath)) {
    $helper = @'
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
'@

    $helper | Out-File -FilePath $helperPath -Encoding UTF8
} else {
    Write-Host "Skipping helper overwrite; file already exists: $helperPath"
}

Write-Host "--- Publicação concluída com sucesso! ---"
Write-Host "Arquivos disponíveis em: $TARGET_DIR"
