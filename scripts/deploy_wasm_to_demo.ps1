<#
Deploy artifacts and PESOS to Vivaz.Demonstracao wwwroot

Usage: run from repository root or call directly with PowerShell
  .\scripts\deploy_wasm_to_demo.ps1 [-ArtifactsDir <path>] [-DemoWwwRoot <path>] [-ForceClean]

This script copies the published AppBundle/artifacts into
`src/Vivaz.Demonstracao/wwwroot/vivaz-wasm` and copies the `PESOS` folder
into `src/Vivaz.Demonstracao/wwwroot/PESOS` so the demo serves runtime files
and model weights at the expected paths.
#>

param(
    [string]$ArtifactsDir = "artifacts\vivaz_wasm_publish",
    [string]$DemoWwwRoot = "src\Vivaz.Demonstracao\wwwroot",
    [switch]$ForceClean
)

$ErrorActionPreference = 'Stop'

$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$ARTIFACTS_FULL = Join-Path $PROJECT_ROOT $ArtifactsDir
$TARGET_VIVAZ = Join-Path $DemoWwwRoot 'vivaz-wasm'
$TARGET_PESOS = Join-Path $DemoWwwRoot 'PESOS'
$PESOS_SRC = Join-Path $PROJECT_ROOT 'PESOS'

Write-Host "Deploying WASM artifacts and PESOS for demo"
Write-Host "Artifacts source: $ARTIFACTS_FULL"
Write-Host "Demo vivaz-wasm target: $TARGET_VIVAZ"
Write-Host "Demo PESOS target: $TARGET_PESOS"

if (-not (Test-Path $ARTIFACTS_FULL)) {
    $alt = Join-Path $PROJECT_ROOT 'artifacts\vivaz_wasm_publish_simple'
    if (Test-Path $alt) { Write-Host "Artifacts dir not found; using $alt"; $ARTIFACTS_FULL = $alt }
}

if (-not (Test-Path $ARTIFACTS_FULL)) {
    Write-Error "Artifacts directory not found: $ArtifactsDir or vivaz_wasm_publish_simple"
    exit 1
}

if ($ForceClean -and (Test-Path $TARGET_VIVAZ)) { Remove-Item -Recurse -Force $TARGET_VIVAZ }
if ($ForceClean -and (Test-Path $TARGET_PESOS)) { Remove-Item -Recurse -Force $TARGET_PESOS }

New-Item -ItemType Directory -Path $TARGET_VIVAZ -Force | Out-Null
Write-Host "Copying artifacts to $TARGET_VIVAZ..."
Copy-Item -Path (Join-Path $ARTIFACTS_FULL '*') -Destination $TARGET_VIVAZ -Recurse -Force

Write-Host "Ensuring PESOS are copied to demo webroot..."
if (-not (Test-Path $PESOS_SRC)) { Write-Warning "Source PESOS folder not found at $PESOS_SRC" } else {
    New-Item -ItemType Directory -Path $TARGET_PESOS -Force | Out-Null
    Copy-Item -Path (Join-Path $PESOS_SRC '*') -Destination $TARGET_PESOS -Recurse -Force
}

Write-Host "Deploy complete. Files placed in:"
Write-Host "  $TARGET_VIVAZ"
Write-Host "  $TARGET_PESOS"

Write-Host "Tip: run the demo and verify /vivaz-wasm/dotnet.js and /pesos/<MODEL>/meta.json are accessible."
