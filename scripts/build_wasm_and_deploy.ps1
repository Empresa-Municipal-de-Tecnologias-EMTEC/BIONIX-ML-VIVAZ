param(
    [string]$Configuration = "Release",
    [string]$Runtime = "browser-wasm",
    [switch]$NoSelfContained
)

$scriptRoot = $PSScriptRoot
$repoRoot = (Get-Item (Join-Path $scriptRoot "..")).FullName

$defsFile = Join-Path $repoRoot "deploy_ports.json"
if (Test-Path $defsFile) {
    $defs = Get-Content $defsFile -Raw | ConvertFrom-Json
} else {
    Write-Host "Definitions file not found at $defsFile — using defaults"
    $defs = @{ VivazApi = 5000; VivazDemonstracao = 5001; VivazWasm = 0; DemoStaticPort = 8080 }
}

$wasmProj = Join-Path $repoRoot "src\Vivaz.WASM\Vivaz.WASM.csproj"
$demoWwwRoot = Join-Path $repoRoot "src\Vivaz.Demonstracao\wwwroot"
$publishDir = Join-Path $repoRoot "artifacts\vivaz_wasm_publish"

Write-Host "Repo root: $repoRoot"
Write-Host "WASM project: $wasmProj"
Write-Host "Demo wwwroot: $demoWwwRoot"
Write-Host "Publish dir: $publishDir"

Write-Host "(debug) PSScriptRoot = $PSScriptRoot"
Write-Host "(debug) CurrentDirectory = $(Get-Location)"

if (-not (Test-Path $wasmProj)) {
    Write-Error "WASM project file not found: $wasmProj"
    exit 1
}

Write-Host "Resolved paths (pre-publish):"
Write-Host "  repoRoot = $repoRoot"
Write-Host "  wasmProj = $wasmProj"
Write-Host "  demoWwwRoot = $demoWwwRoot"
Write-Host "  publishDir = $publishDir"

if (Test-Path $publishDir) { Remove-Item $publishDir -Recurse -Force }
New-Item -ItemType Directory -Path $publishDir | Out-Null

# Build argument list for dotnet publish and run directly
$args = @('publish', $wasmProj, '-c', $Configuration, '-r', $Runtime, '-o', $publishDir)
if ($NoSelfContained) { $args += '--no-self-contained' }

Write-Host "Running: dotnet $($args -join ' ')"
$proc = Start-Process -FilePath 'dotnet' -ArgumentList $args -Wait -PassThru -NoNewWindow
if ($proc.ExitCode -ne 0) {
    Write-Error "dotnet publish failed (exit $($proc.ExitCode)). Check output above."
    exit $proc.ExitCode
}

# verify publish output
if (-not (Test-Path $publishDir)) {
    Write-Error "Publish directory not found: $publishDir"
    exit 1
}
$publishedFiles = Get-ChildItem -Path $publishDir -Recurse -File -ErrorAction SilentlyContinue
if ($publishedFiles.Count -eq 0) {
    Write-Error "No files were published to $publishDir"
    exit 1
}
Write-Host "Published files count: $($publishedFiles.Count)"

# Copy published files into demo wwwroot/vivaz-wasm
$targetDir = Join-Path $demoWwwRoot "vivaz-wasm"
if (Test-Path $targetDir) { Remove-Item $targetDir -Recurse -Force }
New-Item -ItemType Directory -Path $targetDir | Out-Null

Write-Host "Copying published WASM files to $targetDir"
# Copy all published files preserving relative layout
Get-ChildItem -Path $publishDir -Recurse | ForEach-Object {
    $rel = $_.FullName.Substring($publishDir.Length).TrimStart('\','/')
    $dest = Join-Path $targetDir $rel
    if ($_.PSIsContainer) { New-Item -ItemType Directory -Path $dest -Force | Out-Null }
    else { Copy-Item -Path $_.FullName -Destination $dest -Force }
}

# Also copy PESOS folder from repo root if present
$pesosSrc = Join-Path $repoRoot "PESOS"
if (Test-Path $pesosSrc) {
    $pesosDest = Join-Path $demoWwwRoot "PESOS"
    if (Test-Path $pesosDest) { Remove-Item $pesosDest -Recurse -Force }
    Write-Host "Copying PESOS from $pesosSrc to $pesosDest"
    Copy-Item -Path $pesosSrc\* -Destination $pesosDest -Recurse -Force
} else {
    Write-Host "No PESOS directory found at $pesosSrc — skipping copy"
}


Write-Host "WASM publish and deploy complete. Files copied to:"
Get-ChildItem -Path $targetDir -Recurse | Select-Object FullName, Length -First 50

Write-Host "Demo wwwroot PESOS summary (if present):"
if (Test-Path (Join-Path $demoWwwRoot 'PESOS')) { Get-ChildItem -Path (Join-Path $demoWwwRoot 'PESOS') -Recurse | Select-Object FullName, Length -First 50 }

Write-Host "Done. You can now serve the demo (e.g., run the demo project) and pages will load the WASM from /vivaz-wasm/"
