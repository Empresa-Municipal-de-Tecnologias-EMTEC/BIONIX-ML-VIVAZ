param(
    [string]$Configuration = "Release",
    [string]$Runtime = "browser-wasm"
)

try {
    $scriptRoot = $PSScriptRoot
    $repoRoot = (Get-Item (Join-Path $scriptRoot "..")).FullName
    $wasmProj = Join-Path $repoRoot "src\Vivaz.WASM\Vivaz.WASM.csproj"
    $demoWwwRoot = Join-Path $repoRoot "src\Vivaz.Demonstracao\wwwroot"
    $publishDir = Join-Path $repoRoot "artifacts\vivaz_wasm_publish_simple"

    Write-Host "repoRoot: $repoRoot"
    Write-Host "wasmProj: $wasmProj"
    Write-Host "demoWwwRoot: $demoWwwRoot"
    Write-Host "publishDir: $publishDir"

    if (-not (Test-Path $wasmProj)) { throw "WASM project file not found: $wasmProj" }

    if (Test-Path $publishDir) { Remove-Item $publishDir -Recurse -Force }
    New-Item -ItemType Directory -Path $publishDir | Out-Null

    $args = @('publish', $wasmProj, '-c', $Configuration, '-r', $Runtime, '-o', $publishDir)
    Write-Host "Running: dotnet $($args -join ' ')"
    $proc = Start-Process -FilePath 'dotnet' -ArgumentList $args -Wait -PassThru -NoNewWindow
    if ($proc.ExitCode -ne 0) { throw "dotnet publish failed with exit code $($proc.ExitCode)" }

    $files = Get-ChildItem -Path $publishDir -Recurse -File
    if ($files.Count -eq 0) { throw "No files published to $publishDir" }
    Write-Host "Published files: $($files.Count)"

    $targetDir = Join-Path $demoWwwRoot "vivaz-wasm"
    if (Test-Path $targetDir) { Remove-Item $targetDir -Recurse -Force }
    New-Item -ItemType Directory -Path $targetDir | Out-Null

    foreach ($f in $files) {
        $rel = $f.FullName.Substring($publishDir.Length).TrimStart('\','/')
        $dest = Join-Path $targetDir $rel
        $destDir = Split-Path $dest -Parent
        if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir | Out-Null }
        Copy-Item -Path $f.FullName -Destination $dest -Force
    }

    # copy PESOS folder
    $pesosSrc = Join-Path $repoRoot 'PESOS'
    if (Test-Path $pesosSrc) {
        $pesosDest = Join-Path $demoWwwRoot 'PESOS'
        if (Test-Path $pesosDest) { Remove-Item $pesosDest -Recurse -Force }
        Copy-Item -Path (Join-Path $pesosSrc '*') -Destination $pesosDest -Recurse -Force
        Write-Host "Copied PESOS to $pesosDest"
    } else { Write-Host "No PESOS found at $pesosSrc" }

    Write-Host "Deploy finished. vivaz-wasm files in: $targetDir"
    Get-ChildItem -Path $targetDir -Recurse | Select-Object FullName, Length -First 40
} catch {
    Write-Error $_.Exception.Message
    exit 1
}
