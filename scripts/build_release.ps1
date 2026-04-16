<#
PowerShell script to build the Vivaz solution and publish WASM/dll artifacts

Usage:
  .\build_release.ps1 -SolutionRoot "C:\PROJETOS\BIONIX\BIONIX-ML-VIVAZ" -OutDir "C:\PROJETOS\BIONIX\release"

This script will:
 - dotnet restore and dotnet build the solution
 - dotnet publish the WASM project (`Vivaz.WASM`) for `browser-wasm`
 - copy the published runtime and DLLs into a structured release folder
 - copy the `PESOS` weights directory into the release (if present)

Note: adjust project names/paths if your workspace differs.
#>

param(
    [string] $SolutionRoot = "$(Resolve-Path ..\)" ,
    [string] $OutDir = "$(Resolve-Path ..\release)"
)

Set-StrictMode -Version Latest
Write-Host "Solution root: $SolutionRoot"
Write-Host "Output dir: $OutDir"

if (-not (Test-Path $SolutionRoot)) { Write-Error "Solution root not found: $SolutionRoot"; exit 1 }

Push-Location $SolutionRoot
try {
    Write-Host "Restoring solution..."
    dotnet restore "BIONIX-ML-VIVAZ.sln" || throw "dotnet restore failed"

    Write-Host "Building solution (Release)..."
    dotnet build "BIONIX-ML-VIVAZ.sln" -c Release || throw "dotnet build failed"

    # Publish WASM project
    $wasmProj = "src/Vivaz.WASM/Vivaz.WASM.csproj"
    if (-not (Test-Path $wasmProj)) {
        # try alternative project name
        $wasmProj = "src/Vivaz.WASM/Vivaz.WASM.csproj"
    }
    if (-not (Test-Path $wasmProj)) { Write-Warning "WASM project not found at $wasmProj - skipping wasm publish" }
    else {
        $publishOut = Join-Path $OutDir "vivaz-wasm"
        Write-Host "Publishing WASM project to $publishOut..."
        dotnet publish $wasmProj -c Release -r browser-wasm -o $publishOut /p:PublishTrimmed=false /p:RunAOTCompilation=false || throw "dotnet publish failed"
    }

    # Create release layout
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

    # Copy DLLs from build output of server/demo projects (optional)
    $projectsToCopy = @(
        "src/Vivaz.Api/bin/Release/net8.0/publish",
        "src/Vivaz.Demonstracao/bin/Release/net8.0/publish"
    )
    foreach ($p in $projectsToCopy) {
        if (Test-Path $p) {
            $dest = Join-Path $OutDir (Split-Path $p -Leaf)
            Write-Host "Copying $p => $dest"
            robocopy $p $dest /MIR | Out-Null
        }
    }

    # Copy PESOS folder if exists
    $pesosSrc = Join-Path $SolutionRoot "PESOS"
    if (Test-Path $pesosSrc) {
        $pesosDest = Join-Path $OutDir "PESOS"
        Write-Host "Copying PESOS to release folder..."
        robocopy $pesosSrc $pesosDest /MIR | Out-Null
    }

    Write-Host "Release artifacts prepared in: $OutDir"
    Write-Host "Note: large binary runtime files (.wasm, native resources, weights) should be committed with Git LFS. See scripts/README-GIT-LFS.md for instructions."

} finally { Pop-Location }
