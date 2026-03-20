param(
    [string]$PythonExe = ".\venv\Scripts\python.exe",
    [switch]$IncludeCachedModels,
    [switch]$SkipFFmpegBundle
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)

    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($Args -join ' ')"
    }
}

function Write-Section {
    param([string]$Title)

    Write-Host ""
    Write-Host "== $Title ==" -ForegroundColor Cyan
}

function Copy-TorchRuntimeDependencies {
    param([string]$BundleRoot)

    $runtimeRoot = Join-Path $BundleRoot '_internal'
    $torchLibDir = Join-Path $runtimeRoot 'torch\lib'
    if (-not (Test-Path $runtimeRoot) -or -not (Test-Path $torchLibDir)) {
        return
    }

    $sourceDirs = @(
        $runtimeRoot,
        (Join-Path $runtimeRoot 'PyQt6\Qt6\bin')
    )
    $patterns = @(
        'concrt140.dll',
        'msvcp140*.dll',
        'vccorlib140.dll',
        'vcruntime140*.dll'
    )

    $copiedNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($sourceDir in $sourceDirs) {
        if (-not (Test-Path $sourceDir)) {
            continue
        }

        foreach ($pattern in $patterns) {
            Get-ChildItem -Path $sourceDir -Filter $pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
                if ($copiedNames.Add($_.Name)) {
                    Copy-Item -Force $_.FullName (Join-Path $torchLibDir $_.Name)
                }
            }
        }
    }

    if ($copiedNames.Count -gt 0) {
        Write-Host "Copied $($copiedNames.Count) VC++ runtime DLLs into torch\\lib for portable startup." -ForegroundColor Green
    }
}

Write-Section "Ensuring PyInstaller is installed"
& $PythonExe -m pip show pyinstaller >$null 2>&1
if ($LASTEXITCODE -ne 0) {
    Invoke-Python -m pip install -r requirements-packaging.txt
}

Write-Section "Cleaning old build outputs"
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force dist\YouTubeTranscriber -ErrorAction SilentlyContinue

Write-Section "Building standalone bundle"
Invoke-Python -m PyInstaller --noconfirm --clean youtube_transcriber.spec

$bundleRoot = Join-Path $projectRoot 'dist\YouTubeTranscriber'
if (-not (Test-Path $bundleRoot)) {
    throw "Bundle not found: $bundleRoot"
}

Write-Section "Patching torch runtime DLL layout"
Copy-TorchRuntimeDependencies -BundleRoot $bundleRoot

if (-not $SkipFFmpegBundle) {
    Write-Section "Bundling FFmpeg if available"
    $ffmpegExe = $null
    $ffmpegCommand = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegCommand) {
        $ffmpegExe = $ffmpegCommand.Source
    }

    if (-not $ffmpegExe) {
        $ffmpegDir = (& $PythonExe -c "from youtube_transcriber import find_ffmpeg; print(find_ffmpeg() or '')").Trim()
        if ($LASTEXITCODE -eq 0 -and $ffmpegDir) {
            $ffmpegExe = Join-Path $ffmpegDir 'ffmpeg.exe'
        }
    }

    if ($ffmpegExe -and (Test-Path $ffmpegExe)) {
        $ffmpegDir = Split-Path -Parent $ffmpegExe
        Copy-Item -Force (Join-Path $ffmpegDir 'ffmpeg.exe') (Join-Path $bundleRoot 'ffmpeg.exe')
        if (Test-Path (Join-Path $ffmpegDir 'ffprobe.exe')) {
            Copy-Item -Force (Join-Path $ffmpegDir 'ffprobe.exe') (Join-Path $bundleRoot 'ffprobe.exe')
        }
        Write-Host "Bundled FFmpeg from $ffmpegDir" -ForegroundColor Green
    } else {
        Write-Warning "FFmpeg was not found on the build machine. The portable app will still require FFmpeg unless the user installs it."
    }
}

if ($IncludeCachedModels) {
    Write-Section "Copying cached model repositories"
    $manifest = & $PythonExe build_cache_manifest.py
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect cached model repositories."
    }

    if ($manifest) {
        $repoRoots = $manifest | ConvertFrom-Json
    } else {
        $repoRoots = @()
    }

    if ($repoRoots -is [string]) {
        $repoRoots = @($repoRoots)
    }
    if ($null -eq $repoRoots) {
        $repoRoots = @()
    }

    if ($repoRoots.Count -gt 0) {
        $hubRoot = Join-Path $bundleRoot 'hf-cache\hub'
        New-Item -ItemType Directory -Force -Path $hubRoot | Out-Null
        foreach ($repoRoot in $repoRoots) {
            $repoName = Split-Path -Leaf $repoRoot
            Copy-Item -Recurse -Force $repoRoot (Join-Path $hubRoot $repoName)
        }
        Write-Host "Copied $($repoRoots.Count) cached model repositories into hf-cache." -ForegroundColor Green
    } else {
        Write-Warning "No cached models were available to copy. The portable app will download them on first use."
    }
}

$readme = @'
YouTube Transcriber Portable
============================

Launch:
- Double-click YouTubeTranscriber.exe

Notes:
- This is a standalone Windows bundle built from this project.
- If ffmpeg.exe and ffprobe.exe are present in this folder, the app will use them automatically.
- If hf-cache is present, the app will use bundled model caches first.
- If models are not bundled, the first transcription or grammar use may still download them.

Outputs:
- config.json, logs, and temp runtime files are written to this folder when writable.
- If this folder is not writable, the app falls back to LocalAppData\YouTubeTranscriber.
'@
Set-Content -Path (Join-Path $bundleRoot 'README-portable.txt') -Value $readme

Write-Section "Done"
Write-Host "Portable bundle ready at: $bundleRoot" -ForegroundColor Green
Write-Host "Zip the entire folder to hand it off." -ForegroundColor Green
