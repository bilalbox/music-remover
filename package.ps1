# Build a self-contained, relocatable package of music-remover.
#
# Produces dist/music-remover/ containing the exe, the OpenVINO runtime DLLs (incl. the
# Arc GPU plugin), and the model. Drop that whole folder anywhere on the PC and run
# music-remover.exe directly — no setupvars, no PATH changes. (Still needs ffmpeg + yt-dlp
# on PATH, and the Intel GPU driver.)
#
# Usage:  .\package.ps1   [-OvDist <openvino install dir>]

param(
    [string]$OvDist = "$env:USERPROFILE\openvino-dist\openvino_toolkit_windows_2025.4.0.20398.8fdad55727d_x86_64",
    # Also build a single self-extracting exe (dist-single\music-remover.exe).
    [switch]$SingleFile
)
$ErrorActionPreference = "Stop"
$proj = $PSScriptRoot

# Antivirus can briefly lock just-written DLLs; retry the delete a few times.
function Remove-Tree($path) {
    if (-not (Test-Path $path)) { return }
    for ($i = 0; $i -lt 10; $i++) {
        try { Remove-Item $path -Recurse -Force; return }
        catch { Start-Sleep -Milliseconds 400 }
    }
    Remove-Item $path -Recurse -Force  # final try; let it throw if still locked
}
$rel = Join-Path $OvDist "runtime\bin\intel64\Release"
$tbb = Join-Path $OvDist "runtime\3rdparty\tbb\bin"
if (-not (Test-Path $rel)) { throw "OpenVINO runtime not found at $rel (use -OvDist)" }

# 1. Build release (sourcing OpenVINO env so the link step finds openvino_c).
. (Join-Path $OvDist "setupvars.ps1") | Out-Null
$cargo = (Get-Command cargo -ErrorAction SilentlyContinue).Source
if (-not $cargo) { $cargo = "$env:USERPROFILE\.cargo\bin\cargo.exe" }
Write-Host "building release..."
& $cargo build --release --manifest-path (Join-Path $proj "Cargo.toml")
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

# 2. Assemble the package folder.
$dist = Join-Path $proj "dist\music-remover"
Remove-Tree $dist
New-Item -ItemType Directory -Force $dist | Out-Null

Copy-Item (Join-Path $proj "target\release\music-remover.exe") $dist

# Release OpenVINO DLLs (the runtime, IR frontend, and all device plugins) + plugins.xml.
$dlls = @(
    "openvino.dll", "openvino_c.dll", "openvino_ir_frontend.dll",
    "openvino_intel_gpu_plugin.dll", "openvino_intel_cpu_plugin.dll",
    "openvino_intel_npu_plugin.dll", "openvino_auto_plugin.dll",
    "openvino_auto_batch_plugin.dll", "openvino_hetero_plugin.dll"
)
foreach ($d in $dlls) {
    $src = Join-Path $rel $d
    if (Test-Path $src) { Copy-Item $src $dist } else { Write-Warning "missing $d" }
}
# OpenVINO 2025.x has no plugins.xml — device->plugin mapping is built in; the plugin
# DLLs just need to sit next to openvino.dll (they do, in $dist).
foreach ($t in @("tbb12.dll", "tbbmalloc.dll")) {
    $src = Join-Path $tbb $t
    if (Test-Path $src) { Copy-Item $src $dist }
}

# Model (IR only — the checkpoint and any ONNX leftover stay out of the package).
New-Item -ItemType Directory -Force (Join-Path $dist "models\htdemucs_dnr") | Out-Null
Copy-Item (Join-Path $proj "models\htdemucs_dnr\htdemucs_fwd.xml") (Join-Path $dist "models\htdemucs_dnr")
Copy-Item (Join-Path $proj "models\htdemucs_dnr\htdemucs_fwd.bin") (Join-Path $dist "models\htdemucs_dnr")

# Short usage note.
@"
music-remover — remove background music, keeping speech and sound effects (HTDemucs-DnR).

Run from anywhere:
  music-remover.exe "<file-or-URL>"
  music-remover.exe clip.mp4 --keep speech
  music-remover.exe clip.mp4 --keep music
  music-remover.exe https://youtu.be/<id> --mode audio --audio-format mp3

Requires ffmpeg and yt-dlp on PATH, and an Intel GPU driver.
The compiled-model cache is created here as ov_cache\ on first run (fast subsequent launches).
"@ | Set-Content (Join-Path $dist "README.txt")

$size = (Get-ChildItem $dist -Recurse | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("`npackaged -> {0}  ({1:N0} MB)" -f $dist, $size)
Get-ChildItem $dist | Select-Object Name, Length | Format-Table -AutoSize

# 3. Optional single self-extracting exe.
if ($SingleFile) {
    $launcher = Join-Path $proj "launcher"
    $payload = Join-Path $launcher "payload"
    Remove-Tree $payload
    New-Item -ItemType Directory -Force $payload | Out-Null
    Copy-Item (Join-Path $dist "*") $payload -Recurse -Force

    Write-Host "`nbuilding single-file launcher (embeds the whole runtime; this is heavy)..."
    & $cargo build --release --manifest-path (Join-Path $launcher "Cargo.toml")
    if ($LASTEXITCODE -ne 0) { throw "launcher build failed" }

    $single = Join-Path $proj "dist-single"
    New-Item -ItemType Directory -Force $single | Out-Null
    Copy-Item (Join-Path $launcher "target\release\music-remover.exe") $single -Force
    $sz = (Get-Item (Join-Path $single "music-remover.exe")).Length / 1MB
    Write-Host ("single-file -> {0}\music-remover.exe  ({1:N0} MB)" -f $single, $sz)
}
