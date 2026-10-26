# Convenience wrapper: sources the OpenVINO environment (so the runtime DLLs and the
# Arc GPU plugin are on PATH) and runs the built music-remover binary, forwarding all args.
#
#   .\music-remover.ps1 https://youtu.be/<id>
#   .\music-remover.ps1 .\clip.mp4 --keep speech --mode audio --audio-format mp3
#
# Override the OpenVINO location with $env:OPENVINO_DIST if you installed it elsewhere.

$ErrorActionPreference = "Stop"
$ovDist = if ($env:OPENVINO_DIST) { $env:OPENVINO_DIST } else {
    "$env:USERPROFILE\openvino-dist\openvino_toolkit_windows_2025.4.0.20398.8fdad55727d_x86_64"
}
$setup = Join-Path $ovDist "setupvars.ps1"
if (-not (Test-Path $setup)) { throw "OpenVINO not found at $ovDist (set `$env:OPENVINO_DIST)" }
. $setup | Out-Null

$exe = Join-Path $PSScriptRoot "target\release\music-remover.exe"
if (-not (Test-Path $exe)) { throw "build first: cargo build --release   (missing $exe)" }
& $exe @args
exit $LASTEXITCODE
