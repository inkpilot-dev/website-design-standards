# Inkport all-in-one installer build for Windows x64 (NSIS .exe + .msi).
#
# The installers contain a relocatable CPython with the engine and its libraries pre-installed,
# llama.cpp with CPU (all x86 variants) + Vulkan (+ CUDA in the cuda flavor) backends, the MSVC
# runtime DLLs (app-local), and a 4-bit starter model. Users never install Python, CUDA, Visual
# Studio, the VC++ redistributable or anything else.
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_all_in_one.ps1
#   ... -Flavor cuda        # NVIDIA training build (needs the CUDA toolkit on THIS build machine)
#   ... -SmokeModel         # bundle the 1.4M-parameter test model (CI)
#   ... -DebugBuild         # faster debug Rust build
#   ... -SkipPayload        # reuse build\payload-windows-x86_64
#
# Build-machine requirements (never needed by users): uv, Rust (rustup, MSVC toolchain), Node 22 +
# pnpm, CMake, Visual Studio 2022 Build Tools (C++ workload), the Vulkan SDK (VULKAN_SDK set).
# Optional signing: WINDOWS_CERTIFICATE_THUMBPRINT (a code-signing cert in the current user's store).
param(
  [ValidateSet("cpu", "cuda")][string]$Flavor = "cpu",
  [switch]$SmokeModel,
  [switch]$DebugBuild,
  [switch]$SkipPayload,
  [switch]$NoVulkan,                       # CPU-only llama.cpp (when the Vulkan SDK is not installed)
  [string]$Bundles = "nsis,msi"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Platform = "windows-x86_64"
$Triple = "x86_64-pc-windows-msvc"
$Payload = Join-Path $Root "build\payload-$Platform"
$Out = Join-Path $Root "build\installers\$Platform-$Flavor"
$Version = (Get-Content "src-tauri\tauri.conf.json" -Raw | ConvertFrom-Json).version

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Need($cmd, $hint) { if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { throw "missing build tool: $cmd ($hint)" } }
function Py { & uv run --no-project --python 3.12 python @args; if ($LASTEXITCODE -ne 0) { throw "python step failed: $args" } }
function Run { param([string]$Exe, [Parameter(ValueFromRemainingArguments = $true)] $Rest) & $Exe @Rest; if ($LASTEXITCODE -ne 0) { throw "$Exe failed ($LASTEXITCODE)" } }

if ([Environment]::Is64BitOperatingSystem -eq $false) { throw "Build on 64-bit Windows." }
Step "Inkport $Version all-in-one build: $Platform ($Flavor)"
Need uv "https://docs.astral.sh/uv/"
Need cargo "https://rustup.rs"
Need pnpm "npm install -g pnpm"
Need cmake "https://cmake.org"
if (-not $NoVulkan -and -not $env:VULKAN_SDK) { throw "Install the Vulkan SDK (https://vulkan.lunarg.com) so llama.cpp can be built with GPU support for every vendor, or pass -NoVulkan." }

# Enter the Visual Studio developer environment (cl.exe, VCToolsRedistDir) if not already in it.
if (-not $env:VCToolsRedistDir) {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  if (-not (Test-Path $vswhere)) { throw "Visual Studio 2022 Build Tools with the C++ workload are required on the build machine." }
  $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  Import-Module (Join-Path $vs "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
  Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64" | Out-Null
}
$crtDir = Get-ChildItem (Join-Path $env:VCToolsRedistDir "x64") -Directory | Where-Object { $_.Name -like "Microsoft.VC14*.CRT" } | Select-Object -First 1
if (-not $crtDir) { throw "MSVC CRT redistributable folder not found under $env:VCToolsRedistDir" }

Step "1/7 Frontend"
Run pnpm install --frozen-lockfile
Run pnpm build

if (-not $SkipPayload) {
  Step "2/7 Offline Python runtime (engine + libraries, pruned, precompiled)"
  Py scripts/lib/assemble_payload.py --platform $Platform --flavor $Flavor --payload $Payload

  Step "3/7 llama.cpp for $Platform"
  $llamaArgs = @("--platform", $Platform, "--out", (Join-Path $Payload "llama"))
  if ($NoVulkan) { $llamaArgs += "--no-vulkan" }
  if ($Flavor -eq "cuda") { Need nvcc "CUDA toolkit (build machine only; users need just the NVIDIA driver)"; $llamaArgs += "--cuda" }
  Py scripts/lib/build_llamacpp.py @llamaArgs

  Step "3b/7 App-local MSVC runtime (no VC++ redistributable needed on user machines)"
  foreach ($target in @((Join-Path $Payload "llama"), (Join-Path $Payload "python"))) {
    Copy-Item (Join-Path $crtDir.FullName "*.dll") $target -Force
  }

  Step "4/7 Starter model"
  if ($SmokeModel) { Py scripts/lib/starter_model.py --platform $Platform --payload $Payload --smoke }
  else { Py scripts/lib/starter_model.py --platform $Platform --payload $Payload }
}

$signConfig = @{}
if ($env:WINDOWS_CERTIFICATE_THUMBPRINT) {
  Step "5/7 Signing nested executables and libraries"
  $signtool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" | Sort-Object FullName -Descending | Select-Object -First 1
  # Sign only files that are not already signed by their vendor (e.g. python.exe from the PSF).
  $files = @(Get-ChildItem $Payload -Recurse -Include *.exe, *.dll, *.pyd | Where-Object { -not (Get-AuthenticodeSignature $_.FullName).SignerCertificate } | ForEach-Object { $_.FullName })
  for ($i = 0; $i -lt $files.Count; $i += 100) {
    $batch = $files[$i..([math]::Min($i + 99, $files.Count - 1))]
    Run $signtool.FullName sign /sha1 $env:WINDOWS_CERTIFICATE_THUMBPRINT /fd sha256 /tr http://timestamp.digicert.com /td sha256 @batch
  }
  $signConfig = @{ bundle = @{ windows = @{ certificateThumbprint = $env:WINDOWS_CERTIFICATE_THUMBPRINT } } }
} else {
  Step "5/7 Signing: skipped (WINDOWS_CERTIFICATE_THUMBPRINT not set; SmartScreen will warn on unsigned installers)"
}

Step "6/7 Integrity manifest"
Py scripts/lib/finalize_payload.py --platform $Platform --flavor $Flavor --payload $Payload
$env:INKPORT_MANIFEST_SHA256 = (Get-Content (Join-Path $Root "build\payload-$Platform.manifest.sha256") -Raw).Trim()

Step "7/7 Installers (Tauri: NSIS + MSI)"
# Merge the per-platform bundle config with optional signing settings.
$bundleConfig = Get-Content "src-tauri\bundle\$Platform.json" -Raw | ConvertFrom-Json
if ($signConfig.Count -gt 0) { $bundleConfig.bundle.windows | Add-Member -NotePropertyName certificateThumbprint -NotePropertyValue $env:WINDOWS_CERTIFICATE_THUMBPRINT -Force }
$merged = Join-Path $Root "build\tauri.$Platform.build.json"
$bundleConfig | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 $merged
$env:RUSTFLAGS = "-C target-feature=+crt-static"   # the Rust app itself needs no VC++ runtime either
$tauriArgs = @("tauri", "build", "--config", $merged, "--target", $Triple, "--bundles", $Bundles)
if ($DebugBuild) { $tauriArgs += "--debug" }
# NSIS and MSI cannot hold more than 2 GB. If the payload is too large for them, the bundling step
# fails AFTER the app binaries are built; the portable zip below is then the deliverable.
& pnpm @tauriArgs
$bundleOk = ($LASTEXITCODE -eq 0)
if (-not $bundleOk) { Write-Host "    note: installer bundling failed (payload likely over the 2 GB NSIS/MSI limit); shipping the portable zip" -ForegroundColor Yellow }

$buildProfile = if ($DebugBuild) { "debug" } else { "release" }
$targetDir = Join-Path $Root "target\$Triple\$buildProfile"
$bundleDir = Join-Path $targetDir "bundle"
Remove-Item -Recurse -Force $Out -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Out | Out-Null
if ($bundleOk -and $Bundles -like "*nsis*") { Copy-Item (Join-Path $bundleDir "nsis\*.exe") $Out }
if ($bundleOk -and $Bundles -like "*msi*") { Copy-Item (Join-Path $bundleDir "msi\*.msi") $Out }

Step "Portable zip (extract anywhere, double-click Inkport.exe; no installer, no size limit)"
$stage = Join-Path $Root "build\portable-$Platform"
Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
$appDir = Join-Path $stage "Inkport"
New-Item -ItemType Directory -Force -Path $appDir | Out-Null
foreach ($exe in @("inkport.exe", "inkport-cli.exe")) {
  if (-not (Test-Path (Join-Path $targetDir $exe))) { throw "$exe was not built" }
}
Copy-Item (Join-Path $targetDir "inkport.exe") (Join-Path $appDir "Inkport.exe")
Copy-Item (Join-Path $targetDir "inkport-cli.exe") $appDir
Copy-Item -Recurse $Payload (Join-Path $appDir "payload")
$zip = Join-Path $Out "Inkport_${Version}_x64-portable.zip"
Push-Location $stage
try { Run "$env:SystemRoot\System32\tar.exe" -a -c -f $zip Inkport } finally { Pop-Location }
Remove-Item -Recurse -Force $stage
Get-ChildItem $Out -File | ForEach-Object { "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name } | Set-Content -Encoding ASCII (Join-Path $Out "SHA256SUMS")

Step "Done"
Get-ChildItem $Out | ForEach-Object { "    {0,10:N0} MB  {1}" -f ($_.Length / 1MB), $_.Name }
Write-Host "    manifest $env:INKPORT_MANIFEST_SHA256"
Write-Host "    verify on a clean machine: powershell -ExecutionPolicy Bypass -File scripts\clean_machine_check.ps1 -Installer <path to .exe>"
