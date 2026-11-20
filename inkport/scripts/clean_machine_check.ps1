# Clean-machine verification for the Inkport Windows installer (NSIS .exe, .msi, or portable .zip).
#
# Proves that, using ONLY what is inside the installer, a user can verify the installation, build a
# practice exam from the bundled samples, train, run the hidden-test gate, and use the local
# OpenAI-compatible API, with no network access and no Python/Node/Rust/Visual Studio on the machine.
#
#   powershell -ExecutionPolicy Bypass -File scripts\clean_machine_check.ps1 -Installer .\Inkport_0.1.0_x64-setup.exe
#   ... -Installer .\Inkport_0.1.0_x64_en-US.msi
#
# For the strongest result run it inside Windows Sandbox or a fresh VM (no developer tools installed).
# Run as Administrator to also block all outbound traffic of the installed programs with Windows
# Firewall rules (removed afterwards); otherwise network isolation is reported as not enforced.
param(
  [Parameter(Mandatory = $true)][string]$Installer,
  [int]$Port = 11541
)
$ErrorActionPreference = "Stop"
$Installer = (Resolve-Path $Installer).Path
$Work = Join-Path $env:TEMP ("inkport-clean-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
$InstallDir = Join-Path $Work "app"
New-Item -ItemType Directory -Force -Path $Work, (Join-Path $Work "home") | Out-Null
function Say($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Pass($m) { Write-Host "    PASS $m" -ForegroundColor Green }
function Fail($m) { Write-Host "    FAIL $m" -ForegroundColor Red; $log = Join-Path $Work "cli.log"; if (Test-Path $log) { Write-Host "    --- last lines of the Inkport log ---"; Get-Content $log -Tail 40 | ForEach-Object { Write-Host "    $_" } }; Cleanup; exit 1 }
$fwRules = @()
function Cleanup { foreach ($r in $script:fwRules) { Remove-NetFirewallRule -Name $r -ErrorAction SilentlyContinue } }

Say "Installing $(Split-Path $Installer -Leaf) for the current user into $InstallDir (no admin needed)"
if ($Installer -like "*.exe") {
  $p = Start-Process -FilePath $Installer -ArgumentList @("/S", "/D=$InstallDir") -Wait -PassThru
  if ($p.ExitCode -ne 0) { Fail "installer exited with $($p.ExitCode)" }
} elseif ($Installer -like "*.msi") {
  # Administrative install = unpack without registering anything system-wide.
  $p = Start-Process msiexec.exe -ArgumentList @("/a", "`"$Installer`"", "/qn", "TARGETDIR=`"$InstallDir`"") -Wait -PassThru
  if ($p.ExitCode -ne 0) { Fail "msiexec exited with $($p.ExitCode)" }
} elseif ($Installer -like "*.zip") {
  New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
  & "$env:SystemRoot\System32\tar.exe" -x -f $Installer -C $InstallDir
  if ($LASTEXITCODE -ne 0) { Fail "could not extract the portable zip" }
} else { Fail "pass an NSIS .exe, an .msi or the portable .zip" }
$Cli = Get-ChildItem $InstallDir -Recurse -Filter "inkport-cli.exe" | Select-Object -First 1
$Manifest = Get-ChildItem $InstallDir -Recurse -Filter "manifest.json" | Where-Object { $_.DirectoryName -like "*payload" } | Select-Object -First 1
if (-not $Cli) { Fail "inkport-cli.exe not found in the installation" }
if (-not $Manifest) { Fail "bundled payload not found in the installation" }
$size = (Get-ChildItem $InstallDir -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
Write-Host ("    installed size: {0:N2} GB" -f $size)
$before = Get-Date

# ---------------------------------------------------------------- isolation
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if ($isAdmin) {
  Get-ChildItem $InstallDir -Recurse -Include *.exe | ForEach-Object {
    $name = "InkportCleanCheck-" + [guid]::NewGuid().ToString("N").Substring(0, 8)
    New-NetFirewallRule -Name $name -DisplayName $name -Direction Outbound -Program $_.FullName -RemoteAddress Internet -Action Block | Out-Null
    $script:fwRules += $name
  }
  Pass "outbound internet blocked for every installed program (Windows Firewall)"
} else {
  Write-Host "    note: not elevated, so network isolation is not enforced (run as Administrator, or disconnect the network)" -ForegroundColor Yellow
}
# A minimal environment: no Python, Node, Rust, Git, uv or Visual Studio on PATH.
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
foreach ($v in @("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "INKPORT_PYTHON", "INKPORT_BUNDLE_DIR")) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
$env:INKPORT_HOME = Join-Path $Work "home\inkport"
foreach ($tool in @("python", "py", "node", "cargo", "uv", "git", "cl")) {
  if (Get-Command $tool -ErrorAction SilentlyContinue) { Write-Host "    note: $tool is on this machine; Inkport does not use it" }
}

# Plain function (no param block): PowerShell would otherwise try to bind "--full" etc. as its own
# parameters. Everything after the name is passed verbatim to inkport-cli.exe.
function Cli {
  $CliArgs = $args
  $out = & $Cli.FullName @CliArgs 2>> (Join-Path $Work "cli.log")
  $line = $out | Where-Object { $_ -like "RESULT *" } | Select-Object -Last 1
  if (-not $line) { return $null }
  return ($line.Substring(7) | ConvertFrom-Json)
}

# ---------------------------------------------------------------- checks
$t0 = Get-Date
$r = Cli doctor --full
if (-not $r -or -not $r.ok -or -not $r.offline_core) { Fail "pre-flight failed: $(Get-Content (Join-Path $Work 'cli.log') -Tail 3)" }
Pass "pre-flight: $($r.verify.files_checked) bundled files verified in $([math]::Round($r.verify.seconds, 1)) s; training on $($r.training_accelerator); answers on $($r.inference_devices -join ', ')"

$models = Cli models
$model = $models[0].id
if (-not $model) { Fail "no bundled base model" }
Pass "bundled starter model: $model"
$samples = (Cli samples).samples
if (-not (Test-Path (Join-Path $samples "expenses.csv"))) { Fail "bundled sample data missing" }

$p = Cli create-project --name "Clean Check" --template business_rules
if ($p.slug -ne "clean-check") { Fail "create project" }
$ing = Cli ingest --project clean-check --inputs (Join-Path $samples "expenses.csv") (Join-Path $samples "policy.md") --rules '["amount must be positive"]'
if (-not $ing.leakage_free) { Fail "ingest" }
Pass "practice exam built from bundled samples (train $($ing.counts.train) / hidden test $($ing.counts.test), zero leakage)"

if ($model -eq "smoke-tiny") {
  $t = Cli train --project clean-check --preset smoke
  if ($t.decision -ne "PROMOTE") { Fail "training/gate did not promote: $($t | ConvertTo-Json -Compress)" }
} else {
  $t = Cli train --project clean-check --preset fast --overrides '{"sft":{"max_steps":40,"eval_every":20,"checkpoint_every":20},"stage2":{"mode":"none"},"gate":{"general_suite_n":20}}'
  if ($t.decision -notin @("PROMOTE", "REJECT", "INSUFFICIENT_DATA")) { Fail "training/gate did not finish" }
}
Pass "trained + gated on the hidden test: $($t.decision) $($t.version)"

$a = Cli api-smoke --project clean-check --port $Port
if (-not $a -or -not $a.streaming -or $a.unauthorized_status -ne 401) { Fail "local API: $(Get-Content (Join-Path $Work 'cli.log') -Tail 3)" }
Pass "local API on 127.0.0.1:${Port}: 401 without key, models listed, chat + @base + streaming via $($a.engine)"

$r2 = Cli doctor --full
if (-not $r2.ok) { Fail "installation changed during use" }
Pass "installation unchanged after use (full re-verification)"
$changed = Get-ChildItem $InstallDir -Recurse -File | Where-Object { $_.LastWriteTime -gt $before } | Select-Object -First 5
if ($changed) { Fail "files changed inside the installation: $($changed.FullName -join ', ')" }
Pass "installation folder untouched; all data under $env:INKPORT_HOME"
Write-Host ("    total: {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds)

Cleanup
Write-Host "`nClean-machine check passed: one download, fully offline." -ForegroundColor Green
Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
