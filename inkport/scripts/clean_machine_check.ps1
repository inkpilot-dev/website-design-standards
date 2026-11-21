# Clean-machine verification for the Inkport Windows installer (NSIS .exe, .msi, or portable .zip).
#
# Proves that, using ONLY what is inside the installer, a user can verify the installation, build a
# practice exam from the bundled samples, get answers from the bundled model through the local
# OpenAI-compatible API, train, and pass the hidden-test gate, with no network access and no
# Python/Node/Rust/Visual Studio on the machine.
#
#   powershell -ExecutionPolicy Bypass -File scripts\clean_machine_check.ps1 -Installer .\Inkport_0.1.0_x64-setup.exe
#   ... -Installer .\Inkport_0.1.0_x64-portable.zip
#   ... -Quick             # everything except training (about 3 minutes instead of an hour on a CPU-only PC)
#   ... -TrainOverrides '{"sft":{"max_steps":8}}'   # custom plan overrides for the training step
#
# For the strongest result run it inside Windows Sandbox or a fresh VM (no developer tools installed).
# Run as Administrator to also block all outbound traffic of the installed programs with Windows
# Firewall rules (removed afterwards); otherwise network isolation is reported as not enforced.
param(
  [Parameter(Mandatory = $true)][string]$Installer,
  [int]$Port = 11541,
  [switch]$Quick,
  # The side check on general skills is switched off for this run (noisy on a handful of questions),
  # so training on the bundled samples must end in PROMOTE and answering with a trained version is
  # always exercised.
  [string]$TrainOverrides = '{"sft":{"max_steps":40,"eval_every":20,"checkpoint_every":20},"stage2":{"mode":"none"},"gate":{"general_suite_n":20,"max_general_drop_pts":100}}'
)
$ErrorActionPreference = "Stop"
$Installer = (Resolve-Path $Installer).Path
$Work = Join-Path $env:TEMP ("inkport-clean-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
$InstallDir = Join-Path $Work "app"
$Log = Join-Path $Work "cli.log"
New-Item -ItemType Directory -Force -Path $Work, (Join-Path $Work "home") | Out-Null
$script:fwRules = @()
$script:nsis = $false
function Say($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Pass($m) { Write-Host ("    PASS [{0:HH:mm:ss}] {1}" -f (Get-Date), $m) -ForegroundColor Green }
function Cleanup {
  foreach ($r in $script:fwRules) { Remove-NetFirewallRule -Name $r -ErrorAction SilentlyContinue }
  if ($script:nsis) {
    # Remove the Start Menu entry and the "Apps & features" registration the installer created.
    $un = Join-Path $InstallDir "uninstall.exe"
    if (Test-Path $un) { try { Start-Process -FilePath $un -ArgumentList @("/S", "_?=$InstallDir") -Wait } catch { } }
  }
}
function Fail($m) {
  Write-Host "    FAIL $m" -ForegroundColor Red
  if (Test-Path $Log) {
    Write-Host "    --- last lines of the Inkport log ---"
    Get-Content $Log | Where-Object { $_ -notmatch "not an adapter parameter" } | Select-Object -Last 40 | ForEach-Object { Write-Host "    $_" }
  }
  $llama = Join-Path $Work "home\inkport\logs\llama-server.log"
  if (Test-Path $llama) {
    Write-Host "    --- last lines of the answer engine log ---"
    Get-Content $llama -Tail 25 | ForEach-Object { Write-Host "    $_" }
  }
  Cleanup
  exit 1
}

Say "Installing $(Split-Path $Installer -Leaf) for the current user into $InstallDir (no admin needed)"
$tInstall = Get-Date
if ($Installer -like "*.exe") {
  $script:nsis = $true
  $p = Start-Process -FilePath $Installer -ArgumentList @("/S", "/D=$InstallDir") -PassThru
  $null = $p.Handle   # keep the handle so ExitCode is available after WaitForExit
  if (-not $p.WaitForExit(20 * 60 * 1000)) { try { $p.Kill() } catch { }; Fail "the installer did not finish within 20 minutes" }
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
$App = Get-ChildItem $InstallDir -Recurse -Filter "inkport.exe" | Select-Object -First 1
$Manifest = Get-ChildItem $InstallDir -Recurse -Filter "manifest.json" | Where-Object { $_.DirectoryName -like "*payload" } | Select-Object -First 1
if (-not $App) { Fail "Inkport.exe not found in the installation" }
if (-not $Cli) { Fail "inkport-cli.exe not found in the installation" }
if (-not $Manifest) { Fail "bundled payload not found in the installation" }
$size = (Get-ChildItem $InstallDir -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
Write-Host ("    installed in {0:N0} s; size on disk {1:N2} GB" -f ((Get-Date) - $tInstall).TotalSeconds, $size)
if ($script:nsis) {
  $lnk = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Inkport\Inkport.lnk"
  $lnk2 = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Inkport.lnk"
  if ((Test-Path $lnk) -or (Test-Path $lnk2)) { Pass "Start Menu shortcut created" } else { Write-Host "    note: no Start Menu shortcut found (silent installs may skip it)" -ForegroundColor Yellow }
}
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

# Runs inkport-cli.exe with the given arguments (passed verbatim; a plain function so PowerShell does
# not try to bind "--full" etc. as its own parameters), streams the tool's progress lines to the
# console, keeps them in cli.log, and returns the parsed RESULT line.
function InkCli {
  $CliArgs = $args
  $result = $null
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "Continue"   # stderr lines are progress, not failures
  & $Cli.FullName @CliArgs 2>&1 | ForEach-Object {
    if ($_ -is [System.Management.Automation.ErrorRecord]) {
      $line = $_.ToString()
      Add-Content -LiteralPath $Log -Value $line
      if ($line -notmatch "not an adapter parameter") { Write-Host ("      [{0:HH:mm:ss}] {1}" -f (Get-Date), $line) }
    } else {
      $s = [string]$_
      if ($s.StartsWith("RESULT ")) { $result = $s }
    }
  }
  $ErrorActionPreference = $prev
  if (-not $result) { return $null }
  return ($result.Substring(7) | ConvertFrom-Json)
}

# Windows PowerShell 5.1 drops embedded double quotes when it calls a program; escape them there.
$legacyArgs = ($PSVersionTable.PSVersion -lt [version]"7.3") -or ($PSNativeCommandArgumentPassing -eq "Legacy")
function JsonArg([string]$s) { if ($legacyArgs) { return $s.Replace('"', '\"') }; return $s }

# ---------------------------------------------------------------- checks
$t0 = Get-Date
Say "Checking the installation with no network and no developer tools"
$r = InkCli doctor --full
if (-not $r -or -not $r.ok -or -not $r.offline_core) { Fail "pre-flight failed" }
Pass "pre-flight: $($r.verify.files_checked) bundled files verified in $([math]::Round($r.verify.seconds, 1)) s; training on $($r.training_accelerator); answers on $(if ($r.inference_devices) { $r.inference_devices -join ', ' } else { 'CPU' })"

$models = InkCli models
$model = @($models)[0].id
if (-not $model) { Fail "no bundled base model" }
Pass "bundled starter model: $model"
$samples = (InkCli samples).samples
if (-not $samples -or -not (Test-Path -LiteralPath (Join-Path $samples "expenses.csv"))) { Fail "bundled sample data missing (CLI reported: $samples)" }

$p = InkCli create-project --name "Clean Check" --template business_rules
if (-not $p -or $p.slug -ne "clean-check") { Fail "create project" }
$ing = InkCli ingest --project clean-check --inputs (Join-Path $samples "expenses.csv") (Join-Path $samples "policy.md") --rules (JsonArg '["amount must be positive"]')
if (-not $ing -or -not $ing.leakage_free) { Fail "ingest" }
Pass "practice exam built from bundled samples (train $($ing.counts.train) / hidden test $($ing.counts.test), zero leakage)"

# Answers from the bundled model before any training: the bundled answer engine starts and serves.
$a = InkCli api-smoke --project clean-check --port $Port
if (-not $a -or -not $a.streaming -or $a.unauthorized_status -ne 401) { Fail "local API (bundled model)" }
Pass "bundled model answers on 127.0.0.1:${Port}: 401 without key, models listed, chat + streaming via $($a.engine)"

if (-not $Quick) {
  Say "Training on the bundled samples (slow without an NVIDIA GPU: expect 30-90 minutes on a processor)"
  # Every two minutes report free memory and the engine's memory/CPU use, so a stall or a PC that
  # is out of memory is visible in the output instead of looking like a silent wait.
  $sampler = $null
  if (Get-Command Start-ThreadJob -ErrorAction SilentlyContinue) {
    $sampler = Start-ThreadJob -ScriptBlock {
      while ($true) {
        Start-Sleep -Seconds 120
        try {
          $os = Get-CimInstance Win32_OperatingSystem
          $line = "      [{0:HH:mm:ss}] memory: {1:N1} of {2:N1} GB free" -f (Get-Date), ($os.FreePhysicalMemory / 1MB), ($os.TotalVisibleMemorySize / 1MB)
          $py = Get-Process python -ErrorAction SilentlyContinue | Sort-Object WorkingSet64 -Descending | Select-Object -First 1
          if ($py) { $line += "; engine: {0:N1} GB in RAM, {1:N1} GB reserved, {2:N0} s of processor time" -f ($py.WorkingSet64 / 1GB), ($py.PrivateMemorySize64 / 1GB), $py.CPU }
          [Console]::WriteLine($line)
        } catch { }
      }
    }
  }
  if ($model -eq "smoke-tiny") {
    $t = InkCli train --project clean-check --preset smoke
    if (-not $t -or $t.decision -ne "PROMOTE") { Fail "training/gate did not promote: $($t | ConvertTo-Json -Compress)" }
  } else {
    $t = InkCli train --project clean-check --preset fast --overrides (JsonArg $TrainOverrides)
    if (-not $t -or $t.decision -ne "PROMOTE") { Fail "training finished but the trained model was not promoted on the bundled samples: $($t | ConvertTo-Json -Compress -Depth 4)" }
  }
  if ($sampler) { Stop-Job $sampler -ErrorAction SilentlyContinue; Remove-Job $sampler -Force -ErrorAction SilentlyContinue }
  Pass "trained + gated on the hidden test: $($t.decision) $($t.version)"

  $a = InkCli api-smoke --project clean-check --port $Port
  if (-not $a -or -not $a.streaming -or $a.unauthorized_status -ne 401) { Fail "local API (after training)" }
  Pass "local API after training: chat + @base + streaming via $($a.engine)"
}

$r2 = InkCli doctor --full
if (-not $r2 -or -not $r2.ok) { Fail "installation changed during use" }
Pass "installation unchanged after use (full re-verification)"
$changed = Get-ChildItem $InstallDir -Recurse -File | Where-Object { $_.LastWriteTime -gt $before } | Select-Object -First 5
if ($changed) { Fail "files changed inside the installation: $($changed.FullName -join ', ')" }
Pass "installation folder untouched; all data under $env:INKPORT_HOME"
Write-Host ("    total: {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds)

Cleanup
if ($Quick) {
  Write-Host "`nQuick clean-machine check passed (install, integrity, practice exam, answers; training not run)." -ForegroundColor Green
} else {
  Write-Host "`nClean-machine check passed: one download, fully offline." -ForegroundColor Green
}
Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
exit 0
