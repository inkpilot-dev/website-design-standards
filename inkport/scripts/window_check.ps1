# Opens the real Inkport window on Windows and uses it the way a person would, with the real engine
# behind it: create a project, build the exam, train, read the verdict, start the local API, chat,
# stop a run; then closes the app, opens it again and checks everything is still there.
#
# The window is driven through WebView2's remote debugging port by scripts\gui_e2e.cjs (needs Node).
# Also checked here, because only a real Windows session can show them:
#   * the app installs and runs from a folder with spaces and non-Latin characters in its path, and
#     keeps its data in one too (as it does for a user whose Windows name is written that way);
#   * starting Inkport a second time does not open a second copy;
#   * ending the app abruptly (Task Manager) leaves no helper process behind;
#   * the installed files are unchanged afterwards.
#
#   pwsh scripts\window_check.ps1 -Installer .\Inkport_0.1.0_x64-setup.exe -Out window-check
param(
  [Parameter(Mandatory = $true)][string]$Installer,
  [string]$Out = "window-check",
  [int]$CdpPort = 9222,
  [int]$TrainTimeoutSec = 3600,
  # A short run: this proves the window, training, the gate and answering work together; it is not a quality benchmark.
  [string]$TrainOverrides = '{"sft":{"max_steps":12,"eval_every":12,"checkpoint_every":12},"stage2":{"mode":"none"},"gate":{"general_suite_n":8,"max_new_tokens":96,"max_general_drop_pts":100},"dev_eval_n":8}'
)
$ErrorActionPreference = "Stop"
$Installer = (Resolve-Path $Installer).Path
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Out = (Resolve-Path $Out).Path
$Script = Join-Path $PSScriptRoot "gui_e2e.cjs"
# Built from character codes so this file stays plain ASCII (Windows PowerShell 5 reads scripts as ANSI).
$jp = [string][char]0x65E5 + [char]0x672C + [char]0x8A9E      # Japanese
$ru = [string][char]0x0414 + [char]0x0430 + [char]0x043D + [char]0x043D + [char]0x044B + [char]0x0435   # Russian
$acc = [string][char]0x00E9 + [char]0x00FC                     # e-acute, u-umlaut
$Root = Join-Path $env:TEMP ("inkport window " + [guid]::NewGuid().ToString("N").Substring(0, 6))
$InstallDir = Join-Path $Root "Program files $jp $acc\Inkport"
$DataDir = Join-Path $Root "$ru $acc data\Inkport"
New-Item -ItemType Directory -Force -Path $Root, $DataDir | Out-Null
$script:fwRules = @()
$script:failed = 0
function Say($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Pass($m) { Write-Host ("    PASS [{0:HH:mm:ss}] {1}" -f (Get-Date), $m) -ForegroundColor Green }
function Bad($m) { Write-Host ("    FAIL [{0:HH:mm:ss}] {1}" -f (Get-Date), $m) -ForegroundColor Red; $script:failed++ }
function Ours($name) { @(Get-CimInstance Win32_Process -Filter "Name = '$name'" -ErrorAction SilentlyContinue | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($InstallDir, [StringComparison]::OrdinalIgnoreCase) }) }
function StopOurs { foreach ($n in @("inkport.exe", "inkport-cli.exe", "llama-server.exe", "python.exe")) { Ours $n | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } } }
function Cleanup {
  StopOurs
  foreach ($r in $script:fwRules) { Remove-NetFirewallRule -Name $r -ErrorAction SilentlyContinue }
  $un = Join-Path $InstallDir "uninstall.exe"
  if (Test-Path $un) { try { Start-Process -FilePath $un -ArgumentList @("/S", "_?=$InstallDir") -Wait } catch { } }
}
function Die($m) {
  Bad $m
  foreach ($f in @((Join-Path $DataDir "logs\last-engine-error.log"), (Join-Path $DataDir "logs\llama-server.log"))) {
    if (Test-Path -LiteralPath $f) { Write-Host "    --- $(Split-Path $f -Leaf) ---"; Get-Content -LiteralPath $f -Tail 25 | ForEach-Object { Write-Host "    $_" } }
  }
  Cleanup
  exit 1
}
function Screen($name) {
  # The whole desktop, so the real window (title bar and all) is on record next to the page screenshots.
  try {
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    $b = [System.Windows.Forms.SystemInformation]::VirtualScreen
    $bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size)
    $bmp.Save((Join-Path $Out "$name.png"), [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
  } catch { Write-Host "    note: no desktop screenshot ($($_.Exception.Message))" -ForegroundColor Yellow }
}
function StartApp {
  $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS = "--remote-debugging-port=$CdpPort"
  $p = Start-Process -FilePath (Join-Path $InstallDir "inkport.exe") -PassThru
  $null = $p.Handle
  return $p
}
function CloseApp($p) {
  # The way a person closes it: the window's close button. Nothing is training, so no question is asked.
  try { $null = $p.CloseMainWindow() } catch { }
  if (-not $p.WaitForExit(30000)) { Write-Host "    note: the window did not close by itself within 30 s; ending it" -ForegroundColor Yellow; try { $p.Kill() } catch { }; $null = $p.WaitForExit(10000); return $false }
  return $true
}
function Drive($phase) {
  & node $Script --mode=cdp "--cdp=http://127.0.0.1:$CdpPort" "--playwright=$PlaywrightDir" "--out=$Out" "--phase=$phase" "--train-timeout=$TrainTimeoutSec" | ForEach-Object { Write-Host "    $_" }
  return $LASTEXITCODE
}

Say "Installing into a folder with spaces and non-Latin characters: $InstallDir"
$p = Start-Process -FilePath $Installer -ArgumentList @("/S", "/D=$InstallDir") -PassThru
$null = $p.Handle
if (-not $p.WaitForExit(20 * 60 * 1000)) { try { $p.Kill() } catch { }; Die "the installer did not finish within 20 minutes" }
if ($p.ExitCode -ne 0) { Die "installer exited with $($p.ExitCode)" }
$Cli = Join-Path $InstallDir "inkport-cli.exe"
if (-not (Test-Path -LiteralPath (Join-Path $InstallDir "inkport.exe"))) { Die "inkport.exe is not in $InstallDir" }
Pass "installed"
$env:INKPORT_HOME = $DataDir
$env:INKPORT_TEST_OVERRIDES = $TrainOverrides
$env:INKPORT_BUNDLE_DIR = $null

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if ($isAdmin) {
  Get-ChildItem -LiteralPath $InstallDir -Recurse -Include *.exe | ForEach-Object {
    $name = "InkportWindowCheck-" + [guid]::NewGuid().ToString("N").Substring(0, 8)
    New-NetFirewallRule -Name $name -DisplayName $name -Direction Outbound -Program $_.FullName -RemoteAddress Internet -Action Block | Out-Null
    $script:fwRules += $name
  }
  Pass "outbound internet blocked for every installed program (Windows Firewall)"
} else { Write-Host "    note: not running as Administrator, so network isolation is not enforced" -ForegroundColor Yellow }

Say "Getting the test driver (build machine only; not part of Inkport)"
$PlaywrightRoot = Join-Path $env:TEMP "inkport-playwright"
New-Item -ItemType Directory -Force -Path $PlaywrightRoot | Out-Null
Push-Location $PlaywrightRoot
& npm install --no-audit --no-fund --silent playwright-core@1 | Select-Object -Last 3 | ForEach-Object { Write-Host "    $_" }
Pop-Location
$PlaywrightDir = Join-Path $PlaywrightRoot "node_modules\playwright-core"
if (-not (Test-Path $PlaywrightDir)) { Die "could not install playwright-core" }

Say "Opening Inkport and using it: project, exam, training, verdict, local API, chat, stop"
$app = StartApp
$code = Drive "first"
Screen "desktop-after-first-run"
if ($code -ne 0) { Die "the window test failed (see the lines above and the screenshots)" }
Pass "the window works end to end with the real engine, from a path with spaces and non-Latin characters"

Say "Starting Inkport a second time"
$second = Start-Process -FilePath (Join-Path $InstallDir "inkport.exe") -PassThru
$null = $second.Handle
if ($second.WaitForExit(30000)) {
  $copies = (Ours "inkport.exe").Count
  if ($copies -eq 1 -and -not $app.HasExited) { Pass "the second launch handed over to the open window and quit (one copy running)" } else { Bad "after a second launch $copies copies are running (first one alive: $(-not $app.HasExited))" }
} else { Bad "a second copy of Inkport stayed open"; try { $second.Kill() } catch { } }

Say "Ending a serving process abruptly (as Task Manager would)"
$serveOut = Join-Path $Root "serve.out"
$serve = Start-Process -FilePath $Cli -ArgumentList @("serve", "--project", "window-test", "--port", "11577") -PassThru -RedirectStandardOutput $serveOut -RedirectStandardError (Join-Path $Root "serve.err") -WindowStyle Hidden
$null = $serve.Handle
$deadline = (Get-Date).AddSeconds(240)
while ((Get-Date) -lt $deadline -and -not $serve.HasExited -and -not ((Test-Path $serveOut) -and (Select-String -Path $serveOut -Pattern "^RESULT" -Quiet))) { Start-Sleep -Seconds 2 }
$helpers = (Ours "llama-server.exe").Count
if ($serve.HasExited -or $helpers -lt 1) {
  Bad "the command-line server did not come up (helpers: $helpers)"
  if (Test-Path (Join-Path $Root "serve.err")) { Get-Content (Join-Path $Root "serve.err") -Tail 15 | ForEach-Object { Write-Host "    $_" } }
} else {
  Stop-Process -Id $serve.Id -Force
  $gone = $false
  foreach ($i in 1..20) { Start-Sleep -Seconds 1; if ((Ours "llama-server.exe").Count -eq 0) { $gone = $true; break } }
  if ($gone) { Pass "the answer engine ended with the app that started it (no orphan process)" } else { Bad "llama-server.exe kept running after its app was ended"; StopOurs }
}

Say "Closing Inkport with the window's close button"
$closed = CloseApp $app
Start-Sleep -Seconds 3
$left = @((Ours "inkport.exe") + (Ours "llama-server.exe") + (Ours "python.exe"))
if ($closed -and $left.Count -eq 0) { Pass "the app closed and left no process behind" } else { Bad "after closing: closed by itself = $closed, processes left = $(($left | ForEach-Object { $_.Name }) -join ', ')"; StopOurs }

Say "Opening Inkport again: the project, its exam and its trained version must still be there"
$app = StartApp
$code = Drive "reopen"
Screen "desktop-after-reopen"
$null = CloseApp $app
if ($code -ne 0) { Die "the window test after a restart failed" }
Pass "after a restart the project opens on 'Use it' and the trained model answers"

Say "Checking the installation is unchanged"
$doctor = (& $Cli doctor --full 2>$null | Where-Object { $_ -like "RESULT *" } | Select-Object -Last 1)
if ($doctor -and ($doctor.Substring(7) | ConvertFrom-Json).ok) { Pass "every installed file still matches the manifest" } else { Bad "the installation no longer verifies: $doctor" }
$strays = @(Get-ChildItem -LiteralPath $InstallDir -Recurse -File | Where-Object { $_.LastWriteTime -gt (Get-Date).AddHours(-3) -and $_.Extension -in ".pyc", ".log", ".tmp" })
if ($strays.Count -eq 0) { Pass "nothing was written into the installation folder" } else { Bad "$($strays.Count) new file(s) in the installation folder, e.g. $($strays[0].FullName)" }

Cleanup
Write-Host ""
if ($script:failed -gt 0) { Write-Host "Window check FAILED: $($script:failed) problem(s)." -ForegroundColor Red; exit 1 }
Write-Host "Window check passed. Screenshots are in $Out" -ForegroundColor Green
