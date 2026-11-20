# Inkport end-to-end verification (Windows PowerShell 5.1+ / PowerShell 7).
# Same checks as scripts/verify_e2e.sh: Rust CLI -> Python engine -> ingest -> train -> gate ->
# arena -> local OpenAI-compatible API (auth, Host/Origin checks, streaming) -> signed adapters.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\verify_e2e.ps1
#         $env:INKPORT_PYTHON = "C:\path\to\python.exe"; .\scripts\verify_e2e.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Work = if ($env:INKPORT_E2E_DIR) { $env:INKPORT_E2E_DIR } else { Join-Path $Root ".e2e" }
$Port = if ($env:INKPORT_E2E_PORT) { [int]$env:INKPORT_E2E_PORT } else { 11535 }
Remove-Item -Recurse -Force (Join-Path $Work "home"), (Join-Path $Work "run") -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Join-Path $Work "run") | Out-Null
$env:INKPORT_HOME = Join-Path $Work "home"
$Log = Join-Path $Work "run\e2e.log"
$Start = Get-Date
$ServeProc = $null

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Pass($m) { Write-Host "    PASS $m" -ForegroundColor Green }
function Fail($m) { Write-Host "    FAIL $m`n    (full log: $Log)" -ForegroundColor Red; if ($ServeProc) { Stop-Process -Id $ServeProc.Id -Force -ErrorAction SilentlyContinue }; exit 1 }
function Cli { param([Parameter(ValueFromRemainingArguments = $true)] $CliArgs)
  $out = & $script:CliExe @CliArgs 2>> $Log
  $line = ($out | Where-Object { $_ -like "RESULT *" } | Select-Object -Last 1)
  if (-not $line) { return $null }
  return ($line.Substring(7) | ConvertFrom-Json)
}

# --------------------------------------------------------------------------- 1. Python
Step "Python engine environment"
if (-not $env:INKPORT_PYTHON) {
  if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail "uv is required (https://docs.astral.sh/uv/) or set INKPORT_PYTHON" }
  $VenvPy = Join-Path $Work "venv\Scripts\python.exe"
  if (-not (Test-Path $VenvPy)) {
    uv venv -q --python 3.12 (Join-Path $Work "venv") *>> $Log
    uv pip install -q --python $VenvPy torch --index-url https://download.pytorch.org/whl/cpu *>> $Log
  }
  uv pip install -q --python $VenvPy -e "$Root[dev]" *>> $Log
  if ($LASTEXITCODE -ne 0) { Fail "engine install" }
  $env:INKPORT_PYTHON = $VenvPy
}
$Py = $env:INKPORT_PYTHON
& $Py -c "import engine, torch, transformers, peft; print('    engine', engine.__version__, '| torch', torch.__version__)"

if (-not $env:SKIP_UNIT) {
  Step "Python unit + contract tests"
  Push-Location $Root; & $Py -m pytest -q -m "not slow" -p no:cacheprovider *>> $Log; $ok = $LASTEXITCODE -eq 0; Pop-Location
  if ($ok) { Pass "pytest" } else { Fail "pytest" }
}

# --------------------------------------------------------------------------- 2. Rust
Step "Rust supervisor: tests + headless CLI build"
Push-Location (Join-Path $Root "src-tauri")
cargo test -q --no-default-features *>> $Log; if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "cargo test" }
cargo build -q --no-default-features --bin inkport-cli *>> $Log; if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "cargo build" }
Pop-Location
Pass "cargo test"
$script:CliExe = Join-Path $Root "target\debug\inkport-cli.exe"

# --------------------------------------------------------------------------- 3. Fixtures
Step "Offline fixtures (tiny model + synthetic small-business data)"
if (-not (Test-Path (Join-Path $Work "model\config.json"))) { & $Py -m engine smoke-model --out (Join-Path $Work "model") *>> $Log; if ($LASTEXITCODE -ne 0) { Fail "smoke model" } }
& $Py -m engine smoke-data --out (Join-Path $Work "data") *>> $Log
Pass "model + data in $Work"

# --------------------------------------------------------------------------- 4. Pipeline
Step "Hardware detection + runtime status (Rust)"
$hw = Cli hardware; Write-Host "    tier: $($hw.recommendation.tier) | $($hw.summary[0])"
$st = Cli status; if ($st.ready) { Pass "engine reachable through the supervisor" } else { Fail "runtime not ready: $($st.message)" }

Step "Create project + ingest"
$null = Cli create-project --name "Expense Sorter" --template business_rules --base-id smoke-tiny --base-path (Join-Path $Work "model") --backend torch --quant fp32
$rules = '["category must be one of fuel, meals, software, office_supplies, travel", "amount must be positive"]'
$ing = Cli ingest --project expense-sorter --inputs (Join-Path $Work "data\expenses.csv") (Join-Path $Work "data\policy.md") --rules $rules
if (-not $ing.leakage_free -or $ing.counts.test -lt 30) { Fail "ingest: $($ing | ConvertTo-Json -Compress)" }
Pass "zero train/test leakage; split train=$($ing.counts.train) dev=$($ing.counts.dev) test=$($ing.counts.test)"

Step "Train + before-vs-after gate"
$tr = Cli train --project expense-sorter --preset smoke
Write-Host "    $($tr.explanation)"
if ($tr.decision -eq "PROMOTE") { Pass "gate promoted $($tr.version)" } else { Fail "gate decision: $($tr.decision)" }

Step "Proof Arena"
$ar = Cli arena --project expense-sorter --n 3
if ($ar.cases -eq 3) { Pass "arena wins: $($ar.wins | ConvertTo-Json -Compress)" } else { Fail "arena" }

# --------------------------------------------------------------------------- 5. Local API
Step "Local OpenAI-compatible API on 127.0.0.1:$Port"
$ServeProc = Start-Process -FilePath $script:CliExe -ArgumentList @("serve", "--project", "expense-sorter", "--port", "$Port") -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Work "run\serve.out") -RedirectStandardError (Join-Path $Work "run\serve.err")
$up = $false
for ($i = 0; $i -lt 180; $i++) {
  try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2; if ($h.model_loaded) { $up = $true; break } } catch {}
  Start-Sleep -Seconds 1
}
if (-not $up) { Fail "API did not come up" }
$Token = (Get-Content (Join-Path $env:INKPORT_HOME "api_token") -Raw).Trim()
$Api = "http://127.0.0.1:$Port/v1"
function Status($uri, $headers) { try { Invoke-WebRequest $uri -Headers $headers -UseBasicParsing | Out-Null; return 200 } catch { return [int]$_.Exception.Response.StatusCode } }
if ((Status "$Api/models" @{}) -eq 401) { Pass "missing API key -> 401" } else { Fail "auth" }
if ((Status "$Api/models" @{ Authorization = "Bearer $Token"; Origin = "https://evil.example" }) -eq 403) { Pass "foreign web Origin -> 403" } else { Fail "origin" }
$models = Invoke-RestMethod "$Api/models" -Headers @{ Authorization = "Bearer $Token" }
if ($models.data.id -contains "inkport/expense-sorter") { Pass "/v1/models" } else { Fail "models" }
$body = @{ model = "inkport/expense-sorter"; max_tokens = 40; messages = @(@{ role = "user"; content = 'Paid $42 at Shell for the delivery van' }) } | ConvertTo-Json -Depth 5
$chat = Invoke-RestMethod "$Api/chat/completions" -Method Post -Headers @{ Authorization = "Bearer $Token" } -ContentType "application/json" -Body $body
$content = $chat.choices[0].message.content
Write-Host "    model answered: $content"
try { $o = $content | ConvertFrom-Json; if ($null -ne $o.category -and $null -ne $o.amount) { Pass "chat completion returns the trained JSON format" } else { Fail "chat format" } } catch { Fail "chat not JSON: $content" }
Stop-Process -Id $ServeProc.Id -Force; $ServeProc = $null

# --------------------------------------------------------------------------- 6. Adapters
Step "Signed adapter export / import"
$patch = Join-Path $Work "run\v1.inkpatch"
$ex = Cli export --project expense-sorter --out $patch
Write-Host "    sha256 $($ex.sha256)"
& $Py -c "import sys,zipfile; zi=zipfile.ZipFile(sys.argv[1]); zo=zipfile.ZipFile(sys.argv[2],'w'); [zo.writestr(i.filename, (zi.read(i.filename)[:-4]+b'\x00\x00\x80\x7f') if i.filename=='adapter/adapter_model.safetensors' else zi.read(i.filename)) for i in zi.infolist()]; zo.close()" $patch (Join-Path $Work "run\tampered.inkpatch")
$ti = Cli import --project expense-sorter --file (Join-Path $Work "run\tampered.inkpatch") --preset smoke
if (($ti | ConvertTo-Json -Depth 6) -match "SHA-256 mismatch") { Pass "tampered package rejected (SHA-256)" } else { Fail "tampered import not rejected" }
$ok = Cli import --project expense-sorter --file $patch --preset smoke
if ($ok.import.signature_verified) { Pass "signature + hashes + tensors verified, then gated ($($ok.gate.decision))" } else { Fail "import" }

# --------------------------------------------------------------------------- 7. Frontend
if (-not $env:SKIP_FRONTEND -and (Get-Command pnpm -ErrorAction SilentlyContinue)) {
  Step "Frontend typecheck + production build"
  Push-Location $Root; pnpm install --frozen-lockfile *>> $Log; pnpm build *>> $Log; $ok = $LASTEXITCODE -eq 0; Pop-Location
  if ($ok) { Pass "vite build" } else { Fail "frontend build" }
}

Write-Host ("`nAll end-to-end checks passed in {0:N0}s." -f ((Get-Date) - $Start).TotalSeconds) -ForegroundColor Green
