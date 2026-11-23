#!/usr/bin/env bash
# Train on the bundled samples with a built Inkport and print the story of each run: practice
# scores and answers, the forgetting readings, which general questions were lost, the verdict.
# For finding out what a training recipe does on a real machine; not part of any installer.
#
#   scripts/train_probe.sh <path to inkport-cli> '<plan overrides as JSON>' ['<more overrides>' ...]
#
# '{}' is a run with the settings a user gets ("Quick run"). Each run uses its own data folder.
# Several runs can also be given in one argument, separated by ";;" (how the build workflow passes
# them: its "probe" input). A run can name another goal in front of its settings, with that goal's
# sample files and suggested rules:  docs_legal:{}   code_sql:{...}   custom:{}
set -uo pipefail
CLI="${1:?usage: train_probe.sh <inkport-cli> '<overrides json>' ...}"
shift
[ -x "$CLI" ] || { echo "not an executable: $CLI" >&2; exit 2; }
result() { sed -n 's/^RESULT //p' | tail -1; }
field() { tr -d '\n' | grep -o "\"$1\":[^,}]*" | head -1 | cut -d: -f2- | tr -d '"'; }
RUNS=()
for arg in "$@"; do
  while IFS= read -r line; do [ -n "$line" ] && RUNS+=("$line"); done < <(printf '%s\n' "$arg" | awk '{ gsub(/ *;; */, "\n"); print }')
done
n=0
for RUN in "${RUNS[@]}"; do
  n=$((n + 1))
  GOAL=business_rules; OVERRIDES="$RUN"
  case "$RUN" in business_rules:*|code_sql:*|docs_legal:*|custom:*|style:*) GOAL="${RUN%%:*}"; OVERRIDES="${RUN#*:}" ;; esac
  case "$GOAL" in
    business_rules) FILES="expenses.csv policy.md"; RULES='["category must be one of fuel, meals, software, office_supplies, travel","amount must be positive","if the description mentions Uber then category must be travel"]' ;;
    code_sql) FILES="orders.csv customers.csv helpers.py"; RULES='["must not mention DROP","no more than 120 words"]' ;;
    docs_legal) FILES="lease.md"; RULES='["must include a quote","no more than 80 words"]' ;;
    custom) FILES="support.csv"; RULES='["no more than 50 words","must mention \"thank you\""]' ;;
    # A style, not set replies: 120 questions, each with its own answer (tests/fixtures).
    style) FILES=""; STYLE_FILE="$(cd "$(dirname "$0")/.." && pwd)/tests/fixtures/shakespeare_examples.csv"; RULES='["no more than 50 words","must mention \"thou\""]' ;;
  esac
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/inkport-probe.XXXXXX")"
  export INKPORT_HOME="$WORK/home"
  printf '\n\033[1;34m==> Probe %s: %s  %s\033[0m\n' "$n" "$GOAL" "$OVERRIDES"
  t0=$(date +%s)
  SAMPLES=$("$CLI" samples 2>>"$WORK/log" | result | field samples)
  [ -f "$SAMPLES/expenses.csv" ] || { echo "    no bundled samples (see $WORK/log)"; tail -5 "$WORK/log"; continue; }
  TEMPLATE="$GOAL"; [ "$GOAL" = style ] && TEMPLATE=custom
  "$CLI" create-project --name "Probe" --template "$TEMPLATE" >/dev/null 2>>"$WORK/log"
  INPUTS=(); for f in $FILES; do INPUTS+=("$SAMPLES/$f"); done
  [ "$GOAL" = style ] && INPUTS=("$STYLE_FILE")
  "$CLI" ingest --project probe --inputs "${INPUTS[@]}" --rules "$RULES" 2>>"$WORK/log" | result | cut -c1-260 | sed 's/^/    exam: /'
  ARGS=(train --project probe --preset fast)
  [ "$OVERRIDES" = "{}" ] || ARGS+=(--overrides "$OVERRIDES")
  "$CLI" "${ARGS[@]}" 2>"$WORK/train.log" | result | cut -c1-600 >"$WORK/result"
  grep -E "^ *(stage|info|warn|eval|dev|gate|error) \|" "$WORK/train.log" | grep -v "promote \|verify_data\|stay awake" | cut -c1-330 | sed 's/^/    /'
  echo "    RESULT $(cat "$WORK/result")"
  # What the original model answered to the practice questions, next to the expected answers
  # (practice questions only; the hidden exam is never printed).
  if command -v python3 >/dev/null; then
    python3 - "$INKPORT_HOME/projects/probe" <<'PY' | cut -c1-420 | sed 's/^/    /'
import glob, json, sys
root = sys.argv[1]
dev = [json.loads(l) for l in open(root + "/data/dev.jsonl", encoding="utf-8")]
for f in glob.glob(root + "/cache/eval-*.json"):
    d = json.load(open(f, encoding="utf-8"))
    if "dev_outputs" not in d:
        continue
    for item, out, ok in zip(dev, d["dev_outputs"], d.get("dev_pass") or []):
        q = " ".join(item["messages"][-1]["content"].split())
        print("base %s | Q: %s" % ("pass" if ok else "FAIL", q[-150:]))
        print("          expected: %s" % " ".join(str(item.get("reference")).split())[:200])
        print("          original: %s" % " ".join(str(out).split())[:200])
PY
  fi
  echo "    took $(( $(date +%s) - t0 )) s"
  [ -n "${PROBE_KEEP:-}" ] || rm -rf "$WORK"
done
exit 0
