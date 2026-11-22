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
  case "$RUN" in business_rules:*|code_sql:*|docs_legal:*|custom:*) GOAL="${RUN%%:*}"; OVERRIDES="${RUN#*:}" ;; esac
  case "$GOAL" in
    business_rules) FILES="expenses.csv policy.md"; RULES='["category must be one of fuel, meals, software, office_supplies, travel","amount must be positive","if the description mentions Uber then category must be travel"]' ;;
    code_sql) FILES="orders.csv customers.csv helpers.py"; RULES='["must not mention DROP","no more than 120 words"]' ;;
    docs_legal) FILES="lease.md"; RULES='["must include a quote","no more than 80 words"]' ;;
    custom) FILES="support.csv"; RULES='["no more than 50 words","must mention \"thank you\""]' ;;
  esac
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/inkport-probe.XXXXXX")"
  export INKPORT_HOME="$WORK/home"
  printf '\n\033[1;34m==> Probe %s: %s  %s\033[0m\n' "$n" "$GOAL" "$OVERRIDES"
  t0=$(date +%s)
  SAMPLES=$("$CLI" samples 2>>"$WORK/log" | result | field samples)
  [ -f "$SAMPLES/expenses.csv" ] || { echo "    no bundled samples (see $WORK/log)"; tail -5 "$WORK/log"; continue; }
  "$CLI" create-project --name "Probe" --template "$GOAL" >/dev/null 2>>"$WORK/log"
  INPUTS=(); for f in $FILES; do INPUTS+=("$SAMPLES/$f"); done
  "$CLI" ingest --project probe --inputs "${INPUTS[@]}" --rules "$RULES" 2>>"$WORK/log" | result | cut -c1-260 | sed 's/^/    exam: /'
  ARGS=(train --project probe --preset fast)
  [ "$OVERRIDES" = "{}" ] || ARGS+=(--overrides "$OVERRIDES")
  "$CLI" "${ARGS[@]}" 2>"$WORK/train.log" | result | cut -c1-600 >"$WORK/result"
  grep -E "^ *(stage|info|warn|eval|dev|gate|error) \|" "$WORK/train.log" | grep -v "promote \|verify_data\|stay awake" | cut -c1-330 | sed 's/^/    /'
  echo "    RESULT $(cat "$WORK/result")"
  echo "    took $(( $(date +%s) - t0 )) s"
  rm -rf "$WORK"
done
exit 0
