#!/bin/bash
# Run the MALBO design/optimization in a fixed output dir; on a pause (exit 75 = Go quota / rate limit / infra problem) wait and
# resume, reusing every finished point and question. Run on the 2019 host from ~/dev/malbo-revival:
#   nohup malbo/tools/run_until_done.sh config.go3slots_reuse.yaml outputs/go3_init16 > go3_init16.driver.log 2>&1 &
# Env: ENV_FILES (default .env; e.g. ".env .env.go2": several Go subscriptions, each file overrides the API key; the first
# one whose limits are all ok is used, and a key that runs out is swapped for the next), MALBO_SCORER (exact|relaxed), WAIT_S (default 3600), MAX_ATTEMPTS (default 30), MAX_NO_PROGRESS (default 6 resumes without a new point/question).
set -u
CONFIG=${1:?config}; OUT=${2:?output dir relative to malbo/}
WAIT_S=${WAIT_S:-3600}; MAX_ATTEMPTS=${MAX_ATTEMPTS:-30}; MAX_NO_PROGRESS=${MAX_NO_PROGRESS:-6}
export PATH=$PATH:/usr/local/bin
cd "$(dirname "$0")/../.." || exit 1          # ~/dev/malbo-revival
mkdir -p "malbo/$OUT"
progress() { (cat "malbo/$OUT/evaluations.jsonl" 2>/dev/null | grep -vc '"reused": true'; cat malbo/$OUT/qcache/*.jsonl 2>/dev/null | wc -l) | paste -sd+ - | bc; }
ENV_FILES=${ENV_FILES:-.env}
choose() {   # SEL = first env file with every Go window ok; else MINWAIT = soonest reset (-1 = all weekly/monthly exhausted)
  SEL=""; MINWAIT=-1
  for f in $ENV_FILES; do
    w=$(python3 malbo/tools/go_usage.py --env "$f" --wait-seconds 2>/dev/null)
    [ "$w" = "0" ] && { SEL=$f; return; }
    [ -n "$w" ] && [ "$w" != "-1" ] && { [ "$MINWAIT" = "-1" ] || [ "$w" -lt "$MINWAIT" ]; } && MINWAIT=$w
  done
}
choose; USED=${SEL:-$(echo $ENV_FILES | cut -d' ' -f1)}
noprog=0
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  before=$(progress)
  echo "[$(date -u +%FT%TZ)] attempt $attempt (progress units before: $before) key from $USED"
  docker rm -f malbo-go3 >/dev/null 2>&1
  docker run --rm --name malbo-go3 --cpus 3 --memory 5g -v "$PWD":/work -w /work/malbo --env-file .env --env-file "$USED" \
    -e MPLBACKEND=Agg -e MALBO_API_BASE=https://opencode.ai/zen/go/v1 -e MALBO_MANAGER_MAX_STEPS=10 -e MALBO_SEARCH_MAX_STEPS=8 \
    -e MALBO_PARALLEL_TASKS=${MALBO_PARALLEL_TASKS:-20} -e MALBO_OUTPUT_DIR="$OUT" -e MALBO_SCORER=${MALBO_SCORER:-exact} \
    -e MALBO_EXTRA_HEADERS='{"x-opencode-session":"malbo-go3slots"}' \
    malbo-dev python run_multi_agent_optimization.py --config "$CONFIG" > "malbo/$OUT/run_attempt_$attempt.log" 2>&1
  rc=$?
  after=$(progress)
  echo "[$(date -u +%FT%TZ)] attempt $attempt exit $rc (progress units after: $after)"
  if [ "$rc" = 0 ]; then echo "DONE"; exit 0; fi
  if [ "$rc" != 75 ]; then echo "STOP: unexpected exit $rc (see malbo/$OUT/run_attempt_$attempt.log)"; exit "$rc"; fi
  if [ "$after" -gt "$before" ]; then noprog=0; else noprog=$((noprog+1)); fi
  if [ "$noprog" -ge "$MAX_NO_PROGRESS" ]; then echo "STOP: $noprog resumes without progress (monthly cap?)"; exit 76; fi
  # account-wide Go limit: ask GET /usage when every window is usable again; -1 = weekly/monthly exhausted -> stop
  choose
  if [ -n "$SEL" ] && [ "$SEL" != "$USED" ]; then echo "SWITCH: $USED -> $SEL (limits of $USED reached)"; USED=$SEL; continue; fi
  if [ -z "$SEL" ] && [ "$MINWAIT" = "-1" ]; then echo "STOP: Go weekly/monthly limit exhausted on every key ($ENV_FILES)"; exit 78; fi
  if [ -z "$SEL" ]; then wait=$MINWAIT; else wait=$(python3 -c "
import json,sys
from datetime import datetime,timezone
try: p=json.load(open('malbo/$OUT/progress.json'))
except Exception: print($WAIT_S); sys.exit()
best=None
for x in p.get('pauses') or []:
    r=x.get('resets_in_s')
    if r:
        left=r-(datetime.now(timezone.utc)-datetime.fromisoformat(x['ts'])).total_seconds()
        best=left if best is None else max(best,left)
print(int(max(300,best+300)) if best is not None else $WAIT_S)"); fi
  for f in $ENV_FILES; do echo "-- $f"; python3 malbo/tools/go_usage.py --env "$f" 2>/dev/null; done
  echo "PAUSED -> sleeping ${wait}s before resuming (resume at $(date -u -v+${wait}S +%FT%TZ 2>/dev/null || date -u -d "+${wait} sec" +%FT%TZ))"
  sleep "$wait"
  choose; [ -n "$SEL" ] && USED=$SEL
done
echo "STOP: max attempts"; exit 77
