#!/usr/bin/env bash
# Scheduled paper-trading job for a fresh cloud session (Alpaca PAPER account only).
#   bash src/paper_job.sh entry    # start ~15:05 ET: setup, update data, wait until 15:45 ET, market-on-close buys
#   bash src/paper_job.sh exit     # start ~09:05 ET: market-on-open sells of every open position
# Orders are submitted only on or after SUBMIT_FROM (owner approved paper trading after a short dry run).
# Skips holidays and early-close days. Logs go to logs/paper/ and are committed to the working branch.
set -u
cd "$(dirname "$0")/.."
BRANCH=claude/youthful-edison-ca09h8
SUBMIT_FROM=2026-09-29
MODE=${1:-entry}
TODAY=$(TZ=America/New_York date +%F)
mkdir -p logs/paper
LOGF=logs/paper/job_${MODE}_${TODAY}.log
exec > >(tee -a "$LOGF") 2>&1
echo "== paper job $MODE $TODAY $(TZ=America/New_York date +%T) ET"

git fetch -q origin "$BRANCH" && git checkout -q "$BRANCH" && git -c pull.rebase=false pull -q --no-edit origin "$BRANCH"
pip install -q -r requirements.txt huggingface_hub 2>&1 | grep -v "root user" | tail -2

SESSION=$(python src/paper_overnight.py session 2>/dev/null | tail -1)
echo "session: $SESSION"
if [ "$SESSION" = "None" ]; then echo "market closed today"; exit 0; fi

SUBMIT=""
if [[ "$TODAY" > "$SUBMIT_FROM" || "$TODAY" == "$SUBMIT_FROM" ]]; then SUBMIT="--submit"; fi

commit_logs() {
  git add -f logs/paper data/store 2>/dev/null
  git commit -q -m "Paper job $MODE $TODAY" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
  for i in 1 2 3 4; do git -c pull.rebase=false pull -q --no-edit origin "$BRANCH" && git push -q origin "$BRANCH" && break; sleep $((2**i)); done
}

if [ "$MODE" = "exit" ]; then
  python src/paper_overnight.py exit $SUBMIT
  commit_logs
  exit 0
fi

# entry
if [ "$SESSION" != "('09:30', '16:00')" ]; then echo "early close, no entry today"; commit_logs; exit 0; fi
python src/hf_sync.py pull models
python src/update_data.py daily
python src/update_data.py earnings
python src/paper_overnight.py reconcile || true
# build the panel cache and earnings features now so the 15:45 run only fetches today's bars
python -c "import sys; sys.path.insert(0, 'src'); import ml_features as M; M.earnings_features(); print('panel ready', M.P['c'].index[-1].date())"
# wait until 15:46 ET: SIP bars up to 15:30 are then more than 15 minutes old (free plan); MOC cutoff is 15:50
while [ "$(TZ=America/New_York date +%H%M)" -lt 1546 ]; do sleep 20; done
echo "entry at $(TZ=America/New_York date +%T) ET, submit flag: '${SUBMIT}'"
python src/paper_overnight.py entry $SUBMIT
commit_logs
