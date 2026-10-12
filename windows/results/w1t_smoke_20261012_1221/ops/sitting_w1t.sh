#!/bin/bash
# W1-T sitting supervisor (ops only; the frozen runner is called unchanged). The W1 flight's
# ops/sitting.sh v2 adapted: W1-T's runner and commit lines; progress = new commit lines.
# Cap at DEADLINE (SIGINT right after the next chat commit). Restarts ONLY after the heading race
# ("posted message reads ''"), at most MAXR times, never twice in a row without a new chat logged.
# Everything else stops (same input failing twice = stop; "did not post" may be the app's 1095).
DEADLINE=$1; TAG=$2; D=$3; STAGE=$4; MAXR=${5:-2}
cd ~/agi-semantic-core || exit 1
mkdir -p $D
pat='^  (C |W1T-|[AB]-)'
r=0; prev=1
while :; do
  [ $(date +%s) -ge $DEADLINE ] && { echo "$(date -u +%H:%M:%SZ) cap reached between runs"; exit 0; }
  LOG=$D/run_${TAG}_$r.log
  DISPLAY=:0 python3 -u -m windows.run_w1t_ui run $STAGE --shots --dir $D > $LOG 2>&1 &
  PID=$!; echo "$(date -u +%H:%M:%SZ) run $r pid $PID log $LOG"
  sent=0; n0=-1
  while kill -0 $PID 2>/dev/null; do
    if [ $(date +%s) -ge $DEADLINE ]; then
      [ $n0 -lt 0 ] && { n0=$(grep -cE "$pat" $LOG); echo "$(date -u +%H:%M:%SZ) cap reached; waiting for the next commit"; }
      if [ $(grep -cE "$pat" $LOG) -gt $n0 ]; then kill -INT $PID; sent=1; echo "$(date -u +%H:%M:%SZ) SIGINT sent"; break; fi
      sleep 2
    else
      sleep 10
    fi
  done
  wait $PID; rc=$?
  n=$(grep -cE "$pat" $LOG)
  echo "$(date -u +%H:%M:%SZ) run $r exit $rc, $n commit lines"
  [ $sent = 1 ] && { echo "stopped at the cap"; exit 0; }
  [ $rc = 0 ] && { echo "RUNNER FINISHED (all parts done)"; exit 0; }
  last=$(grep -E 'UIError|STOPPED' $LOG | tail -1)
  echo "  error: $last"
  if echo "$last" | grep -qF "posted message reads ''"; then
    [ $n = 0 ] && [ $prev = 0 ] && { echo "stopping: two failures in a row with no new chat"; exit 3; }
    prev=$n
    r=$((r+1)); [ $r -gt $MAXR ] && { echo "too many restarts"; exit 3; }
    sleep 15; continue
  fi
  echo "stopping: not a known driver race"; exit 4
done
