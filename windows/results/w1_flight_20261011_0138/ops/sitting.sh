#!/bin/bash
# W1 sitting supervisor (ops only; the frozen runner is called unchanged).
# Runs the runner until DEADLINE (then SIGINT right after the next chat commit, as stopper2.sh).
# Restarts ONLY after the heading race ("posted message reads ''": the message posted, the driver read
# its heading too early), at most MAXR times, and never twice in a row without a new chat logged.
# Stops on anything else: "the input holds" is deterministic for a prompt (clarification 9), and
# "did not post" may be the app's 1095. Joe saw the old version loop on one chat (2026-10-12).
DEADLINE=$1; TAG=$2; MAXR=${3:-2}
cd ~/agi-semantic-core || exit 1
D=windows/results/w1_flight_20261011_0138
pat='^  (C |W1-D-|[AB]-)'
r=0; prev_ok=1
while :; do
  [ $(date +%s) -ge $DEADLINE ] && { echo "$(date -u +%H:%M:%SZ) cap reached between runs"; exit 0; }
  LOG=$D/run_${TAG}_$r.log
  DISPLAY=:0 python3 -u -m windows.run_w1_ui run flight --size 7x7 --shots --parts C,OBS,GENRE --dir $D > $LOG 2>&1 &
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
  ok=$(grep -c ': ok' $LOG)
  echo "$(date -u +%H:%M:%SZ) run $r exit $rc, $ok ok"
  [ $sent = 1 ] && { echo "stopped at the cap"; exit 0; }
  [ $rc = 0 ] && { echo "RUNNER FINISHED (all parts done)"; exit 0; }
  last=$(grep 'UIError' $LOG | tail -1)
  echo "  error: $last"
  if echo "$last" | grep -qF "posted message reads ''"; then
    [ $ok = 0 ] && [ $prev_ok = 0 ] && { echo "stopping: two failures in a row with no new chat"; exit 3; }
    prev_ok=$ok
    r=$((r+1)); [ $r -gt $MAXR ] && { echo "too many restarts"; exit 3; }
    sleep 15; continue
  fi
  echo "stopping: not a known driver race"; exit 4
done
