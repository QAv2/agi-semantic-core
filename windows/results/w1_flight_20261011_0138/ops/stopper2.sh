#!/bin/bash
# Amendment 2 sitting cap: after DEADLINE, SIGINT the runner right after its next chat commit.
PID=$1; LOG=$2; DEADLINE=$3
pat='^  (C |W1-D-|[AB]-)'
while kill -0 $PID 2>/dev/null && [ $(date +%s) -lt $DEADLINE ]; do sleep 20; done
kill -0 $PID 2>/dev/null || { echo "runner exited before the cap"; exit 0; }
n0=$(grep -cE "$pat" $LOG)
echo "cap reached at $(date -u +%H:%M:%SZ) with $n0 commit lines; waiting for the next commit"
while kill -0 $PID 2>/dev/null; do
  n=$(grep -cE "$pat" $LOG)
  if [ "$n" -gt "$n0" ]; then kill -INT $PID; echo "SIGINT sent at $(date -u +%H:%M:%SZ) after $n commit lines"; exit 0; fi
  sleep 2
done
echo "runner exited on its own"
