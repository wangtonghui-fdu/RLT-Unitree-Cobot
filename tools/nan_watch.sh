#!/bin/bash
# nan_watch.sh LOG (11-20): the pi0.5 run went NaN between steps 1100 and 1150 after a resume and kept running for 40 minutes.
# Every 60 s: when a new "loss=nan" line appears in LOG, stop the supervisor's process group at once (so no NaN checkpoint is written
# and the last good checkpoint is not rotated out) and write NAN_STOP to /root/c2r/logs/pi05_nan_watch.log.  Exits with the supervisor.
LOG=$1; W=/root/c2r/logs/pi05_nan_watch.log
n0=$(grep -a -c "loss=nan" $LOG); echo "$(date '+%F %T') watching $LOG (nan lines so far: $n0)" >> $W
while :; do
  sleep 60
  S=$(pgrep -f "[r]un_pi05_r1_dual6f_main.sh" | head -1)
  [ -z "$S" ] && { echo "$(date '+%F %T') supervisor gone, watcher exits" >> $W; exit 0; }
  n=$(grep -a -c "loss=nan" $LOG)
  if [ "$n" -gt "$n0" ]; then
    echo "$(date '+%F %T') NaN detected: $(grep -a 'Step [0-9]*:' $LOG | tail -n 3 | tr '\n' ' ')" >> $W
    echo "[sup] $(date '+%F %T') STOPPED by nan_watch.sh: NaN in the log" >> /root/c2r/logs/pi05_dual6f_sup.log
    [ "$(ps -o pgid= -p $S | tr -d ' ')" = "$S" ] && kill -TERM -- -$S
    sleep 20; pgrep -g $S > /dev/null && kill -KILL -- -$S
    echo "$(date '+%F %T') NAN_STOP checkpoints: $(ls /root/c2r/pi05/openpi/checkpoints/pi05_r1_dual_v6f/dual6f/ | tr '\n' ' ')" >> $W; exit 1
  fi
done
