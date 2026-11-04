#!/bin/bash
# Run one condition's task list at the same time. Arguments: condition, list name.
COND=$1; LIST=$2
CFG=/root/run/cfg_$COND.yaml
case "$LIST" in
  full) SPECS=("swallow|jamcqa 20 100" "swallow|ifbench_singleturn 20 100" "swallow|mifeval_ja 15 60" "swallow|polymath_japanese:high 10 20" "swallow|polymath_japanese:top 10 20" "swallow|aime_N4 16 10");;
  nonmath) SPECS=("swallow|jamcqa 20 100" "swallow|ifbench_singleturn 20 100" "swallow|mifeval_ja 15 60");;
  light) SPECS=("swallow|jamcqa 20 100" "swallow|ifbench_singleturn 20 100" "swallow|mifeval_ja 15 60" "swallow|polymath_japanese:high 10 10" "swallow|polymath_japanese:top 10 10");;
esac
for spec in "${SPECS[@]}"; do
  set -- $spec
  bash /root/run/run_task.sh "$COND" "$CFG" "$1" "$2" "$3" &
  sleep 3
done
wait
echo "$(date +%T) COND_DONE $COND" | tee -a /root/out/stage.log
