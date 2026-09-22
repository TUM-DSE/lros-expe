#!/usr/bin/env bash
# Decode under a memory quota holding the weights and a third of the KV cache,
# as Linux reclaims weights, swaps the KV cache or keeps the model locked.
# Needs sudo and a swap file.  kvswap.sh <batched-bench> <model> [MiB] [swap]
set -uo pipefail
BIN=$1 M=$2 QUOTA=${3:-3399} SWAP=${4:-/scratch/swapfile}
BENCH="-npp 8192 -ntg 64 -npl 8 -c 66048 -b 2048 -ub 512 -fa -t 8 --no-perf"
LIMIT=14400
ORIG=$(cat /proc/sys/vm/swappiness)

# run <label> <swappiness> <mlock yes|no> <MemoryMax>: one line of results.
run() {
  local lbl=$1 out=/tmp/kv.${1%% *}.log flag="" peak=0 swp=0 mf=0 lck=0 p h s f l
  sudo -n sh -c "echo $2 > /proc/sys/vm/swappiness"
  sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  sudo -n swapoff "$SWAP" 2>/dev/null; sudo -n swapon "$SWAP"
  local i0 o0; i0=$(awk '/^pswpin/{print $2}' /proc/vmstat); o0=$(awk '/^pswpout/{print $2}' /proc/vmstat)
  [ "$3" = yes ] && flag="--mlock"
  sudo -n systemd-run --scope --quiet -p MemoryMax="$4" -p MemorySwapMax=12G \
    -- "$(which bash)" -c "ulimit -l unlimited; exec taskset -c 0-7 timeout $LIMIT $BIN -m $M $BENCH $flag" \
    > "$out" 2>&1 &
  local w=$!
  while kill -0 $w 2>/dev/null; do
    for p in $(pgrep -f 'llama-batched-benc[h]'); do
      h=$(awk '/^VmHWM:/{print $2}' /proc/$p/status 2>/dev/null); [ -n "$h" ] && [ "$h" -gt $peak ] && peak=$h
      s=$(awk '/^VmSwap:/{print $2}' /proc/$p/status 2>/dev/null); [ -n "$s" ] && [ "$s" -gt $swp ] && swp=$s
      l=$(awk '/^VmLck:/{print $2}' /proc/$p/status 2>/dev/null); [ -n "$l" ] && [ "$l" -gt $lck ] && lck=$l
      f=$(awk '{sub(/.*\) /,""); print $10}' /proc/$p/stat 2>/dev/null); [ -n "$f" ] && [ "$f" -gt $mf ] && mf=$f
    done
    sleep 0.3
  done
  wait $w
  local i1 o1 row pp tg; i1=$(awk '/^pswpin/{print $2}' /proc/vmstat); o1=$(awk '/^pswpout/{print $2}' /proc/vmstat)
  row=$(grep -E "^\| +[0-9]" "$out" | tail -1)
  pp=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$7); print $7}')
  tg=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$9); print $9}')
  printf "%-16s s_pp %-7s s_tg %-7s peakRSS %-6d swapped %-6d majflt %-10s locked %-6d swpin_M %-7d swpout_M %-7d\n" \
    "$lbl" "${pp:-FAIL}" "${tg:-FAIL}" \
    $((peak / 1024)) $((swp / 1024)) "$mf" $((lck / 1024)) \
    $(( (i1 - i0) * 4096 / 1048576 )) $(( (o1 - o0) * 4096 / 1048576 ))
}

echo "### 8 requests x 8192 tokens, quota ${QUOTA} MiB, started $(date -Is)"
run "A no-quota"      60  no  infinity
run "B swappiness0"   0   no  "${QUOTA}M"
run "C swappiness100" 100 no  "${QUOTA}M"
run "D mlock-model"   60  yes "${QUOTA}M"
sudo -n sh -c "echo $ORIG > /proc/sys/vm/swappiness"
echo "=== done $(date -Is)"
