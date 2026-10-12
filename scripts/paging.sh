#!/usr/bin/env bash
# One llama.cpp run with its weights mmapped, under a memory cap (MiB, 0 for
# none). Prints the run's output, then a line with the major faults, the MiB
# read from the model's disk, the peak anonymous and resident memory, and the
# memory locked (LLAMA_MLOCK_LIMIT_MB is passed on).
#   paging.sh <binary> <model> <cap MiB> <args...>
set -uo pipefail
BIN=$1 M=$2 CAP=$3; shift 3
CMD=(env ${LLAMA_MLOCK_LIMIT_MB:+LLAMA_MLOCK_LIMIT_MB=$LLAMA_MLOCK_LIMIT_MB} "$BIN")
DEV=$(basename "$(df --output=source "$M" | tail -1)")
sectors() { awk -v d="$DEV" '$3 == d {print $6}' /proc/diskstats; }

sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
r0=$(sectors)
if [ "$CAP" -gt 0 ]; then
  sudo -n systemd-run --scope --quiet -p MemoryMax="${CAP}M" -p MemorySwapMax=0 -- "${CMD[@]}" -m "$M" "$@" </dev/null &
else
  "${CMD[@]}" -m "$M" "$@" </dev/null &
fi
w=$! mf=0 anon=0 rss=0 lck=0 name=$(basename "$BIN" | cut -c1-15)
while [ -d /proc/$w ]; do
  for p in $(pgrep -x "$name"); do
    f=$(awk '{sub(/.*\) /,""); print $10}' /proc/$p/stat 2>/dev/null); [ -n "$f" ] && [ "$f" -gt $mf ] && mf=$f
    a=$(awk '/^RssAnon:/{print $2}' /proc/$p/status 2>/dev/null); [ -n "$a" ] && [ "$a" -gt $anon ] && anon=$a
    h=$(awk '/^VmHWM:/{print $2}' /proc/$p/status 2>/dev/null); [ -n "$h" ] && [ "$h" -gt $rss ] && rss=$h
    l=$(awk '/^VmLck:/{print $2}' /proc/$p/status 2>/dev/null); [ -n "$l" ] && [ "$l" -gt $lck ] && lck=$l
  done
  sleep 0.2
done
wait $w
echo "paging majflt $mf read_mb $(( ($(sectors) - r0) * 512 / 1048576 )) anon_mb $((anon / 1024)) rss_mb $((rss / 1024)) lck_mb $((lck / 1024))"
