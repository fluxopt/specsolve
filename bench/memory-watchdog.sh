#!/usr/bin/env bash
# Kill the case that is about to take the box, so the run survives it.
#
# The memory budget in `bench/conftest.py` is a projection made between rungs,
# and cannot see a cell while it runs. This is a backstop: it samples, so it
# cannot catch an allocation faster than its interval. A killed case leaves the
# ones after it their turn.
set -uo pipefail

#: The running case, by the flag only its pytest carries.
CASE='--benchmark-memory'
#: The memory is in the `benchmem(isolate=True)` spawn child, which carries none
#: of pytest's arguments; killing the pytest alone orphans it.
SPAWNED='multiprocessing.spawn import spawn_main'

available() { free -m | awk '/^Mem:/{print $7}'; }

# Children first: once the pytest is gone its child is reparented, and only its
# own argv is left to find it by.
stop_the_case() {
  local signal=$1 pid
  for pid in $(pgrep -f -- "$CASE" 2>/dev/null); do
    pkill "-$signal" -P "$pid" 2>/dev/null || true
  done
  pkill "-$signal" -f -- "$CASE" 2>/dev/null || true
  pkill "-$signal" -f -- "$SPAWNED" 2>/dev/null || true
}

# `free` is procps, so this samples on Linux only; elsewhere it exits 0 so the run goes on.
if ! command -v free >/dev/null 2>&1; then
  echo "memory watchdog: no \`free\` here, so nothing is watching — a cell too big for this machine will take it down"
  exit 0
fi

#: A killed cell is recorded here, since its own process is the one killed. The
#: cell's name comes from the breadcrumb `bench/conftest.py` writes before each test.
INFLIGHT=bench/results/.inflight
CASUALTIES=${BENCH_CASUALTIES:-bench/results/casualties.json}

record_the_casualty() {
  local avail=$1 peak=$2 cell
  cell=$(cat "$INFLIGHT" 2>/dev/null) || cell=''
  [ -n "$cell" ] || cell='unknown'
  mkdir -p "$(dirname "$CASUALTIES")"
  [ -s "$CASUALTIES" ] || printf '[]' > "$CASUALTIES"
  python3 - "$CASUALTIES" "$cell" "$avail" "$peak" <<'PYEOF'
import json, sys
path, cell, avail, peak = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = json.loads(open(path).read() or '[]')
rows.append({'record': 'casualty', 'cell': cell, 'available_mb': avail, 'peak_mb': peak})
open(path, 'w').write(json.dumps(rows, indent=1))
PYEOF
  echo "recorded: ${cell} did not fit — ${avail} MB free at the kill, ${peak} MB high-water"
}

interval=${BENCH_MEMORY_SAMPLE_SECONDS:-0.25}
#: A line on the clock as well as on a new maximum, so a quiet watchdog does not read as a dead one.
heartbeat=${BENCH_MEMORY_HEARTBEAT_SECONDS:-60}
total=$(free -m | awk '/^Mem:/{print $2}')
floor=${BENCH_MEMORY_FLOOR_MB:-$((total / 4))}
echo "memory watchdog: ${total} MB total, a case is killed under ${floor} MB available"

peak=0
beat=$SECONDS
while sleep "$interval"; do
  read -r used avail <<<"$(free -m | awk '/^Mem:/{print $3, $7}')"
  if [ "$used" -gt "$peak" ]; then
    peak=$used
    echo "MEM high-water ${peak} MB"
  fi
  if [ $((SECONDS - beat)) -ge "$heartbeat" ]; then
    beat=$SECONDS
    echo "MEM ${used} MB used, ${avail} MB available, high-water ${peak} MB"
  fi
  [ "$avail" -ge "$floor" ] && continue

  echo "MEM ${avail} MB available, under the ${floor} MB floor — killing this case before it takes the box"
  record_the_casualty "$avail" "$peak"
  stop_the_case TERM
  # Poll rather than sleep: the next case starts the moment this one dies, while the box is still full.
  waited=0
  while :; do
    sleep "$interval"
    waited=$((waited + 1))
    avail=$(available)
    if [ "$avail" -ge "$floor" ]; then
      echo "MEM ${avail} MB available again — the ladder goes on"
      break
    fi
    if [ "$waited" -eq 20 ]; then
      echo "MEM ${avail} MB still under the floor — the case has not let go, killing harder"
      stop_the_case KILL
    fi
    if [ "$waited" -ge 240 ]; then
      echo "MEM ${avail} MB still under the floor with nothing left to kill — the box is on its own"
      break
    fi
  done
done
