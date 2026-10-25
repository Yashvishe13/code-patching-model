#!/usr/bin/env bash
# Run collect.py as N parallel workers, each with its own udocker storage dir (no shared image layers).
# All workers append to one runs/<name>/traces.jsonl (locked); the last to finish writes the summary.
#   UDOCKER_ROOT=<dir with ~6GB/worker> ./run_workers.sh 20 --policy teacher --split train --limit 100 --attempts 4 --name teacher_train
set -u
N=$1; shift
ROOT=${UDOCKER_ROOT:?set UDOCKER_ROOT to a directory with room for N workers}
NAME=$(printf '%s\n' "$@" | sed -n '/^--name$/{n;p;}')
mkdir -p "runs/$NAME/logs"
for i in $(seq 0 $((N - 1))); do
  w=$(printf 'w%02d' "$i")
  mkdir -p "$ROOT/$w"
  [ -d "$ROOT/$w/bin" ] || cp -r "$HOME/.udocker/bin" "$HOME/.udocker/lib" "$HOME/.udocker/doc" "$ROOT/$w/"
  UDOCKER_DIR="$ROOT/$w" python collect.py "$@" --shard "$i/$N" > "runs/$NAME/logs/$w.log" 2>&1 &
done
wait
echo "all $N workers done: runs/$NAME"
