#!/usr/bin/env bash
# Parallelize the camera-ready SUITE: fan out (experiment x seed) shards, each calling
# cp_experiments.py <exp> --scale <scale> --seed <s> (the real non-trivial drivers), then
# aggregate per-seed JSONs with combine_cp.py.
#
# Usage:
#   ./run_parallel.sh [WORKERS] [SEEDS] [SCALE] [EXPERIMENTS] [MODEL]
#   defaults: 30 20 full "c1 e7 e3 e1 e5 e6 e4a" cnn
#
# NOTE: -e is intentionally OFF (best-effort orchestrator). A single failed shard must not
# abort combine. The xargs exit code is captured via PIPESTATUS and propagated.
set -uo pipefail
cd "$(dirname "$0")"

WORKERS=${1:-30}
SEEDS=${2:-20}
SCALE=${3:-full}
EXPERIMENTS=${4:-"c1 e7 e3 e1 e5 e6 e4a"}
MODEL=${5:-cnn}
OUT_DIR="./camera_ready_results"
JSON_DIR="$OUT_DIR/json"
mkdir -p "$JSON_DIR" "$OUT_DIR/worker_logs"
LOG="$OUT_DIR/run_parallel.log"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] suite start: workers=$WORKERS seeds=$SEEDS scale=$SCALE model=$MODEL exps=[$EXPERIMENTS]" | tee "$LOG"

# Job list: INTERLEAVED (seed-major) so experiments run CONCURRENTLY rather than one-experiment-
# at-a-time. With resume-skip, done shards drop out instantly and the remaining experiments (e.g.
# e1 + c1) share the workers from the start instead of the heavy one waiting behind the others.
JOBS=()
for s in $(seq 0 $((SEEDS-1))); do
    for e in $EXPERIMENTS; do
        JOBS+=("$e $s")
    done
done
JOBS+=("e7cal 0")
JOBS+=("e7scal 0")
echo "total jobs: ${#JOBS[@]}" | tee -a "$LOG"

# Worker: runs one (exp, seed); reads SCALE/MODEL/CP_PYTHON from env.
# RESUME: if a valid JSON already exists for this shard (e.g. recovered from GCS after a
# preemption), skip it. This makes the whole suite resumable across preemptible interruptions.
cat > "$OUT_DIR/_worker.sh" <<'WORKER'
#!/usr/bin/env bash
set -uo pipefail
EXP=$1; SEED=$2
DIR=$(dirname "$0"); SCRIPT_DIR="$DIR/.."
OUT="$DIR/json/${EXP}_seed${SEED}.json"
LOGF="$DIR/worker_logs/${EXP}_s${SEED}.log"
PY="${CP_PYTHON:-python3}"
if [ -s "$OUT" ] && "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$OUT" >/dev/null 2>&1; then
    echo "${EXP} seed${SEED} SKIP (valid json already present -- resume)"
    exit 0
fi
START=$(date +%s)
"$PY" "$SCRIPT_DIR/cp_experiments.py" "$EXP" --scale "${SCALE:-full}" --seed "$SEED" \
    --model "${MODEL:-cnn}" --out "$OUT" > "$LOGF" 2>&1
RC=$?
echo "${EXP} seed${SEED} rc=$RC dur=$(( $(date +%s) - START ))s"
exit $RC      # propagate so xargs returns 123 on any failure (honest suite status)
WORKER
chmod +x "$OUT_DIR/_worker.sh"

# Pin each shard to ONE math thread. On a 128-vCPU box we run ~WORKERS single-threaded shards
# in parallel; without this every torch/numpy/BLAS process would spawn ~128 threads => N*N
# oversubscription and thrash. Throughput here comes from shard parallelism, not intra-op threads.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# Reduce CUDA fragmentation OOM when many shards share one GPU (the suggested mitigation in the
# OOM error). Harmless on CPU.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SCALE MODEL CP_PYTHON OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS PYTORCH_CUDA_ALLOC_CONF
printf "%s\n" "${JOBS[@]}" \
    | xargs -L1 -P "$WORKERS" -I {} bash -c '$0 $@' "$OUT_DIR/_worker.sh" {} \
    | tee -a "$LOG"
XARGS_RC=${PIPESTATUS[0]}
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] workers finished; xargs_rc=$XARGS_RC (123 => >=1 shard failed)" | tee -a "$LOG"

# Honest accounting: which expected shard JSONs are missing or invalid? (A shard can exit !=0
# OR exit 0 but write no/truncated JSON; both must count as failures so the suite status is
# never falsely DONE -- the silent-failure class we must avoid.)
FAILED=()
for j in "${JOBS[@]}"; do
    set -- $j; je=$1; js=$2
    f="$JSON_DIR/${je}_seed${js}.json"
    if [ ! -s "$f" ] || ! "${CP_PYTHON:-python3}" -c "import json,sys; json.load(open(sys.argv[1]))" "$f" >/dev/null 2>&1; then
        FAILED+=("${je}_seed${js}")
    fi
done
NTOT=${#JOBS[@]}; NFAIL=${#FAILED[@]}; NOK=$((NTOT - NFAIL))
if [ "$NFAIL" -gt 0 ]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] SHARDS: $NOK/$NTOT ok; FAILED ($NFAIL): ${FAILED[*]}" | tee -a "$LOG"
else
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] SHARDS: all $NTOT produced valid JSON" | tee -a "$LOG"
fi

# Aggregate per-seed JSONs (mean/std/CI of headline metrics per experiment). Best-effort.
"${CP_PYTHON:-python3}" combine_cp.py --dir "$JSON_DIR" --out "$OUT_DIR/summary.json" 2>&1 | tee -a "$LOG" \
    || echo "combine_cp.py exited nonzero" | tee -a "$LOG"

# Suite is DONE only if EVERY expected shard produced valid JSON and no worker errored.
SUITE_RC=$XARGS_RC
[ "$NFAIL" -gt 0 ] && SUITE_RC=1
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] suite complete (xargs_rc=$XARGS_RC, failed_shards=$NFAIL, suite_rc=$SUITE_RC)" | tee -a "$LOG"
exit "$SUITE_RC"
