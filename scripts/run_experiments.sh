#!/usr/bin/env bash
# Phase runner for the g-series experiments on one GPU (RUNBOOK.md). Run from anywhere, ideally inside tmux.
#
#   bash scripts/run_experiments.sh validate              # RUNBOOK 1-2: tests, short LAFAN1 run, gate checks
#   bash scripts/run_experiments.sh prepare               # RUNBOOK 3: BONES-SEED download / convert / filter, then --inspect-only
#   # RUNBOOK 4 by hand: choose the category column and held-out categories, run make_split.py --out data/splits/g1
#   bash scripts/run_experiments.sh main data/splits/g1   # RUNBOOK 5-6: train, evaluate, report, margins, margin checks
#
# Every step skips work whose output already exists, so rerunning the same command resumes after an
# interruption (delete an output to redo that step). The script stops at the first failed command or check.
# Settings (environment variables): RUN=g1 LAFAN_HOURS=1.5 MAIN_HOURS=4 NUM_ENVS=4096 BATCH=4096 WORKERS=20
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${RUN:-g1}
LAFAN_HOURS=${LAFAN_HOURS:-1.5}
MAIN_HOURS=${MAIN_HOURS:-4}
NUM_ENVS=${NUM_ENVS:-4096}
BATCH=${BATCH:-4096}
WORKERS=${WORKERS:-20}
PY=(env -u PYTHONPATH uv run python)
TRAIN=(env -u PYTHONPATH uv run sm-train)
BONES=data/raw/bones_seed
SETS=(train_eval heldout_in heldout_cat)

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { printf '\n[STOP] %s\n' "$*" >&2; exit 1; }
hf_cli() { if uv run hf --help >/dev/null 2>&1; then uv run hf "$@"; else uv run huggingface-cli "$@"; fi; }

# Background jobs on the same GPU: launch <logfile> <command...>; wait_all fails if any job failed.
PIDS=()
LOGS=()
launch() { local logf=$1; shift; "$@" >"$logf" 2>&1 & PIDS+=("$!"); LOGS+=("$logf"); }
wait_all() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "FAILED: see ${LOGS[$i]}" >&2; failed=1; }
  done
  PIDS=()
  LOGS=()
  return "$failed"
}

train() {  # train <run name> <hours> <data args...>; skipped if finished, resumed if interrupted
  local name=$1 hours=$2; shift 2
  local dir=logs/$name resume=()
  if [ -f "$dir/.train_done" ]; then log "train $name: already done"; return; fi
  mkdir -p "$dir"
  if [ -f "$dir/ckpt_latest.pt" ]; then
    # max-hours counts again from the resume; lower LAFAN_HOURS / MAIN_HOURS for the rest if needed.
    log "train $name: resuming from $dir/ckpt_latest.pt"
    resume=(--resume "$dir/ckpt_latest.pt")
  fi
  log "train $name for $hours h (log: $dir/train_stdout.log)"
  "${TRAIN[@]}" --run-name "$name" "$@" ${resume[@]+"${resume[@]}"} --env.anchor-obs ori --env.num-envs "$NUM_ENVS" \
    --policy-trainer.batch-size "$BATCH" --schedule.max-hours "$hours" 2>&1 | tee -a "$dir/train_stdout.log"
  [ -f "$dir/ckpt_latest.pt" ] || die "training $name left no ckpt_latest.pt"
  touch "$dir/.train_done"
}

validate() {
  if [ "$(ls data/npz/lafan1/*.npz 2>/dev/null | wc -l)" -lt 40 ]; then
    log "LAFAN1: download and convert"
    hf_cli download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset --include "g1/*" --local-dir data/raw/lafan1
    "${PY[@]}" scripts/data/csv_to_npz_batch.py --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 --input-fps 30 --workers "$WORKERS"
  fi

  log "tests (log: logs/pytest.log)"
  mkdir -p logs
  env -u PYTHONPATH uv run pytest tests -q -rs 2>&1 | tee logs/pytest.log || die "tests failed (logs/pytest.log)"
  if grep -E "SKIPPED.*(needs CUDA|not built|not downloaded)" logs/pytest.log; then
    die "tests were skipped for missing CUDA or data, so the simulator checks did not run"
  fi
  env -u PYTHONPATH uv run pytest "tests/test_analytic_heads.py::test_anchor_relative_obs_matches_env" -q 2>&1 | tee logs/pytest_parity.log \
    || die "the ori/pos_ori sim-vs-imagination parity test failed (logs/pytest_parity.log)"
  grep -q "2 passed" logs/pytest_parity.log || die "the parity test did not pass for both modes (logs/pytest_parity.log)"

  train lafan_ori "$LAFAN_HOURS" --data.npz-dir data/npz/lafan1
  local dir=logs/lafan_ori ck=logs/lafan_ori/ckpt_latest.pt
  log "LAFAN1 checks: benchmark, drift, world-model error, margin smoke test (parallel)"
  [ -f "$dir/benchmark_ckpt_latest.json" ] || launch "$dir/eval_benchmark.log" "${PY[@]}" scripts/eval_benchmark.py --ckpt "$ck" --npz-dir data/npz/lafan1 --out "$dir/benchmark_ckpt_latest.json"
  [ -f "$dir/drift_ckpt_latest.json" ] || launch "$dir/eval_drift.log" "${PY[@]}" scripts/diag_drift.py --ckpt "$ck" --npz-dir data/npz/lafan1 --out "$dir/drift_ckpt_latest.json"
  [ -f "$dir/wm_error_ckpt_latest.json" ] || launch "$dir/eval_wm.log" "${PY[@]}" scripts/eval_wm_heldout.py --ckpt "$ck" --npz-dir data/npz/lafan1 --out "$dir/wm_error_ckpt_latest.json"
  [ -f "$dir/margins_ckpt_latest.json" ] || launch "$dir/eval_margins.log" "${PY[@]}" scripts/eval_margins.py --ckpt "$ck" --npz-dir data/npz/lafan1 --max-clips 4 --max-times 8 --out "$dir/margins_ckpt_latest.json"
  wait_all || die "an evaluation failed"
  "${PY[@]}" scripts/check_run.py lafan --run-dir "$dir" || die "LAFAN1 gate failed: fix this before running BONES-SEED"
  log "validate: all gates passed"
}

prepare() {
  if [ ! -f "$BONES/g1.tar.gz" ] || [ ! -f "$BONES/metadata/seed_metadata_v004.csv" ]; then
    log "BONES-SEED: download (needs 'uv run hf auth login' and the accepted dataset license)"
    hf_cli download bones-studio/seed g1.tar.gz metadata/seed_metadata_v004.csv --repo-type dataset --local-dir "$BONES"
  fi
  if [ ! -f "$BONES/g1_csv/.extracted" ]; then
    log "extract (into $BONES/g1_csv, away from the metadata CSV)"
    mkdir -p "$BONES/g1_csv"
    tar -xzf "$BONES/g1.tar.gz" -C "$BONES/g1_csv"
    touch "$BONES/g1_csv/.extracted"
  fi
  if [ ! -f data/npz/bones_seed/manifest.csv ]; then
    log "convert 120 fps CSV -> 50 Hz npz"
    "${PY[@]}" scripts/data/csv_to_npz_batch.py --input-dir "$BONES/g1_csv" --output-dir data/npz/bones_seed --input-fps 120 --workers "$WORKERS"
  fi
  if [ ! -f data/npz/bones_seed/manifest_filtered.csv ]; then
    log "name + feasibility filter (expect about 129,491 kept)"
    "${PY[@]}" scripts/data/filter_bones_seed.py --npz-dir data/npz/bones_seed --workers "$WORKERS"
  fi
  log "metadata columns and category sizes (choose the category column and held-out categories from this)"
  mkdir -p logs
  "${PY[@]}" scripts/data/make_split.py --manifest data/npz/bones_seed/manifest_filtered.csv --metadata "$BONES/metadata/seed_metadata_v004.csv" --inspect-only \
    | tee logs/bones_inspect.txt
  log "next: make_split.py --category-col ... --heldout-categories ... --train-hours 30 --out data/splits/$RUN, then: bash scripts/run_experiments.sh main data/splits/$RUN"
}

main_run() {
  local split=${1:?usage: run_experiments.sh main <split dir>}
  local dir=logs/$RUN ck=logs/$RUN/ckpt_latest.pt s k
  for f in train_manifest.csv clip_categories.csv "${SETS[@]/%/.txt}"; do
    [ -f "$split/$f" ] || die "$split/$f missing: run make_split.py first"
  done
  mkdir -p "$dir" data/library
  if [ ! -f "data/library/${RUN}_train.pt" ]; then
    log "pack the training library"
    "${PY[@]}" scripts/data/build_library.py --npz-dir data/npz/bones_seed --manifest "$split/train_manifest.csv" --out "data/library/${RUN}_train.pt"
  fi
  train "$RUN" "$MAIN_HOURS" --data.library "data/library/${RUN}_train.pt"

  log "benchmark and world-model error on ${SETS[*]} (parallel)"
  for s in "${SETS[@]}"; do
    [ -f "$dir/benchmark_$s.json" ] || launch "$dir/eval_benchmark_$s.log" "${PY[@]}" scripts/eval_benchmark.py --ckpt "$ck" \
      --npz-dir data/npz/bones_seed --clip-list "$split/$s.txt" --out "$dir/benchmark_$s.json"
    [ -f "$dir/wm_$s.json" ] || launch "$dir/eval_wm_$s.log" "${PY[@]}" scripts/eval_wm_heldout.py --ckpt "$ck" \
      --npz-dir data/npz/bones_seed --clip-list "$split/$s.txt" --categories "$split/clip_categories.csv" --out "$dir/wm_$s.json"
  done
  wait_all || die "an evaluation failed"

  log "report (logs/$RUN/report.txt)"
  "${PY[@]}" scripts/report_split.py --categories "$split/clip_categories.csv" \
    --bench "train_eval=$dir/benchmark_train_eval.json" "heldout_in=$dir/benchmark_heldout_in.json" "heldout_cat=$dir/benchmark_heldout_cat.json" \
    --wm "train_eval=$dir/wm_train_eval.json" "heldout_in=$dir/wm_heldout_in.json" "heldout_cat=$dir/wm_heldout_cat.json" \
    --out "$dir/report.json" | tee "$dir/report.txt"

  log "robustness margins: state and wm kinds on train_eval and heldout_cat (parallel)"
  for s in train_eval heldout_cat; do
    for k in state wm; do
      [ -f "$dir/margins_${s}_$k.json" ] || launch "$dir/eval_margins_${s}_$k.log" "${PY[@]}" scripts/eval_margins.py --ckpt "$ck" \
        --npz-dir data/npz/bones_seed --clip-list "$split/$s.txt" --categories "$split/clip_categories.csv" --kind "$k" --out "$dir/margins_${s}_$k.json"
    done
  done
  wait_all || die "a margin run failed"
  "${PY[@]}" scripts/check_run.py margins "$dir"/margins_*.json || die "margin controls failed: the margins are not trustworthy"
  log "main: done. Results: $dir/report.txt and $dir/margins_*.json (fill in the RUNBOOK results table)"
}

case "${1:-}" in
  validate) validate ;;
  prepare) prepare ;;
  main) shift; main_run "$@" ;;
  *) sed -n '2,11p' "$0"; exit 1 ;;
esac
