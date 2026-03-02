#!/usr/bin/env bash
set -euo pipefail

# Run from this script's directory so relative paths match train.py expectations.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------------------------
# Edit task list here
# ---------------------------
TASKS=(
  "lamp_on"
  "open_box"
  "pick_up_cup"
  "press_switch"
  "push_button"
  "put_books_on_bookshelf"
  "toilet_seat_up"
  "turn_tap"
  "play_jenga"
)

DATASET_ROOT="../data/train_data"

# CoppeliaSim environment
export COPPELIASIM_ROOT="${HOME}/.local/bin/CoppeliaSim"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
unset QT_QPA_PLATFORM

run_train() {
  local task="$1"
  local variant="$2"
  local action_sequence="$3"
  local execution_length="$4"
  local tail_noise_n="$5"

  local exp_name="${task}_diff50_${variant}"
  echo "============================================================"
  echo "Task: ${task} | Variant: ${variant}"
  echo "experiment_name=${exp_name}"
  echo "action_sequence=${action_sequence}, execution_length=${execution_length}, replay.tail_noise_n=${tail_noise_n}"
  echo "============================================================"

  xvfb-run -a -s "-screen 0 1024x768x24" env \
    COPPELIASIM_ROOT="${COPPELIASIM_ROOT}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH}" \
    python train.py \
      hydra/launcher=basic \
      "experiment_name=${exp_name}" \
      'hydra.run.dir=./exp_local/${now:%Y.%m.%d}/${now:%H%M%S}_${experiment_name}' \
      'hydra.sweep.dir=./exp_local/${now:%Y.%m.%d}/${now:%H%M}_${experiment_name}' \
      method=diffusion \
      "env=rlbench/${task}" \
      "env.dataset_root=${DATASET_ROOT}" \
      create_train_env=false \
      num_eval_episodes=50 \
      num_eval_envs=10 \
      demos=50 \
      pixels=true \
      is_imitation_learning=true \
      num_pretrain_steps=100000 \
      num_train_frames=0 \
      "action_sequence=${action_sequence}" \
      "execution_length=${execution_length}" \
      replay.nstep=1 \
      batch_size=64 \
      eval_every_steps=10000 \
      save_snapshot=true \
      snapshot_every_n=10000 \
      use_min_max_normalization=true \
      "replay.tail_noise_n=${tail_noise_n}" \
      num_gpus=1
}

for task in "${TASKS[@]}"; do
  # 1) action sequence = 16
  run_train "${task}" "a16" 16 8 0

  # 2) action sequence = 16 + noise padding length 8
  run_train "${task}" "a16_noise8" 16 8 8

  # 3) action sequence = 8
  run_train "${task}" "a8" 8 8 0
done

echo "All experiments finished."
