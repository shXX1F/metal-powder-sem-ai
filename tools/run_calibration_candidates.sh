#!/usr/bin/env bash
set -uo pipefail

cd /root/code/stat_calibration_20260717

output_root=/tos-mlp-zgci/fxx/runs/stat_calibration_20260717/candidate_screen
mkdir -p "$output_root"

run_one() {
  local name=$1
  local weights=$2
  local score=$3
  local output_dir="$output_root/$name"
  local log_file="/root/code/stat_calibration_20260717/candidate_${name}.log"

  mkdir -p "$output_dir"
  echo "[$name] starting weights=$weights score=$score" > "$log_file"
  if OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. python \
    tools/evaluate_data_98986_baseline.py \
    --fixed-eval-dir fixed_eval_20_scaled \
    --weights "$weights" \
    --output-dir "$output_dir" \
    --segmenter maskrcnn \
    --score-threshold "$score" \
    --mask-threshold 0.50 \
    --max-detections 10000 \
    --max-inference-side 1280 \
    --tile-size 640 \
    --tile-overlap 192 \
    --tile-ownership-filter \
    --inference-mode merged \
    --device cuda \
    --infer-pixel-size-from-field-area \
    --diameter-method bbox_max \
    --roundness-method crofton \
    --spherical-roundness-threshold 0.91 \
    --source-kind result-return \
    --limit 20 >> "$log_file" 2>&1
  then
    PYTHONPATH=. python tools/calibrate_particle_statistics.py \
      --system-particle-csv "$output_dir/system_particle_features.csv" \
      --reference-particle-csv fixed_eval_20_scaled/reference_particles.csv \
      --output-dir "$output_dir/calibration" >> "$log_file" 2>&1
    echo "[$name] complete" >> "$log_file"
  else
    echo "[$name] evaluation failed" >> "$log_file"
  fi
}

run_one production_a0200 \
  /tos-mlp-zgci/fxx/runs/production_best_20260716/maskrcnn_particle_production_balanced_a0200_20260716.pth \
  0.59 &
run_one old_stable \
  /tos-mlp-zgci/fxx/maskrcnn_particle_last.pth \
  0.55 &
run_one round4_best_val \
  /tos-mlp-zgci/fxx/runs/train_maskrcnn_round4_random_crop_v1/maskrcnn_particle_best_val.pth \
  0.59 &
run_one round5_best_val \
  /tos-mlp-zgci/fxx/runs/train_maskrcnn_round5_dense_recall_v1/maskrcnn_particle_best_val.pth \
  0.59 &
run_one round5_last \
  /tos-mlp-zgci/fxx/runs/train_maskrcnn_round5_dense_recall_v1/maskrcnn_particle_last.pth \
  0.59 &

wait
echo "All candidate evaluations finished."
