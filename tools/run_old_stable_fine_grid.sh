#!/usr/bin/env bash
set -uo pipefail

cd /root/code/stat_calibration_20260717
output_root=/tos-mlp-zgci/fxx/runs/stat_calibration_20260717/old_stable_fine_grid
weights=/tos-mlp-zgci/fxx/maskrcnn_particle_last.pth
mkdir -p "$output_root"

run_one() {
  local score=$1 mask=$2
  local name="score_${score/./}_mask_${mask/./}"
  local output_dir="$output_root/$name"
  local log_file="/root/code/stat_calibration_20260717/fine_${name}.log"
  mkdir -p "$output_dir"
  echo "[$name] starting" > "$log_file"
  if OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONPATH=. python \
    tools/evaluate_data_98986_baseline.py \
      --fixed-eval-dir fixed_eval_20_scaled \
      --weights "$weights" \
      --output-dir "$output_dir" \
      --segmenter maskrcnn \
      --score-threshold "$score" \
      --mask-threshold "$mask" \
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
      --spherical-roundness-threshold 0.9075 \
      --source-kind result-return \
      --limit 20 >> "$log_file" 2>&1
  then
    PYTHONPATH=. python tools/calibrate_particle_statistics.py \
      --system-particle-csv "$output_dir/system_particle_features.csv" \
      --reference-particle-csv fixed_eval_20_scaled/reference_particles.csv \
      --output-dir "$output_dir/calibration" >> "$log_file" 2>&1
    echo "[$name] complete" >> "$log_file"
  else
    echo "[$name] failed" >> "$log_file"
  fi
}

for score in 0.62 0.66 0.70; do
  for mask in 0.56 0.60 0.64 0.68; do
    run_one "$score" "$mask" &
  done
done
wait
echo "Fine threshold grid finished."
