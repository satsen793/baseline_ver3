# Baseline (Rule-Based Heuristic) — Repro Scripts

This repo contains scripts to run and analyze the rule-based baseline controller for the adaptive learning simulator.

## Outputs
- `outputs/baseline/episode_metrics.csv`: per-seed episode metrics (seed, cumulative_reward, final_mastery, time_to_mastery, question_accuracy, blueprint_adherence, content_rate, post_content_gain, wall_clock_ms)
- `outputs/baseline/summary.json`: aggregate summary across seeds
- `outputs/baseline/steps/seed_*.csv`: per-step logs per seed (step, action_type, difficulty/modality, reward, mastery_gain, mastery_mean, correct)

## Run (baseline only)
```
python train_rule_based_baseline.py --save-step-logs --include-timing
```
Adjust `--seeds` and `--output-dir` if needed.

## Analyze and Plot
```
python analyze_baseline.py --metrics outputs/baseline/episode_metrics.csv --steps-dir outputs/baseline/steps --out-dir outputs/baseline/analysis
python plot_baseline.py --metrics outputs/baseline/episode_metrics.csv --steps-dir outputs/baseline/steps --out-dir figures
```
This will produce:
- `outputs/baseline/analysis/baseline_report.json` and `table_baseline.csv`
- `figures/learning_curve_baseline.png`, `time_to_mastery_baseline.png`, `post_content_gain_by_modality_baseline.png`, `blueprint_adherence_baseline.png`

## Requirements
```
numpy
pandas
matplotlib
seaborn
```
Install via:
```
pip install -r requirements.txt
```
