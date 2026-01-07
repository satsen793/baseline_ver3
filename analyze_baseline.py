"""Baseline analysis utility.

Reads outputs/baseline/episode_metrics.csv and step logs to produce:
- baseline_report.json: mean±SD and 95% CI for key metrics
- table_baseline.csv: single-row table-ready metrics for manuscript
- post_content_gain_by_modality.csv: mean gains per modality

Usage:
    python analyze_baseline.py --metrics outputs/baseline/episode_metrics.csv \
        --steps-dir outputs/baseline/steps --out-dir outputs/baseline/analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODALITY_NAMES = {0: "video", 1: "PPT", 2: "text", 3: "blog", 4: "article", 5: "handout"}


def ci95(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n <= 1:
        return 0.0
    sd = s.std(ddof=1)
    return float(1.96 * sd / np.sqrt(n))


def load_step_logs(steps_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(steps_dir.glob("seed_*.csv")):
        df = pd.read_csv(csv_path)
        df["seed"] = int(csv_path.stem.split("_")[-1])
        frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, required=True)
    ap.add_argument("--steps-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="outputs/baseline/analysis")
    args = ap.parse_args()

    metrics_df = pd.read_csv(args.metrics)
    steps_df = load_step_logs(Path(args.steps_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert columns
    for col in [
        "cumulative_reward",
        "final_mastery",
        "time_to_mastery",
        "question_accuracy",
        "blueprint_adherence",
        "content_rate",
        "post_content_gain",
        "wall_clock_ms",
    ]:
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

    # Summary stats
    summary = {
        "mean_reward": float(metrics_df["cumulative_reward"].mean()),
        "reward_sd": float(metrics_df["cumulative_reward"].std(ddof=1)),
        "reward_ci95": ci95(metrics_df["cumulative_reward"]),
        "mean_final_mastery": float(metrics_df["final_mastery"].mean()),
        "final_mastery_ci95": ci95(metrics_df["final_mastery"]),
        "mean_question_accuracy": float(metrics_df["question_accuracy"].mean()),
        "mean_blueprint_adherence": float(metrics_df["blueprint_adherence"].mean()),
        "blueprint_adherence_ci95": ci95(metrics_df["blueprint_adherence"]),
        "mean_content_rate": float(metrics_df["content_rate"].mean()),
        "mean_post_content_gain": float(metrics_df["post_content_gain"].mean()),
        "post_content_gain_ci95": ci95(metrics_df["post_content_gain"]),
    }
    ttm = metrics_df["time_to_mastery"].dropna()
    if not ttm.empty:
        summary["mean_time_to_mastery"] = float(ttm.mean())
        summary["time_to_mastery_ci95"] = ci95(ttm)
    else:
        summary["mean_time_to_mastery"] = None
        summary["time_to_mastery_ci95"] = None

    if "wall_clock_ms" in metrics_df.columns:
        summary["mean_wall_clock_ms"] = float(metrics_df["wall_clock_ms"].mean())
        summary["wall_clock_ms_ci95"] = ci95(metrics_df["wall_clock_ms"]) 

    # Table row (single)
    table_row = {
        "Method": "Rule-Based Baseline",
        "Time_to_Mastery": summary.get("mean_time_to_mastery", "NA"),
        "Cumulative_Reward": summary["mean_reward"],
        "Reward_SD": summary["reward_sd"],
        "Blueprint_Adherence_%": summary["mean_blueprint_adherence"],
        "Content_Rate": summary["mean_content_rate"],
        "Post_Content_Gain": summary["mean_post_content_gain"],
        "Question_Accuracy": summary["mean_question_accuracy"],
    }

    # Modality gains from step logs
    modality_out = pd.DataFrame()
    if not steps_df.empty:
        dfc = steps_df[steps_df["action_type"] == "content"].copy()
        if not dfc.empty:
            dfc["modality_name"] = dfc["modality"].map(MODALITY_NAMES)
            dfc["mastery_gain"] = pd.to_numeric(dfc["mastery_gain"], errors="coerce")
            modality_out = dfc.groupby("modality_name")["mastery_gain"].mean().reset_index()
            modality_out.to_csv(out_dir / "post_content_gain_by_modality.csv", index=False)

    # Write outputs
    (out_dir / "baseline_report.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([table_row]).to_csv(out_dir / "table_baseline.csv", index=False)

    print(f"Saved analysis to {out_dir}")


if __name__ == "__main__":
    main()
