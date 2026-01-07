"""Baseline plotting utilities.

Generates figures for the rule-based baseline from outputs/baseline:
- Learning curve (average cumulative reward vs step across seeds)
- Time-to-mastery distribution with mean±95% CI
- Post-content gain by modality (bar)
- Blueprint adherence (single value annotated)

Usage:
    python plot_baseline.py --metrics outputs/baseline/episode_metrics.csv \
        --steps-dir outputs/baseline/steps --out-dir figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="talk")

MODALITY_NAMES = {0: "video", 1: "PPT", 2: "text", 3: "blog", 4: "article", 5: "handout"}


def load_step_logs(steps_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(steps_dir.glob("seed_*.csv")):
        df = pd.read_csv(csv_path)
        df["seed"] = int(csv_path.stem.split("_")[-1])
        frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def plot_learning_curve(step_df: pd.DataFrame, out_dir: Path) -> None:
    if step_df.empty:
        return
    # Compute per-seed cumulative reward vs step
    step_df = step_df.copy()
    step_df["reward"] = step_df["reward"].astype(float)
    # Use global_step if present to avoid episode boundary resets
    step_col = "global_step" if "global_step" in step_df.columns else "step"
    curves = []
    for seed, df_s in step_df.groupby("seed"):
        df_s = df_s.sort_values(step_col)
        df_s["cum_reward"] = df_s["reward"].cumsum()
        curves.append(df_s[[step_col, "cum_reward"]].assign(seed=seed))
    curve_all = pd.concat(curves, ignore_index=True)
    # Average across seeds on common steps
    mean_curve = curve_all.groupby(step_col)["cum_reward"].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(mean_curve[step_col], mean_curve["cum_reward"], label="Baseline (avg)")
    plt.xlabel("Global Step" if step_col == "global_step" else "Step")
    plt.ylabel("Cumulative Reward (avg across seeds)")
    plt.title("Baseline Learning Curve (Average)")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "learning_curve_baseline.png"
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_time_to_mastery(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    # Filter non-empty time_to_mastery
    df = metrics_df.copy()
    df = df[df["time_to_mastery"].notna() & (df["time_to_mastery"] != "")]
    if df.empty:
        return
    df["time_to_mastery"] = df["time_to_mastery"].astype(float)
    mean = df["time_to_mastery"].mean()
    sd = df["time_to_mastery"].std(ddof=1)
    n = len(df)
    ci = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0

    plt.figure(figsize=(8, 5))
    sns.kdeplot(df["time_to_mastery"], fill=True, color="#4C78A8")
    plt.axvline(mean, color="k", linestyle="--", label=f"mean={mean:.1f}")
    plt.axvspan(mean - ci, mean + ci, color="orange", alpha=0.3, label="95% CI")
    plt.xlabel("Time to Mastery (steps)")
    plt.ylabel("Density")
    plt.title("Baseline Time-to-Mastery Distribution")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "time_to_mastery_baseline.png"
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_post_content_by_modality(steps_df: pd.DataFrame, out_dir: Path) -> None:
    if steps_df.empty:
        return
    df = steps_df.copy()
    df = df[df["action_type"] == "content"].copy()
    if df.empty:
        return
    df["modality_name"] = df["modality"].map(MODALITY_NAMES)
    df["mastery_gain"] = pd.to_numeric(df["mastery_gain"], errors="coerce")
    g = df.groupby("modality_name")["mastery_gain"].mean().reset_index()
    g = g.dropna()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=g, x="modality_name", y="mastery_gain", color="#72B7B2")
    plt.xlabel("Modality")
    plt.ylabel("Mean Post-Content Gain")
    plt.title("Baseline Post-Content Gain by Modality")
    plt.tight_layout()
    out_path = out_dir / "post_content_gain_by_modality_baseline.png"
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_blueprint_adherence(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    df = metrics_df.copy()
    df["blueprint_adherence"] = pd.to_numeric(df["blueprint_adherence"], errors="coerce")
    val = df["blueprint_adherence"].mean()

    plt.figure(figsize=(6, 4))
    plt.bar(["Baseline"], [val], color="#E17C05")
    plt.ylabel("Blueprint Adherence (%)")
    plt.title("Baseline Blueprint Adherence (mean across seeds)")
    plt.ylim(0, 100)
    plt.tight_layout()
    out_path = out_dir / "blueprint_adherence_baseline.png"
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, required=True, help="Path to episode_metrics.csv")
    ap.add_argument("--steps-dir", type=str, required=True, help="Directory with per-seed step logs")
    ap.add_argument("--out-dir", type=str, default="figures", help="Output directory for figures")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    steps_dir = Path(args.steps_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(metrics_path)
    steps_df = load_step_logs(steps_dir)

    plot_learning_curve(steps_df, out_dir)
    plot_time_to_mastery(metrics_df, out_dir)
    plot_post_content_by_modality(steps_df, out_dir)
    plot_blueprint_adherence(metrics_df, out_dir)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
