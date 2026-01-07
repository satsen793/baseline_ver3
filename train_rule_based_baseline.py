"""Train/evaluate the rule-based baseline controller.

The script runs the deterministic heuristic defined in spec_baseline.md on the
simulator described in spec_simulator.md. Metrics follow spec_evaluation.md.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np

# Expect an AdaptiveLearningEnv implementation matching spec_simulator.md
try:
    from simulator import AdaptiveLearningEnv, SIMULATOR_CONFIG  # type: ignore
except ImportError as exc:  # pragma: no cover - environment may not yet exist
    raise SystemExit(
        "Missing simulator implementation. Ensure simulator.py defines AdaptiveLearningEnv."
    ) from exc


# Fallback simulator config mirroring spec_simulator.md values (used if SIMULATOR_CONFIG missing)
DEFAULT_SIMULATOR_CONFIG = {
    "num_los": 30,
    "num_questions_per_lo": 20,
    "num_contents_per_lo": 6,
    "max_episode_steps": 140,
    "irt": {
        "difficulty_ranges": {
            "easy": (-2.0, -0.5),
            "medium": (-0.5, 0.5),
            "hard": (0.5, 2.0),
        },
        "discrimination_ranges": {
            "easy": (0.5, 1.0),
            "medium": (1.0, 1.5),
            "hard": (1.5, 2.0),
        },
        "guessing_range": (0.1, 0.25),
    },
    "content": {
        "effectiveness_by_modality": {
            "video": (0.10, 0.15),
            "PPT": (0.08, 0.12),
            "text": (0.05, 0.08),
            "blog": (0.07, 0.10),
            "article": (0.06, 0.09),
            "handout": (0.05, 0.08),
        }
    },
    "reward_weights": {
        "correctness": 1.0,
        "mastery_gain": 0.5,
        "frustration_penalty": 0.3,
        "post_content_gain": 2.0,
    },
    "termination": {"mastery_threshold": 0.8, "max_frustration": 0.95},
}


@dataclass
class BaselineConfig:
    seeds: List[int]
    max_steps: int = 140
    f_max: float = 0.7
    tau_max: float = 60.0
    mastery_easy: float = 0.4
    mastery_medium: float = 0.7
    high_frustration: float = 0.6
    mastery_target: float = 0.8
    response_time_max: float = 120.0  # seconds used for normalization in simulator
    output_dir: Path = Path("outputs/baseline")
    include_timing: bool = False
    save_step_logs: bool = False
    total_steps: int | None = None


class RuleBasedController:
    def __init__(self, cfg: BaselineConfig):
        self.cfg = cfg
        self.num_los = 30
        self.num_difficulties = 3
        self.num_modalities = 6
        self.rng = np.random.default_rng(0)

    def set_seed(self, seed: int) -> None:
        """Seed the controller RNG for deterministic modality selection per episode."""
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray, fail_streak: int) -> int:
        masteries = state[: self.num_los]
        frustration = float(state[self.num_los])
        response_time_norm = float(state[self.num_los + 1])
        response_time = response_time_norm * self.cfg.response_time_max

        # Gate: content if frustration/time too high (per spec)
        force_content = frustration >= self.cfg.f_max or response_time >= self.cfg.tau_max

        lo_index = int(np.argmin(masteries))
        lo_mastery = float(masteries[lo_index])

        if force_content:
            # Content modality limited to handout/text/PPT/video per baseline spec
            if frustration > self.cfg.high_frustration:
                modality = int(self.rng.choice([5, 2]))  # handout or text
            else:
                modality = int(self.rng.choice([1, 0]))  # PPT or video
            action_id = 90 + (lo_index * self.num_modalities + modality)
            return action_id

        # Question path: mastery-band difficulty
        if lo_mastery < self.cfg.mastery_easy:
            difficulty = 0  # Easy
        elif lo_mastery < self.cfg.mastery_medium:
            difficulty = 1  # Medium
        else:
            difficulty = 2  # Hard

        action_id = lo_index * self.num_difficulties + difficulty
        return action_id


@dataclass
class EpisodeMetrics:
    seed: int
    cumulative_reward: float
    final_mastery: float
    time_to_mastery: int | None
    question_accuracy: float
    blueprint_adherence: float
    content_rate: float
    post_content_gain: float
    wall_clock_ms: float | None = None


DIFFICULTY_NAMES = {0: "easy", 1: "medium", 2: "hard"}
MODALITY_NAMES = {0: "video", 1: "PPT", 2: "text", 3: "blog", 4: "article", 5: "handout"}


def decode_action(action_id: int) -> Dict[str, int]:
    if action_id < 90:
        return {
            "type": "question",
            "lo": action_id // 3,
            "difficulty": action_id % 3,
        }
    content_id = action_id - 90
    return {
        "type": "content",
        "lo": content_id // 6,
        "modality": content_id % 6,
    }


def run_episode(env: AdaptiveLearningEnv, controller: RuleBasedController, seed: int, cfg: BaselineConfig) -> Tuple[List[Dict], EpisodeMetrics]:
    np.random.seed(seed)
    controller.set_seed(seed)
    obs = env.reset(seed=seed)
    done = False
    step = 0
    log: List[Dict] = []
    fail_streak = 0
    cumulative_reward = 0.0
    time_to_mastery: int | None = None

    while not done and step < cfg.max_steps:
        action = controller.select_action(obs, fail_streak)
        next_obs, reward, done, info = env.step(action)
        step += 1
        cumulative_reward += float(reward)

        action_decoded = decode_action(action)
        result = info.get("result", {})

        if action_decoded["type"] == "question":
            correct = bool(result.get("correct", False))
            fail_streak = 0 if correct else fail_streak + 1
        else:
            fail_streak = 0

        mean_mastery = float(info.get("mean_mastery", np.mean(next_obs[: controller.num_los])))
        if time_to_mastery is None and mean_mastery >= cfg.mastery_target:
            time_to_mastery = step

        log.append(
            {
                "step": step,
                "action_id": action,
                "action_type": action_decoded["type"],
                "difficulty": action_decoded.get("difficulty"),
                "modality": action_decoded.get("modality"),
                "reward": float(reward),
                "correct": result.get("correct"),
                "mastery_gain": result.get("mastery_gain"),
                "mastery_mean": mean_mastery,
            }
        )

        obs = next_obs

    metrics = compute_episode_metrics(log, cumulative_reward, cfg.mastery_target, seed)
    return log, metrics


def compute_episode_metrics(log: List[Dict], cumulative_reward: float, mastery_target: float, seed: int) -> EpisodeMetrics:
    if not log:
        return EpisodeMetrics(seed, 0.0, 0.0, None, 0.0, 100.0, 0.0, 0.0)

    final_mastery = float(log[-1].get("mastery_mean", 0.0))

    time_to_mastery = None
    for entry in log:
        if entry.get("mastery_mean", 0.0) >= mastery_target:
            time_to_mastery = entry["step"]
            break

    question_logs = [e for e in log if e["action_type"] == "question"]
    content_logs = [e for e in log if e["action_type"] == "content"]

    correct = sum(1 for e in question_logs if e.get("correct"))
    accuracy = correct / len(question_logs) if question_logs else 0.0

    blueprint_counts = {"easy": 0, "medium": 0, "hard": 0}
    for e in question_logs:
        diff = DIFFICULTY_NAMES.get(e["difficulty"], "easy")
        blueprint_counts[diff] += 1

    total_questions = sum(blueprint_counts.values())
    if total_questions == 0:
        adherence = 100.0
    else:
        actual = {
            "easy": blueprint_counts["easy"] / total_questions,
            "medium": blueprint_counts["medium"] / total_questions,
            "hard": blueprint_counts["hard"] / total_questions,
        }
        target = {"easy": 0.20, "medium": 0.60, "hard": 0.20}
        deviation = sum(abs(actual[d] - target[d]) for d in target) / 3
        adherence = (1.0 - deviation) * 100

    content_rate = len(content_logs) / len(log)

    gains = [float(e.get("mastery_gain", 0.0)) for e in content_logs if e.get("mastery_gain") is not None]
    post_content_gain = float(np.mean(gains)) if gains else 0.0

    return EpisodeMetrics(
        seed=seed,
        cumulative_reward=cumulative_reward,
        final_mastery=final_mastery,
        time_to_mastery=time_to_mastery,
        question_accuracy=accuracy,
        blueprint_adherence=adherence,
        content_rate=content_rate,
        post_content_gain=post_content_gain,
    )


def aggregate_results(results: List[EpisodeMetrics]) -> Dict[str, float]:
    rewards = np.array([r.cumulative_reward for r in results], dtype=float)
    ttm_values = np.array([r.time_to_mastery for r in results if r.time_to_mastery is not None], dtype=float)

    summary = {
        "mean_reward": float(rewards.mean()),
        "reward_sd": float(rewards.std()),
        "policy_stability": float(rewards.std()),  # alias for clarity
        "mean_final_mastery": float(np.mean([r.final_mastery for r in results])),
        "mean_question_accuracy": float(np.mean([r.question_accuracy for r in results])),
        "mean_blueprint_adherence": float(np.mean([r.blueprint_adherence for r in results])),
        "mean_content_rate": float(np.mean([r.content_rate for r in results])),
        "mean_post_content_gain": float(np.mean([r.post_content_gain for r in results])),
    }

    if ttm_values.size > 0:
        summary.update(
            {
                "mean_time_to_mastery": float(ttm_values.mean()),
                "time_to_mastery_sd": float(ttm_values.std()),
            }
        )
    else:
        summary["mean_time_to_mastery"] = None
        summary["time_to_mastery_sd"] = None

    return summary


def save_results(results: List[EpisodeMetrics], summary: Dict[str, float], cfg: BaselineConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = cfg.output_dir / "episode_metrics.csv"
    with detail_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "cumulative_reward",
                "final_mastery",
                "time_to_mastery",
                "question_accuracy",
                "blueprint_adherence",
                "content_rate",
                "post_content_gain",
                "wall_clock_ms",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.seed,
                    r.cumulative_reward,
                    r.final_mastery,
                    r.time_to_mastery if r.time_to_mastery is not None else "",
                    r.question_accuracy,
                    r.blueprint_adherence,
                    r.content_rate,
                    r.post_content_gain,
                    r.wall_clock_ms if r.wall_clock_ms is not None else "",
                ]
            )

    summary_path = cfg.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))


def parse_args() -> BaselineConfig:
    parser = argparse.ArgumentParser(description="Run rule-based baseline controller")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19",
        help="Comma-separated random seeds (spec recommends 20 paired seeds)",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/baseline", help="Directory to store metrics")
    parser.add_argument("--save-step-logs", action="store_true", help="Save per-step logs per seed to outputs/baseline/steps")
    parser.add_argument("--include-timing", action="store_true", help="Record wall-clock time per episode")
    parser.add_argument("--total-steps", type=int, default=None, help="Total environment steps per seed (run multiple episodes until reached)")
    args = parser.parse_args()

    seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    return BaselineConfig(
        seeds=seed_list,
        output_dir=Path(args.output_dir),
        include_timing=bool(args.include_timing),
        save_step_logs=bool(args.save_step_logs),
        total_steps=args.total_steps,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    sim_config = globals().get("SIMULATOR_CONFIG", DEFAULT_SIMULATOR_CONFIG)
    env = AdaptiveLearningEnv(config=sim_config)
    controller = RuleBasedController(cfg)

    episode_results: List[EpisodeMetrics] = []
    for seed in cfg.seeds:
        steps_dir = cfg.output_dir / "steps"
        if cfg.save_step_logs:
            steps_dir.mkdir(parents=True, exist_ok=True)
            step_path = steps_dir / f"seed_{seed}.csv"
            # Write header fresh for each seed run
            with step_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "global_step",
                    "step",
                    "action_id",
                    "action_type",
                    "difficulty",
                    "modality",
                    "reward",
                    "correct",
                    "mastery_gain",
                    "mastery_mean",
                ])

        global_step = 0
        episode_index = 0
        # Run episodes until total_steps reached or run single episode if not specified
        while True:
            start_t = time.perf_counter()
            log, metrics = run_episode(env, controller, seed + episode_index, cfg)
            end_t = time.perf_counter()
            if cfg.include_timing:
                metrics.wall_clock_ms = (end_t - start_t) * 1000.0
            episode_results.append(metrics)

            if cfg.save_step_logs:
                step_path = steps_dir / f"seed_{seed}.csv"
                with step_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    for e in log:
                        global_step += 1
                        writer.writerow([
                            episode_index,
                            global_step,
                            e.get("step"),
                            e.get("action_id"),
                            e.get("action_type"),
                            e.get("difficulty"),
                            e.get("modality"),
                            e.get("reward"),
                            e.get("correct"),
                            e.get("mastery_gain"),
                            e.get("mastery_mean"),
                        ])

            episode_index += 1
            if cfg.total_steps is None:
                break
            if global_step >= cfg.total_steps:
                break

    summary = aggregate_results(episode_results)
    save_results(episode_results, summary, cfg)

    print("Baseline run complete. Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
