"""Adaptive learning environment simulator per spec_simulator.md.

Pure-Python implementation (no third-party deps) so it runs in constrained
environments. Uses only stdlib math/random/statistics.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Configuration aligned with spec_simulator.md
SIMULATOR_CONFIG: Dict = {
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
class Discrete:
    n: int
    rng: random.Random

    def sample(self) -> int:
        return int(self.rng.randrange(0, self.n))


@dataclass
class Box:
    low: float
    high: float
    shape: Tuple[int, ...]
    rng: random.Random

    def sample(self) -> List[float]:
        return [self.rng.uniform(self.low, self.high) for _ in range(int(math.prod(self.shape)))]


DIFFICULTY_MAP = {0: "easy", 1: "medium", 2: "hard"}
MODALITY_MAP = {0: "video", 1: "PPT", 2: "text", 3: "blog", 4: "article", 5: "handout"}


class AdaptiveLearningEnv:
    def __init__(self, config: Dict | None = None):
        self.config = config or SIMULATOR_CONFIG
        self.num_los = int(self.config.get("num_los", 30))
        self.num_questions_per_lo = int(self.config.get("num_questions_per_lo", 20))
        self.num_contents_per_lo = int(self.config.get("num_contents_per_lo", 6))
        self.num_questions = self.num_los * self.num_questions_per_lo
        self.num_contents = self.num_los * self.num_contents_per_lo
        self.max_steps = int(self.config.get("max_episode_steps", 140))
        self.response_time_max = 120.0

        self.rng = random.Random(0)
        self.questions = self._load_questions()
        self.contents = self._load_contents()

        self.observation_space = Box(0.0, 1.0, (self.num_los + 2,), self.rng)
        self.action_space = Discrete(self.num_los * 3 + self.num_los * 6, self.rng)

        self.learner_state: Dict = {}
        self.episode_log: List[Dict] = []
        self.event_log: List[Dict] = []  # For offline/OPE: question_shown, answered, content_shown
        self.step_count = 0

    def reset(self, seed: int | None = None) -> List[float]:
        if seed is not None:
            self.rng = random.Random(seed)
            self.observation_space.rng = self.rng
            self.action_space.rng = self.rng

        self.learner_state = self._initialize_learner()
        self.step_count = 0
        self.episode_log = []
        self.event_log = []
        return self._get_observation()

    def step(self, action: int) -> Tuple[List[float], float, bool, Dict]:
        action_dict = self._decode_action(action)
        if action_dict["type"] == "question":
            result = self._execute_question(action_dict)
        else:
            result = self._execute_content(action_dict)

        reward = float(self._compute_reward(result))
        self.step_count += 1
        done, reason = self._is_terminal()
        obs = self._get_observation()

        info = {
            "result": result,
            "termination_reason": reason,
            "step": self.step_count,
            "mean_mastery": float(statistics.mean(self.learner_state["mastery"])),
        }

        self.episode_log.append(
            {
                "step": self.step_count,
                "action": action,
                "reward": reward,
                "done": done,
                **info,
            }
        )

        return obs, reward, done, info

    def get_episode_metrics(self) -> Dict:
        if not self.episode_log:
            return {}
        return {
            "total_steps": self.step_count,
            "final_mastery": float(statistics.mean(self.learner_state["mastery"])),
            "cumulative_reward": float(sum(e["reward"] for e in self.episode_log)),
            "question_accuracy": self._compute_accuracy(),
            "content_count": self._count_content_actions(),
            "blueprint_adherence": self._compute_blueprint_adherence(),
        }

    def _initialize_learner(self) -> Dict:
        mastery = [self.rng.betavariate(2, 5) for _ in range(self.num_los)]
        return {
            "mastery": mastery,
            "ability": float(self.rng.gauss(0, 1)),
            "frustration": 0.0,
            "response_time": 0.0,
            "fail_streak": 0,
            "engagement": 1.0,
            "last_response_time": 0.0,
        }

    def _load_questions(self) -> Dict[Tuple[int, int], List[Dict]]:
        ranges = self.config.get("irt", {})
        diff_ranges = ranges.get("difficulty_ranges", {})
        disc_ranges = ranges.get("discrimination_ranges", {})
        guess_range = ranges.get("guessing_range", (0.1, 0.25))

        questions: Dict[Tuple[int, int], List[Dict]] = {}
        q_id = 0
        for lo in range(self.num_los):
            for diff_idx in range(3):
                questions[(lo, diff_idx)] = []
            for i in range(self.num_questions_per_lo):
                diff_idx = i % 3
                diff_key = DIFFICULTY_MAP[diff_idx]
                b_low, b_high = diff_ranges.get(diff_key, (-2.0, 0.5))
                a_low, a_high = disc_ranges.get(diff_key, (0.5, 1.5))
                b = float(self.rng.uniform(b_low, b_high))
                a = float(self.rng.uniform(a_low, a_high))
                c = float(self.rng.uniform(guess_range[0], guess_range[1]))
                questions[(lo, diff_idx)].append(
                    {
                        "id": q_id,
                        "learning_outcome": lo,
                        "difficulty": diff_key,
                        "difficulty_idx": diff_idx,
                        "irt_a": a,
                        "irt_b": b,
                        "irt_c": c,
                        "response_time_mean": 30.0,
                        "response_time_std": 10.0,
                    }
                )
                q_id += 1
        return questions

    def _load_contents(self) -> Dict[Tuple[int, int], Dict]:
        eff_ranges = self.config.get("content", {}).get("effectiveness_by_modality", {})
        contents: Dict[Tuple[int, int], Dict] = {}
        c_id = 0
        for lo in range(self.num_los):
            for modality_idx in range(6):
                modality = MODALITY_MAP[modality_idx]
                eff_low, eff_high = eff_ranges.get(modality, (0.05, 0.1))
                effectiveness = float(self.rng.uniform(eff_low, eff_high))
                engagement_impact = self._engagement_impact(modality_idx)
                contents[(lo, modality_idx)] = {
                    "id": c_id,
                    "learning_outcome": lo,
                    "modality": modality,
                    "modality_idx": modality_idx,
                    "duration": float(self.rng.uniform(5.0, 30.0)),
                    "effectiveness": effectiveness,
                    "engagement_impact": engagement_impact,
                }
                c_id += 1
        return contents

    def _decode_action(self, action_id: int) -> Dict:
        if action_id < 90:
            lo_index = action_id // 3
            difficulty = action_id % 3
            return {"type": "question", "lo": lo_index, "difficulty": difficulty}
        content_id = action_id - 90
        lo_index = content_id // 6
        modality = content_id % 6
        return {"type": "content", "lo": lo_index, "modality": modality}

    def _execute_question(self, action_dict: Dict) -> Dict:
        lo = action_dict["lo"]
        difficulty_idx = action_dict["difficulty"]
        question = self._get_question(lo, difficulty_idx)

        self._record_event(
            "question_shown",
            {
                "step": self.step_count + 1,
                "lo": lo,
                "difficulty": difficulty_idx,
                "question_id": question["id"],
            },
        )
        correct = self._sample_correctness(question)

        pre_mastery = float(self.learner_state["mastery"][lo])
        if correct:
            gain = 0.05 * (1 - pre_mastery)
            self.learner_state["mastery"][lo] = min(1.0, pre_mastery + gain)
            self.learner_state["fail_streak"] = 0
            self.learner_state["ability"] += 0.02
        else:
            gain = 0.0
            self.learner_state["fail_streak"] += 1

        self._update_frustration(correct, difficulty_idx, pre_mastery)

        response_time = self._sample_response_time(question)
        self.learner_state["last_response_time"] = response_time

        self._record_event(
            "answered",
            {
                "step": self.step_count + 1,
                "lo": lo,
                "difficulty": difficulty_idx,
                "question_id": question["id"],
                "correct": correct,
                "response_time": response_time,
            },
        )

        return {
            "type": "question",
            "lo": lo,
            "difficulty": difficulty_idx,
            "correct": correct,
            "mastery_gain": gain,
            "frustration": float(self.learner_state["frustration"]),
            "response_time": response_time,
        }

    def _execute_content(self, action_dict: Dict) -> Dict:
        lo = action_dict["lo"]
        modality_idx = action_dict["modality"]
        content = self._get_content(lo, modality_idx)

        self._record_event(
            "content_shown",
            {
                "step": self.step_count + 1,
                "lo": lo,
                "modality": modality_idx,
                "content_id": content["id"],
            },
        )

        pre_mastery = float(self.learner_state["mastery"][lo])
        gain = self._compute_content_gain(content, pre_mastery)
        self.learner_state["mastery"][lo] = min(1.0, pre_mastery + gain)
        post_mastery = float(self.learner_state["mastery"][lo])

        frustration_delta = content["engagement_impact"]
        new_f = self.learner_state["frustration"] + frustration_delta
        self.learner_state["frustration"] = float(min(1.0, max(0.0, new_f)))
        self.learner_state["fail_streak"] = 0

        return {
            "type": "content",
            "lo": lo,
            "modality": modality_idx,
            "mastery_gain": post_mastery - pre_mastery,
            "frustration_delta": frustration_delta,
            "frustration": float(self.learner_state["frustration"]),
        }

    def _record_event(self, event_type: str, payload: Dict) -> None:
        self.event_log.append({"event": event_type, **payload})

    def _get_question(self, lo: int, difficulty_idx: int) -> Dict:
        pool = self.questions.get((lo, difficulty_idx))
        if not pool:
            raise ValueError(f"No questions for LO {lo} difficulty {difficulty_idx}")
        return pool[int(self.rng.randrange(0, len(pool)))]

    def _get_content(self, lo: int, modality_idx: int) -> Dict:
        content = self.contents.get((lo, modality_idx))
        if content is None:
            raise ValueError(f"No content for LO {lo} modality {modality_idx}")
        return content

    def _sample_correctness(self, question: Dict) -> bool:
        theta = float(self.learner_state["ability"])
        a, b, c = question["irt_a"], question["irt_b"], question["irt_c"]
        prob_correct = c + (1 - c) / (1 + math.exp(-a * (theta - b)))
        return bool(self.rng.random() < prob_correct)

    def _update_frustration(self, correct: bool, difficulty_idx: int, mastery: float) -> None:
        if correct:
            self.learner_state["frustration"] = max(0.0, self.learner_state["frustration"] - 0.05)
            return

        self.learner_state["frustration"] = min(1.0, self.learner_state["frustration"] + 0.10)
        if difficulty_idx == 2 and mastery < 0.5:
            self.learner_state["frustration"] = min(1.0, self.learner_state["frustration"] + 0.05)

    def _sample_response_time(self, question: Dict) -> float:
        mean = question.get("response_time_mean", 30.0)
        std = question.get("response_time_std", 10.0)
        response_time = float(self.rng.gauss(mean, std))
        return max(5.0, response_time)

    def _compute_content_gain(self, content: Dict, pre_mastery: float) -> float:
        base_gain = content.get("effectiveness", 0.08)
        effective_gain = base_gain * (1 - pre_mastery)
        frustration_penalty = self.learner_state.get("frustration", 0.0) * 0.5
        effective_gain *= (1 - frustration_penalty)
        noise = float(self.rng.gauss(0.0, 0.02))
        return max(0.0, effective_gain + noise)

    def _compute_reward(self, action_result: Dict) -> float:
        weights = self.config.get("reward_weights", {})
        correctness_w = float(weights.get("correctness", 1.0))
        mastery_w = float(weights.get("mastery_gain", 0.5))
        frustration_w = float(weights.get("frustration_penalty", 0.3))
        post_content_w = float(weights.get("post_content_gain", 2.0))

        reward = 0.0
        if action_result["type"] == "question":
            if action_result.get("correct"):
                reward += correctness_w
            reward += mastery_w * float(action_result.get("mastery_gain", 0.0))
            reward -= frustration_w * float(self.learner_state.get("frustration", 0.0))
        else:
            post_gain = float(action_result.get("mastery_gain", 0.0))
            reward += post_content_w * post_gain
            engagement_delta = -float(action_result.get("frustration_delta", 0.0))
            reward += 0.5 * engagement_delta
        return reward

    def _is_terminal(self) -> Tuple[bool, str | None]:
        mean_mastery = float(statistics.mean(self.learner_state["mastery"]))
        if mean_mastery >= self.config.get("termination", {}).get("mastery_threshold", 0.8):
            return True, "mastery_achieved"
        if self.step_count >= self.max_steps:
            return True, "step_limit"
        if self.learner_state.get("frustration", 0.0) >= self.config.get("termination", {}).get("max_frustration", 0.95):
            return True, "critical_frustration"
        return False, None

    def _get_observation(self) -> List[float]:
        return list(self.learner_state["mastery"]) + [
            float(self.learner_state.get("frustration", 0.0)),
            self._normalize_response_time(self.learner_state.get("last_response_time", 0.0)),
        ]

    def _normalize_response_time(self, response_time: float) -> float:
        ratio = response_time / self.response_time_max if self.response_time_max else 0.0
        return float(min(1.0, max(0.0, ratio)))

    def _compute_accuracy(self) -> float:
        question_events = [e for e in self.episode_log if e.get("result", {}).get("type") == "question"]
        if not question_events:
            return 0.0
        correct = sum(1 for e in question_events if e["result"].get("correct"))
        return correct / len(question_events)

    def _count_content_actions(self) -> int:
        return sum(1 for e in self.episode_log if e.get("result", {}).get("type") == "content")

    def _compute_blueprint_adherence(self) -> float:
        question_events = [e for e in self.episode_log if e.get("result", {}).get("type") == "question"]
        if not question_events:
            return 100.0
        counts = {"easy": 0, "medium": 0, "hard": 0}
        for e in question_events:
            diff_idx = int(e["result"].get("difficulty", 0))
            counts[DIFFICULTY_MAP.get(diff_idx, "easy")] += 1
        total = sum(counts.values())
        if total == 0:
            return 100.0
        actual = {k: v / total for k, v in counts.items()}
        target = {"easy": 0.20, "medium": 0.60, "hard": 0.20}
        deviation = sum(abs(actual[d] - target[d]) for d in target) / 3
        return (1.0 - deviation) * 100

    def _engagement_impact(self, modality_idx: int) -> float:
        if modality_idx == 0:
            return -0.08
        if modality_idx == 1:
            return -0.05
        if modality_idx == 2:
            return 0.02
        if modality_idx == 3:
            return -0.03
        if modality_idx == 4:
            return 0.0
        return 0.05



def generate_dataset(config: Dict | None = None, num_learners: int = 200, episodes_per_learner: int = 1, policy=None) -> List[Dict]:
    env = AdaptiveLearningEnv(config=config or SIMULATOR_CONFIG)
    dataset: List[Dict] = []

    for learner_id in range(num_learners):
        for episode in range(episodes_per_learner):
            obs = env.reset(seed=learner_id * episodes_per_learner + episode)
            done = False

            while not done:
                action = policy(obs) if policy is not None else env.action_space.sample()
                next_obs, reward, done, info = env.step(action)
                dataset.append(
                    {
                        "learner_id": learner_id,
                        "episode": episode,
                        "obs": obs,
                        "action": action,
                        "reward": reward,
                        "next_obs": next_obs,
                        "done": done,
                        "info": info,
                    }
                )
                obs = next_obs
    return dataset


def validate_simulator(config: Dict | None = None) -> None:
    env = AdaptiveLearningEnv(config=config or SIMULATOR_CONFIG)

    obs = env.reset()
    assert len(obs) == env.num_los + 2, "State shape mismatch"
    assert all(0.0 <= x <= 1.0 for x in obs), "State out of bounds"

    for action in range(env.action_space.n):
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, float), "Reward not float"
        assert isinstance(done, bool), "Done not bool"
        assert "mean_mastery" in info, "Missing mean_mastery in info"
        if done:
            env.reset()

    mastery_increase_test = AdaptiveLearningEnv(config=config or SIMULATOR_CONFIG)
    o = mastery_increase_test.reset(seed=123)
    lo0 = 0
    pre = float(o[lo0])
    _, _, _, info = mastery_increase_test.step(lo0 * 3)  # easy question
    post = float(info["mean_mastery"])
    assert post >= pre, "Mastery did not increase on easy question"

    print("\u2713 All validation checks passed")


__all__ = ["AdaptiveLearningEnv", "SIMULATOR_CONFIG", "generate_dataset", "validate_simulator"]
