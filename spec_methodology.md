# Methodology and System Overview

## Components
- **Learner Environment Simulator:** Implements IRT-based question correctness, mastery updates, frustration/response-time dynamics, blueprint difficulty mix, and content effects by modality.
- **RL Agent:** Supports rule-based baseline, DQN (PER), PPO, PETS (categorical CEM MPC), and MBPO (factorized discrete SAC). Uses the MDP/state/action/reward defined in spec_overview.md and simulator details in spec_simulator.md.
- **Instructor/Analytics Dashboard (conceptual):** Consumes logs/metrics (time-to-mastery, reward, blueprint adherence, post-content gains, stability).

## Content Recommendation Trigger
- **Fail-streak rule:** After three consecutive incorrect answers (fail-streak >= 3), trigger content recommendation instead of another question. This is the default remediation gate to reduce frustration and support mastery.

## Content and LO Metadata
- LOs: tagged with Bloom levels and prerequisites.
- Content items: modality, duration, and reading complexity recorded; modality-specific effectiveness and frustration impacts are in spec_simulator.md.

## Session Structure
- One environment step = one tutoring decision (question or content).
- Episode: mock-interview session up to T steps (typically 80–140) or until mastery threshold reached.

## Data/Assets (from simulator)
- ~30 Learning Outcomes (LOs) with Bloom tags and prerequisites.
- ~600 questions with IRT (a, b, c); ~180 content items across modalities (video, PPT, text, blog, article, handout) with duration/effectiveness/frustration impacts.

## Algorithms Compared
- Rule-based heuristic (gated question/content, mastery-band difficulty, modality by frustration).
- Model-free: DQN with PER; PPO (categorical policy).
- Model-based: PETS (categorical CEM MPC over horizon H); MBPO (short model rollouts K mixed with real replay).

## Reporting
- Learning curves, time-to-mastery, post-content gains by modality, blueprint adherence, seed stability, and compute–reward trade-offs.

## Rationale for Simulation-Based Evaluation
- Real data collection is slow, privacy constrained, and costly for early-stage RL; simulator enables controlled, reproducible experiments across seeds.
- Simulator captures mastery progression, fatigue, and difficulty adaptation; aligns state/reward features (accuracy, response time, engagement) with LMS-observable signals for transfer to live settings.

## PETS/MBPO Episodes and Horizons
- One real environment step = one tutoring decision. Episodes end on mastery threshold or step budget T.
- PETS: plans a length-H discrete sequence each real step and executes only the first action before replanning.
- MBPO: generates short learned-model rollouts of length K from real states; K is independent of episode length T.

## Discrete Adaptation Highlights (PETS/MBPO)
- Action space: multi-discrete (difficulty, LO, modality/gate), joint or factorized categorical.
- PETS: categorical CEM planner over logits; elite-frequency updates; executes first action of best sequence.
- MBPO: discrete actor-critic (categorical or factorized); mixes real/model rollouts; entropy regularization retained.
