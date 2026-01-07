# Adaptive Mock Interview System - Overview & MDP Formulation

## Problem Context

This system addresses adaptive learning for mock interview preparation in a 6-month data science program. The goal is to intelligently select question difficulty and recommend learning content based on learner state using Reinforcement Learning.

## Core Problem Formulation

### MDP Definition

The adaptive mock-interview system is modeled as a finite-horizon Markov Decision Process (MDP):

```
M = ⟨S, A, P, R, γ⟩
```

Where:
- **S**: State space (learner knowledge and engagement)
- **A**: Action space (pedagogical interventions)
- **P(s'|s,a)**: Transition probability (knowledge update dynamics)
- **R(s,a)**: Reward function (learning progress + engagement)
- **γ ∈ [0,1]**: Discount factor

### Objective

Learn a policy π_θ(a|s) that maximizes expected cumulative reward:

```
J(π_θ) = E_π_θ[Σ(t=0 to T) γ^t * r_t]
```

## State Space (s_t)

The learner's current state at timestep t:

```
s_t = [m₁ᵗ, m₂ᵗ, ..., mₖᵗ, f_t, τ_t]
```

**Components:**
- `mᵢᵗ ∈ [0,1]`: Mastery over Learning Outcome i (K learning outcomes total)
- `f_t`: Frustration/engagement level
- `τ_t`: Recent response time

**Markov Property:** Engagement variables (f_t, τ_t) are explicitly included so the simulator can update them from (s_t, a_t), preserving the Markov assumption.

## Action Space (a_t)

### Gated Discrete Action Representation

The agent chooses either a question OR content recommendation:

```
a_t = (κ_t, a_t^(κ))
```

Where:
- `κ_t ∈ {Q, C}` - gate variable (Q=question, C=content)
- If κ_t = Q: `a_t^(κ) ∈ A_Q` (question actions)
- If κ_t = C: `a_t^(κ) ∈ A_C` (content actions)

### Unified Action Space

```
A = {(Q,a): a ∈ A_Q} ∪ {(C,a): a ∈ A_C}
```

**Implementation Note:** This can be flattened into a single categorical action index compatible with DQN and PPO. Invalid actions are masked (zero probability or excluded from argmax).

### Question Actions (A_Q)
- Learning Outcome (LO) selection (30 LOs)
- Difficulty level: {Easy, Medium, Hard}
- Total: ~90 question action combinations

### Content Actions (A_C)
- Learning Outcome selection (30 LOs)
- Modality: {video, PPT, text, blog, article, handout} (6 types)
- Total: ~180 content action combinations

## Transition Dynamics (P)

Mastery evolution given an action:

```
mᵢᵗ⁺¹ = mᵢᵗ + η · φ(a_t, r_t, content_type) + ξᵢᵗ
```

Where:
- `η`: Learning rate
- `φ(·)`: Content efficacy function (from empirical post-content gain stats)
- `ξᵢᵗ`: Stochastic noise in learning

**Simulator Stochasticity:** Captures randomness in:
- Correctness outcomes
- Mastery updates
- Engagement dynamics

**IRT-Based Performance:** Success probability uses logistic IRT model:

```
P(correct) = 1 / (1 + exp[-aᵢ(θⱼ - bᵢ)])
```

Where:
- `aᵢ`: Question discrimination
- `bᵢ`: Question difficulty
- `θⱼ`: Learner's latent ability at step t

## Reward Function (r_t)

Encodes pedagogical desirability:

```
r_t = α · correct_t + β · Δm_t - γ_f · f_t
```

Where:
- `correct_t ∈ {0,1}`: Whether learner answered correctly
- `Δm_t`: Mastery increment
- `f_t`: Frustration penalty
- `(α, β, γ_f)`: Weights balancing accuracy, improvement, and well-being

**Typical Values:** α=1.0, β=0.5, γ_f=0.3

## Episode Structure

### Episode Definition
One episode = one mock-interview session of at most T decisions (typically T=80-140 items)

### Termination Conditions
1. Mastery criterion reached (e.g., target LO mastery > 0.8)
2. Step budget T exhausted

### Terminal State Handling
Terminal transitions have next-state value set to zero in bootstrapped targets.

## Dataset & Simulator Specifications

### Simulated Environment
- **Learners:** 200 simulated learners
- **Learning Outcomes:** 30 LOs tagged with Bloom's taxonomy
- **Questions:** 600 questions parameterized by IRT (a, b, c factors)
- **Content:** ~180 learning materials across 6 modalities
- **Session Length:** 80-140 items per learner
- **Total Interactions:** >50,000 logged events

### Event Types
- `question_shown`
- `answered` (with correctness)
- `content_shown`
- `content_completed`

### Initial Conditions
- Learner proficiency drawn from beta distribution (heterogeneous population)
- Mastery levels initialized based on ability distribution

## Blueprint Constraints

### Difficulty Distribution Target
Questions should maintain a 20%-60%-20% distribution:
- Easy: 20%
- Medium: 60%
- Hard: 20%

**Implementation:** Use masking, reward penalties, or constrained action sampling to enforce blueprint adherence.

## Evaluation Metrics

### Primary Metrics
1. **Time-to-Mastery**: Average steps to reach 0.8 mastery per LO
2. **Post-Content Gain**: Change in correctness after content recommendation
3. **Cumulative Reward**: Total reward per learner session
4. **Blueprint Adherence**: Deviation from 20-60-20 difficulty distribution
5. **Policy Stability**: Reward variance across episodes

### Statistical Testing
- **Seeds:** S=5-20 independent random seeds (paired design)
- **Reporting:** mean ± SD and 95% confidence intervals
- **Tests:** Paired t-test (or Wilcoxon if normality fails)
- **Effect Size:** Cohen's d or Cliff's δ
- **Bootstrap:** 1,000 iterations for CI estimation

## Common Hyperparameters

### Shared Across All Algorithms
- `discount_factor (γ)`: 0.99
- `max_episode_length (T)`: 140
- `learning_rate`: 3e-4 (typical starting point)
- `batch_size`: 64-256
- `random_seeds`: 5-20 seeds for statistical validation

### Reward Shaping Weights
- `α (correctness)`: 1.0
- `β (mastery_gain)`: 0.5
- `γ_f (frustration_penalty)`: 0.3

## Conceptual Flow Diagram

```
Learner State (s_t)
    ↓
RL Agent (π_θ) observes state
    ↓
Selects Action (a_t): question or content

    ## Educational Relevance
    - The MDP framing ties actions (assessment vs. remediation) directly to learner mastery and engagement, enabling adaptive pacing and content selection.
    - Model-based methods (PETS, MBPO) plan or augment with learned dynamics to gain sample efficiency—important when learner interactions are costly and exploration must be safe.
    - Discrete action handling (categorical/gated) makes RL compatible with question/content decisions without continuous control hacks.

    ## Conceptual Decision Loop (textual)
    1. Observe learner state \(s_t\): mastery vector, frustration, response time.
    2. Policy selects action \(a_t\): question (LO, difficulty) or content (LO, modality).
    3. Environment presents item/content, returns reward \(r_t\) and updates state to \(s_{t+1}\) via dynamics (true or learned).
    4. Repeat until mastery threshold or step budget reached; policy updates from collected transitions.
    ↓
Environment/Simulator
    ↓
Computes reward (r_t) + updates state (s_{t+1})
    ↓
Loop continues until termination
```

## Implementation Notes

### State Encoding
- Mastery vector: float32 array of shape (K,) where K=30
- Frustration: single float in [0,1]
- Response time: normalized float

### Action Encoding
- Flatten gated actions to single discrete index
- Use one-hot encoding for neural network input
- Mask invalid actions during action selection

### Simulator Interface
```python
class AdaptiveLearningEnv:
    def reset() -> state
    def step(action) -> (next_state, reward, done, info)
    def get_state() -> state_vector
    def is_terminal() -> bool
```

---

**Next Steps:** See individual algorithm specification files for:
- Rule-Based Baseline
- DQN with Prioritized Replay
- PPO (Discrete)
- PETS (with factorized categorical CEM)
- MBPO (with factorized discrete SAC)
