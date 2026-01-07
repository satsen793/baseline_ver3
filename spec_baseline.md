# Rule-Based Baseline Controller Specification

## Overview

A deterministic pedagogical controller that mimics common adaptive-learning logic. This serves as a non-ML baseline for comparison against RL algorithms.

## Purpose

- Provides interpretable comparison baseline
- Represents traditional adaptive learning systems
- Uses mastery-band difficulty progression with remediation triggers
- No learning/optimization—purely rule-based logic

## Algorithm Structure

The rule-based heuristic makes decisions in three sequential steps:

### Step 1: Decide Intervention Type (Question vs Content)

**Input:** Current state s_t = [m₁ᵗ, ..., mₖᵗ, f_t, τ_t]

**Thresholds:**
- `f_max`: Maximum frustration threshold (default: 0.7)
- `τ_max`: Maximum response time threshold (default: 60 seconds)

**Rule:**
```
If f_t ≥ f_max OR τ_t ≥ τ_max:
    κ_t = C (recommend content)
Else:
    κ_t = Q (ask question)
```

**Rationale:** High frustration or slow response indicates learner needs remediation, not more assessment.

---

### Step 2: If Question (κ_t = Q) - Select LO and Difficulty

#### A) Choose Learning Outcome
Select the LO with lowest mastery:

```
ℓ_t = argmin_i mᵢᵗ
```

**Rationale:** Focus on weakest area first.

#### B) Choose Difficulty Based on Mastery Bands

Use blueprint-aligned mastery bands:

```
If mₗₜᵗ < 0.4:
    difficulty = Easy
Elif 0.4 ≤ mₗₜᵗ < 0.7:
    difficulty = Medium
Else (mₗₜᵗ ≥ 0.7):
    difficulty = Hard
```

**Mastery Bands:**
- **Easy:** [0.0, 0.4) - Beginner/struggling
- **Medium:** [0.4, 0.7) - Developing competence
- **Hard:** [0.7, 1.0] - High competence

**Blueprint Adherence:** This naturally produces a distribution close to 20-60-20 if mastery is uniformly distributed across learners.

---

### Step 3: If Content (κ_t = C) - Select Modality and Topic

#### A) Choose Topic
Same as question selection—lowest mastery LO:

```
topic_t = ℓ_t = argmin_i mᵢᵗ
```

#### B) Choose Modality Based on Frustration

Allowed modalities for the baseline: **handout**, **text**, **PPT**, **video** (no blog/article options are used by the baseline).

```
If f_t is high (f_t > 0.6):
    modality_t = handout or text
Else:
    modality_t = PPT or video
```

**Rationale:**
- High frustration → brief/low-load content (handout/text)
- Lower frustration → richer media (PPT/video) that historically yield higher post-content gains

**Modality Selection Logic (Detailed):**
```python
if f_t > 0.6:
    modality = random.choice(["handout", "text"])
else:
    modality = random.choice(["PPT", "video"])
```

---

## Complete Algorithm Pseudocode

```
Algorithm: Rule-Based Baseline Controller
Input: State s_t = [m₁ᵗ, ..., mₖᵗ, f_t, τ_t]
Parameters: f_max=0.7, τ_max=60

1. Decide intervention type:
   If (f_t ≥ f_max) OR (τ_t ≥ τ_max):
       κ_t ← C
   Else:
       κ_t ← Q

2. If κ_t = Q (ask question):
   a) ℓ_t ← argmin_i mᵢᵗ
   b) If mₗₜᵗ < 0.4:
          difficulty ← Easy
      Elif 0.4 ≤ mₗₜᵗ < 0.7:
          difficulty ← Medium
      Else:
          difficulty ← Hard
   c) Return question_action(ℓ_t, difficulty)

3. If κ_t = C (recommend content):
   a) topic ← argmin_i mᵢᵗ
   b) If f_t > 0.6:
          modality ← random.choice(["handout", "text"])
      Else:
          modality ← random.choice(["video", "PPT"])
   c) Return content_action(topic, modality)
```

---

## Implementation Details

### Input Requirements
- **State Vector:** numpy array of shape (K+2,) where K=30
  - Indices 0..29: mastery values mᵢᵗ
  - Index 30: frustration f_t
  - Index 31: response time τ_t (normalized)

### Output Format
- Return flattened action index compatible with environment's action space
- Map (gate, LO, difficulty/modality) → single integer action_id

### Hyperparameters

```python
HYPERPARAMETERS = {
    'f_max': 0.7,           # Frustration threshold
    'tau_max': 60.0,        # Response time threshold (seconds)
    'mastery_easy': 0.4,    # Upper bound for Easy difficulty
    'mastery_medium': 0.7,  # Upper bound for Medium difficulty
    'high_frustration': 0.6 # Threshold for content modality switch
}
```

### Action Space Mapping

**Question Actions:**
```
action_id = LO_index * 3 + difficulty_level
# difficulty_level: 0=Easy, 1=Medium, 2=Hard
# Range: 0 to 89 (30 LOs × 3 difficulties)
```

**Content Actions:**
```
action_id = 90 + (LO_index * 6 + modality_index)
# modality_index: 0=video, 1=PPT, 2=text, 3=blog, 4=article, 5=handout
# Range: 90 to 269 (30 LOs × 6 modalities)
```

---

## Expected Behavior

### Early Episode (Low Mastery)
- Most questions will be **Easy** (mastery < 0.4)
- Frequent content recommendations as learners struggle
- Blueprint: Heavy Easy bias initially

### Mid Episode (Moderate Mastery)
- Shift to **Medium** difficulty questions (0.4 ≤ mastery < 0.7)
- Fewer content interventions
- Blueprint: Approaches 20-60-20 distribution

### Late Episode (High Mastery)
- More **Hard** questions (mastery ≥ 0.7)
- Rare content recommendations
- Blueprint: Heavy Hard bias

---

## Evaluation Metrics (Same as RL Agents)

1. **Time-to-Mastery:** Average steps to reach 0.8 mastery per LO
2. **Post-Content Gain:** Improvement after content (if any content recommended)
3. **Cumulative Reward:** Total reward per session
4. **Blueprint Adherence:** Measure deviation from 20-60-20
5. **Stability:** Should be deterministic (zero variance for same seed)

---

## Statistical Testing

### Comparison Protocol
- Run on same S=20 seeds as RL algorithms
- Report mean ± SD (should be low SD due to determinism)
- Compare against DQN, PPO, PETS, MBPO using:
  - Independent samples t-test (baseline vs each RL method)
  - Effect size (Cohen's d)

### Expected Results
- **Time-to-Mastery:** Likely slower than RL (no optimization)
- **Cumulative Reward:** Lower than optimized RL policies
- **Blueprint:** May deviate significantly (no enforcement mechanism)
- **Post-Content Gain:** Similar to RL if content selection rules are reasonable

---

## Implementation Template (Python)

```python
import numpy as np

class RuleBasedController:
    def __init__(self, f_max=0.7, tau_max=60.0, 
                 mastery_easy=0.4, mastery_medium=0.7,
                 high_frustration=0.6):
        self.f_max = f_max
        self.tau_max = tau_max
        self.mastery_easy = mastery_easy
        self.mastery_medium = mastery_medium
        self.high_frustration = high_frustration
        
        # Action space: 90 questions + 180 content = 270 total
        self.num_los = 30
        self.num_difficulties = 3
        self.num_modalities = 6
    
    def select_action(self, state):
        """
        state: numpy array [m1, m2, ..., m30, frustration, response_time]
        returns: action_id (int)
        """
        masteries = state[:self.num_los]
        frustration = state[self.num_los]
        response_time = state[self.num_los + 1]
        
        # Step 1: Decide question vs content
        if frustration >= self.f_max or response_time >= self.tau_max:
            gate = 'content'
        else:
            gate = 'question'
        
        # Step 2a: Select lowest-mastery LO
        lo_index = np.argmin(masteries)
        lo_mastery = masteries[lo_index]
        
        if gate == 'question':
            # Step 2b: Choose difficulty
            if lo_mastery < self.mastery_easy:
                difficulty = 0  # Easy
            elif lo_mastery < self.mastery_medium:
                difficulty = 1  # Medium
            else:
                difficulty = 2  # Hard
            
            action_id = lo_index * self.num_difficulties + difficulty
            
        else:  # content
            # Step 3b: Choose modality
            if frustration > self.high_frustration:
                modality = np.random.choice([2, 5])  # text=2, handout=5
            else:
                modality = np.random.choice([0, 1])  # video=0, PPT=1
            
            action_id = 90 + (lo_index * self.num_modalities + modality)
        
        return action_id
    
    def reset(self):
        """No internal state to reset (stateless controller)"""
        pass
```

---

## Advantages of Rule-Based Baseline

1. **Interpretability:** Every decision is traceable
2. **No Training Required:** Immediate deployment
3. **Deterministic:** Consistent behavior across runs (given same seed)
4. **Fast:** Zero compute overhead
5. **Domain-Grounded:** Based on established pedagogical principles

## Limitations

1. **No Optimization:** Cannot improve from experience
2. **Fixed Thresholds:** May be suboptimal for diverse learner populations
3. **No Long-Term Planning:** Myopic (only considers current state)
4. **No Credit Assignment:** Cannot learn which content types work best
5. **Blueprint Drift:** No mechanism to enforce difficulty distribution

---

## Expected Use in Paper

**Purpose:** Demonstrate that model-free and model-based RL methods significantly outperform traditional rule-based approaches in:
- Sample efficiency (time-to-mastery)
- Reward optimization
- Adaptive blueprint adherence

**Reporting:**
> "The rule-based baseline achieved a mean time-to-mastery of XXX steps (±YY SD), with cumulative reward of ZZZ. In contrast, MBPO reduced time-to-mastery by 35% while improving blueprint adherence by 15 percentage points, demonstrating the value of learned policies over heuristic controllers."


