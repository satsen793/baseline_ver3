# Adaptive Learning Environment Simulator - Full Specification

## Overview

This simulator models learner interactions with questions and learning content in an adaptive mock-interview system. It implements realistic cognitive dynamics using Item Response Theory (IRT) and stochastic mastery updates.

## Purpose

- Provide controlled, reproducible environment for RL algorithm evaluation
- Model learner knowledge evolution, engagement, and fatigue
- Support both question-based assessment and content-based learning
- Enable fair comparison across algorithms with identical conditions

---

## Core Components

### 1. Learner Model

#### Initial State
Each learner starts with:
```python
learner_state = {
    'mastery': beta.rvs(a=2, b=5, size=30),  # Per-LO mastery in [0,1]
    'ability': np.random.normal(0, 1),        # Latent IRT ability θ
    'frustration': 0.0,                       # Initial frustration level
    'response_time': 0.0,                     # Recent response time (normalized)
    'fail_streak': 0,                         # Consecutive incorrect answers
    'engagement': 1.0                         # Engagement level
}
```

**Mastery Initialization:**
- Beta(2,5) distribution creates heterogeneous starting levels
- Mean ≈ 0.29, ensuring most learners start low-to-moderate mastery
- Variance ensures population diversity

### 2. Question Bank

#### Question Properties
```python
question = {
    'id': unique_id,
    'learning_outcome': LO_index (0-29),
    'difficulty': 'Easy' | 'Medium' | 'Hard',
    'irt_a': discrimination (0.5-2.0),
    'irt_b': difficulty (-2.0 to 2.0),
    'irt_c': guessing (0.1-0.25),
    'response_time_mean': 30,  # seconds
    'response_time_std': 10
}
```

**IRT Model:**
3-Parameter Logistic (3PL):
```
P(correct | θ, a, b, c) = c + (1-c) / (1 + exp(-a(θ - b)))
```

Where:
- `θ`: Learner ability (latent)
- `a`: Item discrimination
- `b`: Item difficulty
- `c`: Guessing parameter

**Difficulty Mapping:**
- Easy: b ∈ [-2.0, -0.5], a ∈ [0.5, 1.0]
- Medium: b ∈ [-0.5, 0.5], a ∈ [1.0, 1.5]
- Hard: b ∈ [0.5, 2.0], a ∈ [1.5, 2.0]

### 3. Content Repository

#### Content Properties
```python
content = {
    'id': unique_id,
    'learning_outcome': LO_index (0-29),
    'modality': 'video' | 'PPT' | 'text' | 'blog' | 'article' | 'handout',
    'duration': minutes (5-30),
    'effectiveness': base_gain (0.05-0.15),
    'engagement_impact': delta_frustration (-0.1 to 0.05)
}
```

**Modality Characteristics:**
| Modality | Duration | Base Gain | Frustration Impact |
|----------|----------|-----------|-------------------|
| Video | 15-25 min | 0.10-0.15 | -0.08 (engaging) |
| PPT | 10-20 min | 0.08-0.12 | -0.05 |
| Text | 5-10 min | 0.05-0.08 | +0.02 (dry) |
| Blog | 8-15 min | 0.07-0.10 | -0.03 |
| Article | 10-18 min | 0.06-0.09 | 0.00 |
| Handout | 5-12 min | 0.05-0.08 | +0.05 (boring) |

**Post-Content Gain:** Empirically observed improvements after content consumption

---

## State Representation

### State Vector (32 dimensions)

```python
state = np.concatenate([
    mastery_vector,      # (30,) - mastery per LO
    [frustration],       # (1,) - current frustration level
    [response_time_norm] # (1,) - normalized response time
])
```

**Normalization:**
- Mastery: Already in [0, 1]
- Frustration: Clipped to [0, 1]
- Response time: Normalized to [0, 1] using max_time=120s

---

## Action Space

### Unified Discrete Action Space (270 actions)

**Question Actions (0-89):**
```
action_id = LO_index * 3 + difficulty_level
# difficulty_level: 0=Easy, 1=Medium, 2=Hard
# Range: 0 to 89
```

**Content Actions (90-269):**
```
action_id = 90 + (LO_index * 6 + modality_index)
# modality_index: 0=video, 1=PPT, 2=text, 3=blog, 4=article, 5=handout
# Range: 90 to 269
```

### Action Decoding

```python
def decode_action(action_id):
    if action_id < 90:
        # Question action
        lo_index = action_id // 3
        difficulty = action_id % 3  # 0=Easy, 1=Medium, 2=Hard
        return {'type': 'question', 'lo': lo_index, 'difficulty': difficulty}
    else:
        # Content action
        content_id = action_id - 90
        lo_index = content_id // 6
        modality = content_id % 6
        return {'type': 'content', 'lo': lo_index, 'modality': modality}
```

---

## Transition Dynamics

### Question Interaction

When a question action is selected:

1. **Sample Correctness** (IRT-based):
```python
theta = learner['ability']
a, b, c = question['irt_a'], question['irt_b'], question['irt_c']

prob_correct = c + (1 - c) / (1 + np.exp(-a * (theta - b)))
correct = np.random.rand() < prob_correct
```

2. **Update Mastery** (if correct):
```python
if correct:
    current_mastery = learner['mastery'][lo_index]
    gain = 0.05 * (1 - current_mastery)  # Diminishing returns
    learner['mastery'][lo_index] += gain
    learner['fail_streak'] = 0
else:
    learner['fail_streak'] += 1
```

3. **Update Frustration**:
```python
if correct:
    learner['frustration'] = max(0, learner['frustration'] - 0.05)
else:
    learner['frustration'] = min(1, learner['frustration'] + 0.10)
    
    # Extra penalty for hard questions when mastery is low
    if difficulty == 'Hard' and current_mastery < 0.5:
        learner['frustration'] += 0.05
```

4. **Sample Response Time**:
```python
base_time = question['response_time_mean']
time_std = question['response_time_std']
response_time = np.random.normal(base_time, time_std)
response_time = max(5, response_time)  # Minimum 5 seconds
```

5. **Update Ability** (IRT latent variable):
```python
if correct:
    learner['ability'] += 0.02  # Gradual ability improvement
```

### Content Interaction

When a content action is selected:

1. **Pre-Content Mastery**:
```python
pre_mastery = learner['mastery'][lo_index]
```

2. **Content Effectiveness**:
```python
base_gain = content['effectiveness']

# Adjust for current mastery (diminishing returns)
effective_gain = base_gain * (1 - pre_mastery)

# Adjust for frustration (high frustration reduces learning)
frustration_penalty = learner['frustration'] * 0.5
effective_gain *= (1 - frustration_penalty)

# Stochastic variability
noise = np.random.normal(0, 0.02)
final_gain = max(0, effective_gain + noise)
```

3. **Update Mastery**:
```python
learner['mastery'][lo_index] = min(1.0, pre_mastery + final_gain)
post_mastery = learner['mastery'][lo_index]
```

4. **Update Frustration**:
```python
frustration_delta = content['engagement_impact']
learner['frustration'] = np.clip(
    learner['frustration'] + frustration_delta, 
    0, 1
)
```

5. **Reset Fail Streak**:
```python
learner['fail_streak'] = 0  # Content remediation resets streak
```

6. **Post-Content Gain** (for metrics):
```python
post_content_gain = post_mastery - pre_mastery
```

---

## Reward Function

### Components

```python
def compute_reward(action_result, learner_state):
    reward = 0
    
    if action_result['type'] == 'question':
        # Correctness bonus
        if action_result['correct']:
            reward += 1.0
        
        # Mastery gain bonus
        mastery_gain = action_result['mastery_gain']
        reward += 0.5 * mastery_gain
        
        # Frustration penalty
        reward -= 0.3 * learner_state['frustration']
        
    elif action_result['type'] == 'content':
        # Content effectiveness bonus
        post_content_gain = action_result['mastery_gain']
        reward += 2.0 * post_content_gain  # Higher weight for content
        
        # Engagement bonus (negative frustration delta)
        engagement_delta = -action_result['frustration_delta']
        reward += 0.5 * engagement_delta
    
    return reward
```

**Reward Shaping Rationale:**
- **Correctness (+1.0):** Direct assessment success
- **Mastery Gain (+0.5):** Encourages learning progress
- **Frustration (-0.3):** Discourages learner overload
- **Post-Content Gain (+2.0):** Strongly rewards effective remediation

---

## Episode Structure

### Episode Definition

- Event types recorded: `question_shown`, `answered`, `content_shown` (for offline/OPE use)
One episode = one mock-interview session

**Duration:** 80-140 steps (variable based on mastery progression)

### Termination Conditions

Episode terminates when:
1. **Mastery Threshold:** Mean mastery across all LOs ≥ 0.8
2. **Step Limit:** Reached max_steps (default: 140)
3. **Critical Frustration:** frustration ≥ 0.95 (learner gives up)

```python
def is_terminal(learner_state, step_count):
    mean_mastery = np.mean(learner_state['mastery'])
    
    if mean_mastery >= 0.8:
        return True, 'mastery_achieved'
    
    if step_count >= 140:
        return True, 'step_limit'
    
    if learner_state['frustration'] >= 0.95:
        return True, 'critical_frustration'
    
    return False, None
```

---

## Simulator Interface (OpenAI Gym Style)

### Class Structure

```python
class AdaptiveLearningEnv:
    def __init__(self, config):
        self.num_los = 30
        self.num_questions = 600
        self.num_contents = 180
        self.max_steps = 140
        
        # Load question bank and content repository
        self.questions = self._load_questions()
        self.contents = self._load_contents()
        
        # State/action spaces
        self.observation_space = Box(low=0, high=1, shape=(32,))
        self.action_space = Discrete(270)
        
        self.config = config
    
    def reset(self, seed=None):
        """Initialize a new learner and episode"""
        if seed is not None:
            np.random.seed(seed)
        
        self.learner_state = self._initialize_learner()
        self.step_count = 0
        self.episode_log = []
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one action and return (obs, reward, done, info)"""
        action_dict = self._decode_action(action)
        
        if action_dict['type'] == 'question':
            result = self._execute_question(action_dict)
        else:
            result = self._execute_content(action_dict)
        
        # Compute reward
        reward = self._compute_reward(result)
        
        # Update step count
        self.step_count += 1
        
        # Check termination
        done, reason = self._is_terminal()
        
        # Get next observation
        next_obs = self._get_observation()
        
        # Info dict
        info = {
            'result': result,
            'termination_reason': reason,
            'step': self.step_count,
            'mean_mastery': np.mean(self.learner_state['mastery'])
        }
        
        # Log event
        self.episode_log.append({
            'step': self.step_count,
            'action': action,
            'reward': reward,
            'done': done,
            **info
        })
        
        return next_obs, reward, done, info
    
    def _get_observation(self):
        """Return current state vector"""
        return np.concatenate([
            self.learner_state['mastery'],
            [self.learner_state['frustration']],
            [self._normalize_response_time(
                self.learner_state.get('last_response_time', 0)
            )]
        ])
    
    def get_episode_metrics(self):
        """Compute episode-level metrics"""
        if not self.episode_log:
            return {}
        
        return {
            'total_steps': self.step_count,
            'final_mastery': np.mean(self.learner_state['mastery']),
            'cumulative_reward': sum(e['reward'] for e in self.episode_log),
            'question_accuracy': self._compute_accuracy(),
            'content_count': self._count_content_actions(),
            'blueprint_adherence': self._compute_blueprint_adherence()
        }
```

### Key Methods

```python
def _execute_question(self, action_dict):
    """Execute a question interaction"""
    lo = action_dict['lo']
    difficulty = action_dict['difficulty']
    
    # Select question
    question = self._get_question(lo, difficulty)
    
    # IRT-based correctness
    correct = self._sample_correctness(question)
    
    # Update mastery if correct
    pre_mastery = self.learner_state['mastery'][lo]
    if correct:
        gain = 0.05 * (1 - pre_mastery)
        self.learner_state['mastery'][lo] += gain
        self.learner_state['fail_streak'] = 0
    else:
        gain = 0
        self.learner_state['fail_streak'] += 1
    
    # Update frustration
    self._update_frustration(correct, difficulty, pre_mastery)
    
    # Sample response time
    response_time = self._sample_response_time(question)
    self.learner_state['last_response_time'] = response_time
    
    return {
        'type': 'question',
        'lo': lo,
        'difficulty': difficulty,
        'correct': correct,
        'mastery_gain': gain,
        'frustration': self.learner_state['frustration'],
        'response_time': response_time
    }

def _execute_content(self, action_dict):
    """Execute a content interaction"""
    lo = action_dict['lo']
    modality = action_dict['modality']
    
    # Select content
    content = self._get_content(lo, modality)
    
    # Pre-content mastery
    pre_mastery = self.learner_state['mastery'][lo]
    
    # Content effectiveness (stochastic)
    gain = self._compute_content_gain(content, pre_mastery)
    
    # Update mastery
    self.learner_state['mastery'][lo] = min(1.0, pre_mastery + gain)
    post_mastery = self.learner_state['mastery'][lo]
    
    # Update frustration
    frustration_delta = content['engagement_impact']
    self.learner_state['frustration'] = np.clip(
        self.learner_state['frustration'] + frustration_delta,
        0, 1
    )
    
    # Reset fail streak
    self.learner_state['fail_streak'] = 0
    
    return {
        'type': 'content',
        'lo': lo,
        'modality': modality,
        'mastery_gain': post_mastery - pre_mastery,
        'frustration_delta': frustration_delta,
        'frustration': self.learner_state['frustration']
    }
```

---

## Data Generation for Experiments

### Population Simulation (200 Learners)

```python
def generate_dataset(num_learners=200, episodes_per_learner=1):
    env = AdaptiveLearningEnv(config)
    dataset = []
    
    for learner_id in range(num_learners):
        for episode in range(episodes_per_learner):
            obs = env.reset(seed=learner_id * episodes_per_learner + episode)
            done = False
            episode_data = []
            
            while not done:
                # Random or heuristic policy for data collection
                action = env.action_space.sample()
                next_obs, reward, done, info = env.step(action)
                
                episode_data.append({
                    'learner_id': learner_id,
                    'episode': episode,
                    'obs': obs,
                    'action': action,
                    'reward': reward,
                    'next_obs': next_obs,
                    'done': done,
                    'info': info
                })
                
                obs = next_obs
            
            dataset.extend(episode_data)
    
    return dataset
```

---

## Validation and Debugging

### Sanity Checks

```python
def validate_simulator():
    env = AdaptiveLearningEnv(config)
    
    # Test 1: State bounds
    obs = env.reset()
    assert obs.shape == (32,), "State shape mismatch"
    assert np.all(obs >= 0) and np.all(obs <= 1), "State out of bounds"
    
    # Test 2: Action validity
    for action in range(270):
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, float), "Reward not float"
        assert isinstance(done, bool), "Done not bool"
    
    # Test 3: Mastery progression
    # ... (verify mastery increases on correct answers)
    
    # Test 4: Episode termination
    # ... (verify termination conditions)
    
    print("✓ All validation checks passed")
```

---

## Configuration File

```python
SIMULATOR_CONFIG = {
    'num_los': 30,
    'num_questions_per_lo': 20,
    'num_contents_per_lo': 6,
    'max_episode_steps': 140,
    
    'irt': {
        'difficulty_ranges': {
            'easy': (-2.0, -0.5),
            'medium': (-0.5, 0.5),
            'hard': (0.5, 2.0)
        },
        'discrimination_ranges': {
            'easy': (0.5, 1.0),
            'medium': (1.0, 1.5),
            'hard': (1.5, 2.0)
        },
        'guessing_range': (0.1, 0.25)
    },
    
    'content': {
        'effectiveness_by_modality': {
            'video': (0.10, 0.15),
            'PPT': (0.08, 0.12),
            'text': (0.05, 0.08),
            'blog': (0.07, 0.10),
            'article': (0.06, 0.09),
            'handout': (0.05, 0.08)
        }
    },
    
    'reward_weights': {
        'correctness': 1.0,
        'mastery_gain': 0.5,
        'frustration_penalty': 0.3,
        'post_content_gain': 2.0
    },
    
    'termination': {
        'mastery_threshold': 0.8,
        'max_frustration': 0.95
    }
}
```

---

## Deliverables for Developer

1. **AdaptiveLearningEnv Class** (Gym interface)
2. **Question Bank** (600 questions, IRT-parameterized)
3. **Content Repository** (180 items across modalities)
4. **Learner Model** (IRT, mastery updates, engagement)
5. **Reward Function** (multi-objective)
6. **Data Generation Scripts** (200 learners, >50k transitions)
7. **Validation Suite** (sanity checks, unit tests)
8. **Config File** (all hyperparameters)
9. **Logging Utilities** (episode metrics, event tracking)
