# Evaluation Framework & Metrics - Complete Specification

## Overview

This document specifies all evaluation metrics, statistical testing procedures, and reporting requirements for comparing RL algorithms in the adaptive mock-interview system.

---

## Primary Evaluation Metrics

### 1. Time-to-Mastery (TTM)

**Definition:** Number of steps required for mean mastery across all LOs to reach threshold (0.8).

**Computation:**
```python
def compute_time_to_mastery(episode_log, threshold=0.8):
    for step, state in enumerate(episode_log):
        mean_mastery = np.mean(state['mastery_vector'])
        if mean_mastery >= threshold:
            return step + 1
    return None  # Mastery not achieved
```

**Reporting:**
- Mean ± SD across S seeds
- 95% Confidence Interval
- Median and IQR (robust to outliers)

**Interpretation:**
- Lower is better (faster learning)
- Key sample-efficiency metric

---

### 2. Cumulative Reward

**Definition:** Total reward accumulated over entire episode.

**Computation:**
```python
def compute_cumulative_reward(episode_log):
    return sum(transition['reward'] for transition in episode_log)
```

**Reporting:**
- Mean ± SD across seeds
- 95% CI
- Learning curves (reward vs. episode)

**Interpretation:**
- Higher is better
- Reflects overall pedagogical effectiveness

---

### 3. Post-Content Gain

**Definition:** Average mastery improvement immediately after content recommendation.

**Computation:**
```python
def compute_post_content_gain(episode_log):
    gains = []
    for transition in episode_log:
        if transition['action_type'] == 'content':
            gains.append(transition['mastery_gain'])
    
    if not gains:
        return 0.0
    return np.mean(gains)
```

**Reporting:**
- Mean gain per content action
- Breakdown by modality (video, PPT, text, etc.)
- SD across seeds

**Interpretation:**
- Higher is better
- Measures content recommendation quality

---

### 4. Blueprint Adherence

**Definition:** Deviation from target difficulty distribution (20% Easy, 60% Medium, 20% Hard).

**Computation:**
```python
def compute_blueprint_adherence(episode_log):
    difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
    
    for transition in episode_log:
        if transition['action_type'] == 'question':
            difficulty_counts[transition['difficulty']] += 1
    
    total = sum(difficulty_counts.values())
    if total == 0:
        return 100.0  # No questions asked
    
    actual = {
        'easy': difficulty_counts['easy'] / total,
        'medium': difficulty_counts['medium'] / total,
        'hard': difficulty_counts['hard'] / total
    }
    
    target = {'easy': 0.20, 'medium': 0.60, 'hard': 0.20}
    
    # Mean absolute deviation
    deviation = sum(abs(actual[d] - target[d]) for d in target) / len(target)
    adherence = 1.0 - deviation
    
    return adherence * 100  # Percentage
```

**Reporting:**
- Adherence % (100% = perfect match)
- Deviation in percentage points
- Per-difficulty breakdown

**Interpretation:**
- Higher is better
- Ensures balanced assessment

---

### 5. Policy Stability (Reward Variance)

**Definition:** Variance in cumulative reward across random seeds.

**Computation:**
```python
def compute_policy_stability(results_across_seeds):
    cumulative_rewards = [r['cumulative_reward'] for r in results_across_seeds]
    return np.std(cumulative_rewards)
```

**Reporting:**
- Standard deviation of rewards
- Coefficient of variation (CV = SD/mean)
- Range (max - min)

**Interpretation:**
- Lower is better (more stable policy)
- Critical for deployment reliability

---

## Secondary Metrics

### 6. Question Accuracy

**Definition:** Percentage of questions answered correctly.

```python
def compute_question_accuracy(episode_log):
    correct = sum(1 for t in episode_log 
                  if t['action_type'] == 'question' and t['correct'])
    total = sum(1 for t in episode_log if t['action_type'] == 'question')
    
    return correct / total if total > 0 else 0.0
```

---

### 7. Content Recommendation Rate

**Definition:** Proportion of actions that are content recommendations.

```python
def compute_content_rate(episode_log):
    content_count = sum(1 for t in episode_log if t['action_type'] == 'content')
    return content_count / len(episode_log)
```

---

### 8. Final Mastery Level

**Definition:** Mean mastery across all LOs at episode end.

```python
def compute_final_mastery(episode_log):
    final_state = episode_log[-1]
    return np.mean(final_state['mastery_vector'])
```

---

### 9. Frustration Trajectory

**Definition:** Mean frustration level over episode.

```python
def compute_mean_frustration(episode_log):
    return np.mean([t['frustration'] for t in episode_log])
```

---

### 10. Model-Specific Metrics (PETS/MBPO)

#### Model Prediction Error
```python
def compute_model_error(predictions, actuals):
    # MSE for state predictions
    return np.mean((predictions - actuals) ** 2)
```

#### Ensemble Disagreement
```python
def compute_ensemble_disagreement(ensemble_predictions):
    # Std dev across ensemble members
    return np.std(ensemble_predictions, axis=0).mean()
```

#### Calibration Score
```python
def compute_calibration(predicted_returns, actual_returns, num_bins=10):
    # ECE (Expected Calibration Error)
    bins = np.linspace(0, 1, num_bins + 1)
    # ... implementation
    return ece
```

---

## Statistical Testing Protocol

### Experimental Design

**Setup:**
- **Algorithms:** Rule-Based, DQN, PPO, PETS, MBPO
- **Seeds:** Main runs S = 20 independent random initializations; pilot paired runs for PETS/MBPO used S = 5 (same seeds) for early tuning/comparison.
- **Environment:** Identical simulator configuration per seed
- **Paired Design:** Each algorithm runs on same seeds

### Comparison Framework

#### 1. Pairwise Comparisons

**Paired t-test:**
```python
from scipy.stats import ttest_rel, shapiro

def compare_algorithms(algo1_results, algo2_results, metric='time_to_mastery'):
    # Extract metric values
    values1 = [r[metric] for r in algo1_results]
    values2 = [r[metric] for r in algo2_results]
    
    # Check normality (Shapiro-Wilk)
    _, p1 = shapiro(values1)
    _, p2 = shapiro(values2)
    
    if p1 > 0.05 and p2 > 0.05:
        # Use paired t-test
        t_stat, p_value = ttest_rel(values1, values2)
        test_used = 'paired_t_test'
    else:
        # Use Wilcoxon signed-rank test
        from scipy.stats import wilcoxon
        t_stat, p_value = wilcoxon(values1, values2)
        test_used = 'wilcoxon'
    
    # Effect size (Cohen's d for paired samples)
    differences = np.array(values1) - np.array(values2)
    d = np.mean(differences) / np.std(differences)
    
    return {
        'test': test_used,
        'statistic': t_stat,
        'p_value': p_value,
        'cohens_d': d,
        'significant': p_value < 0.05
    }
```

#### 2. Effect Size Interpretation

**Cohen's d:**
- Small: |d| < 0.5
- Medium: 0.5 ≤ |d| < 0.8
- Large: |d| ≥ 0.8

#### 3. Multiple Comparison Correction

When comparing 5 algorithms (10 pairwise tests):
```python
from statsmodels.stats.multitest import multipletests

def correct_multiple_comparisons(p_values, method='holm'):
    reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method=method)
    return p_corrected, reject
```

**Methods:**
- Bonferroni (conservative)
- Holm (less conservative)
- FDR (Benjamini-Hochberg)

---

## Confidence Intervals

### Bootstrap CI (1000 iterations)

```python
def bootstrap_ci(data, statistic_fn, confidence=0.95, n_bootstrap=1000):
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_fn(sample))
    
    lower = np.percentile(bootstrap_stats, (1-confidence)/2 * 100)
    upper = np.percentile(bootstrap_stats, (1+confidence)/2 * 100)
    
    return lower, upper
```

### Parametric CI (for normally distributed metrics)

```python
from scipy import stats

def parametric_ci(data, confidence=0.95):
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return mean - margin, mean + margin
```

---

## Learning Curves

### Episode-Level Tracking

```python
def collect_learning_curve(algorithm, env, num_episodes, seeds):
    results = []
    
    for seed in seeds:
        episode_rewards = []
        episode_mastery = []
        
        for episode in range(num_episodes):
            obs = env.reset(seed=seed*num_episodes + episode)
            done = False
            ep_reward = 0
            
            while not done:
                action = algorithm.select_action(obs)
                obs, reward, done, info = env.step(action)
                ep_reward += reward
            
            episode_rewards.append(ep_reward)
            episode_mastery.append(info['mean_mastery'])
        
        results.append({
            'seed': seed,
            'episode_rewards': episode_rewards,
            'episode_mastery': episode_mastery
        })
    
    return results
```

### Plotting with Confidence Bands

```python
import matplotlib.pyplot as plt

def plot_learning_curves(results_dict, metric='episode_rewards'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo_name, results in results_dict.items():
        # Extract metric across all seeds
        all_curves = [r[metric] for r in results]
        
        # Compute mean and 95% CI at each episode
        mean_curve = np.mean(all_curves, axis=0)
        ci_lower, ci_upper = bootstrap_ci_per_episode(all_curves)
        
        episodes = np.arange(len(mean_curve))
        
        ax.plot(episodes, mean_curve, label=algo_name, linewidth=2)
        ax.fill_between(episodes, ci_lower, ci_upper, alpha=0.2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Learning Curves (Mean ± 95% CI)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig
```

---

## Reporting Template

### Table 1: Primary Metrics Comparison

```
| Algorithm   | TTM (steps)    | Cumulative Reward | Post-Content Gain | Blueprint (%) | Variance |
|-------------|----------------|-------------------|-------------------|---------------|----------|
| Rule-Based  | 120 ± 15       | 70 ± 8            | 0.08 ± 0.02       | 78 ± 5        | 8.0      |
| DQN-PER     | 95 ± 12 **     | 85 ± 15           | 0.10 ± 0.03       | 92 ± 3        | 15.0     |
| PPO         | 87 ± 8 ***     | 92 ± 10 *         | 0.11 ± 0.02       | 94 ± 2        | 10.0     |
| PETS        | 75 ± 10 ***    | 95 ± 12 **        | 0.12 ± 0.03       | 91 ± 3        | 12.0     |
| MBPO        | 70 ± 6 ***     | 98 ± 8 ***        | 0.13 ± 0.02       | 93 ± 2        | 8.0      |

* p < 0.05, ** p < 0.01, *** p < 0.001 (vs Rule-Based)
Values: Mean ± SD (S=20 seeds)
```

### Table 2: Pairwise Comparisons

```
| Comparison      | TTM Δ (%) | p-value | Cohen's d | Interpretation       |
|-----------------|-----------|---------|-----------|----------------------|
| PPO vs DQN      | -8.4%     | 0.042   | 0.52      | PPO faster (medium)  |
| MBPO vs PPO     | -19.5%    | 0.003   | 1.24      | MBPO faster (large)  |
| MBPO vs PETS    | -6.7%     | 0.089   | 0.38      | Not significant      |
| PETS vs PPO     | -13.8%    | 0.012   | 0.89      | PETS faster (large)  |
| PPO vs Rule     | -27.5%    | <0.001  | 2.15      | PPO much faster      |
```

---

## Off-Policy Evaluation (Optional)

For algorithms trained on logged data:

### Inverse Propensity Scoring (IPS)

```python
def ips_estimator(logged_data, policy, behavior_policy):
    total_return = 0
    importance_weights = []
    
    for trajectory in logged_data:
        trajectory_return = 0
        cumulative_weight = 1.0
        
        for (s, a, r) in trajectory:
            pi_target = policy.prob(a | s)
            pi_behavior = behavior_policy.prob(a | s)
            
            weight = pi_target / pi_behavior
            cumulative_weight *= weight
            
            trajectory_return += r
            importance_weights.append(cumulative_weight)
        
        total_return += cumulative_weight * trajectory_return
    
    # Effective sample size
    ess = (sum(importance_weights) ** 2) / sum(w**2 for w in importance_weights)
    
    return total_return / len(logged_data), ess
```

### Doubly Robust (DR) Estimator

Combines IPS with model-based value estimation for lower variance.

---

## Compute Cost Analysis

### Metrics to Track

1. **Wall-Clock Time:** Total training time per seed
2. **Steps per Second:** Throughput metric
3. **Memory Usage:** Peak RAM/GPU memory
4. **Model Calls:** Number of dynamics model queries (MBPO/PETS)

```python
import time
import psutil

def track_compute_cost(training_fn):
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024**2  # MB
    
    result = training_fn()
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024**2
    
    return {
        'wall_clock_time': end_time - start_time,
        'peak_memory_mb': end_memory,
        'result': result
    }
```

---

## Reproducibility Checklist

### Requirements for Full Reproducibility

- [ ] **Seeds Recorded:** All random seeds saved
- [ ] **Config Files:** Hyperparameters version-controlled
- [ ] **Environment Version:** Simulator code + dependencies pinned
- [ ] **Data Checksums:** MD5/SHA for question bank, content repo
- [ ] **Code Version:** Git commit hash recorded
- [ ] **Hardware Specs:** CPU/GPU model documented
- [ ] **Logs:** Complete episode logs saved (per seed)

### Replication Protocol

```python
def replicate_experiment(algorithm, config, seeds):
    results = []
    
    for seed in seeds:
        # Set all random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize environment and agent
        env = AdaptiveLearningEnv(config['env'])
        agent = algorithm(config['agent'])
        
        # Train
        episode_log = train(agent, env, config['training'])
        
        # Evaluate
        metrics = evaluate(agent, env, config['evaluation'])
        
        results.append({
            'seed': seed,
            'config': config,
            'metrics': metrics,
            'log': episode_log
        })
    
    return results
```

---

## Deliverables for Developer

1. **Metrics Module** (`metrics.py`)
   - All metric computation functions
   - Statistical test implementations
   
2. **Evaluation Script** (`evaluate.py`)
   - Run all algorithms on test seeds
   - Generate comparison tables
   
3. **Plotting Utilities** (`plot_utils.py`)
   - Learning curves
   - Bar charts for metric comparison
   - Heatmaps for pairwise tests
   
4. **Report Generator** (`generate_report.py`)
   - Markdown/LaTeX output
   - Automated tables and figures
   
5. **Config Templates** (`eval_configs/`)
   - YAML/JSON configs for each experiment
   
6. **CI/CD Integration**
   - Unit tests for metric functions
   - Regression tests (expected ranges)

---

## Example Report Snippet (Markdown)

```markdown
# Results: Time-to-Mastery Comparison

MBPO achieved the fastest time-to-mastery (70 ± 6 steps, S=20), 
representing a **19.5% improvement** over PPO (87 ± 8 steps, 
p=0.003, Cohen's d=1.24, large effect). PETS also significantly 
outperformed model-free baselines (75 ± 10 steps, p<0.01 vs DQN).

All RL methods substantially outperformed the rule-based baseline 
(120 ± 15 steps), with reductions of 27-42% (all p<0.001).

**Key Finding:** Model-based methods (PETS, MBPO) demonstrate 
superior sample efficiency, critical for real-world deployment 
where learner interaction data is expensive.
```

---

## Final Notes

- **Prioritize Reproducibility:** Document everything
- **Report Uncertainty:** Always include SD/CI, never just means
- **Effect Sizes Matter:** Statistical significance ≠ practical importance
- **Multiple Metrics:** No single metric tells the full story
- **Transparency:** Report negative results and limitations
