# Metadata, Abstract, and Keywords

## Paper Metadata
- **Title:** Sample-Efficient Adaptive Mock Interview System using Model-Based Reinforcement Learning
- **Journal:** Decision Analytics Journal
- **Authors:** Shilpa Kadam; Snehanshu Banerjee; Jabez Christopher; P. T. V. Praveen Kumar; Dipak K. Satpathi

## Abstract (verbatim)
Adaptive learning platforms can improve mock-interview preparation by adapting question difficulty and recommending targeted learning content. However, learning effective policies is challenging under limited interaction budgets and large discrete action spaces. We present a reinforcement learning framework for adaptive mock interviews that selects question difficulty and learning outcomes and recommends multi-modal content based on learner state. Using a simulated environment with tagged learning outcomes, question banks, and content modalities, we compare five controllers: a rule-based heuristic, Deep Q-Network (DQN with prioritized replay), Proximal Policy Optimization (PPO), Probabilistic Ensembles with Trajectory Sampling (PETS), and Model-Based Policy Optimization (MBPO) adapted to factorized discrete actions.
Results show that PPO achieves strong reward-aligned performance with favorable compute cost, while DQN provides competitive returns with higher training time. Among model-based methods, PETS demonstrates stable learning with planning-driven improvements but higher compute overhead, whereas MBPO reduces average time-to-mastery but exhibits sensitivity across random seeds and weaker reward-aligned performance under the current reward specification. We report learning curves, time-to-mastery, post-content gains by modality, calibration of mastery estimates for model-based methods, stability across seeds, and compute–reward trade-offs. These findings highlight practical trade-offs between pedagogical efficiency, stability, and deployment cost when applying model-free versus model-based RL to adaptive learning.

## Keywords
Model-Based Reinforcement Learning; PETS; MBPO; Adaptive Assessment; Personalized Learning; Intelligent Tutoring System; Learning Outcomes; Content Recommendation; Data Science Education; Educational Simulation; IRT-based Question Modeling; Off-Policy Evaluation

## LaTeX Macros
- `\Cat`, `\softmax`, `\onehot` defined for categorical distributions, softmax, and one-hot encodings. Used conceptually in planning sections (PETS/MBPO) and MDP notation.
