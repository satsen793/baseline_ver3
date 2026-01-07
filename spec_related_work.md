# Related Work

Summary of prior methods and their relevance to this system:

- **Knowledge tracing and ability estimation:** Bayesian Knowledge Tracing (BKT) and Item Response Theory (IRT) adapt difficulty from estimated proficiency; Deep Knowledge Tracing (DKT) and memory-augmented variants improve prediction but are not sequential decision-makers.
- **RL in education (model-free):** Q-learning/DQN applied to content sequencing and hints; policy-gradient methods used for curriculum optimization. They can be sample-inefficient and risk learner experience during exploration.
- **Model-based RL:** PETS (probabilistic ensembles with trajectory sampling) plans with uncertainty-aware rollouts; MBPO uses short model rollouts plus model-free updates to reduce model bias. Both show strong sample efficiency in other domains; this work adapts them to discrete pedagogical actions.
- **Reproducibility concerns:** RL is sensitive to seeds/noise; prior work stresses repeated runs and reporting variance. We follow paired-seed designs and statistical testing.
