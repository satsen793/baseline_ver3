# Introduction

Scope and gap:
- Adaptive learning platforms must decide when, how, and through which content modality learners progress; static rules fail to capture temporal dependencies between decisions.
- Existing mock-interview systems provided real-time feedback (e.g., facial/speech/grammar) but were primarily rule-based and could not tailor future sessions sequentially (see Patil2021).
- GAN-based virtual interview trainers identified behavioral weaknesses and boosted perceived performance (Heimerl2022) but did not solve the sequential decision problem.
- Goal-directed design work for AI mock interviews (Miao2020) improved UX but not long-horizon pedagogy.

Problem framing:
- We treat tutoring as a sequential decision problem and build an RL-driven Intelligent Decision Support System (IDSS) that selects question difficulty × LO and content modality based on learner state.
- Objectives: maximize cumulative mastery gain while minimizing frustration/overload; operate under limited interaction budgets and large discrete action spaces.

Algorithms compared:
- Rule-based heuristic; model-free DQN (with PER) and PPO; model-based PETS and MBPO adapted to factorized discrete actions.

Setup and reporting:
- Simulator with tagged LOs, question bank, and multi-modal content; six-month data-science mock-interview context.
- Outputs: learning curves, time-to-mastery, post-content gains by modality, mastery calibration for model-based methods, stability across seeds, and compute–reward trade-offs.
