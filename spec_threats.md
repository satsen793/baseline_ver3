# Threats to Validity

- **Internal validity:** Mitigated by identical simulator/reward/hyperparameter grids across algorithms and fixed seeds per trial; residual risk from initialization and hyperparameter sensitivity.
- **External validity:** Results come from simulation; generalization to real learners requires caution, though state/reward features align with LMS-observable signals (accuracy, response time, engagement).
- **Construct validity:** Reward combines correctness, mastery gain, and frustration; mastery and frustration are inferred. Future work: add multimodal affective signals for better constructs.
- **Statistical validity:** Multiple seeds (paired design), mean ± SD with 95% CIs, normality checks, paired t-test or Wilcoxon, effect sizes (Cohen’s d/Cliff’s δ), bootstrap CIs. Larger replications improve confidence.
