# Protocol fixed before the final test

- Source: complete KION archive, pinned revision and SHA-256 in `checksums.json`.
- Cohort: 12,000 users sampled with seed 42 from users with at least five item pairs dated before 2021-07-12. Future activity is not an eligibility rule.
- Vocabulary: up to 1,000 most frequent items in this cohort strictly before 2021-07-12. No vocabulary expansion on validation/test.
- Supervised training / reward fitting: target date < 2021-07-12. At most 80,000 examples, sampled without replacement, preserving full history construction.
- Development: 2021-07-12 through 2021-07-25 (baseline and reward checkpoint selection).
- RL prompts: histories strictly before 2021-07-26, sampled from the original cohort; no future response labels used by RL.
- Policy selection: fixed cutoff 2021-08-02, first subsequent observed day through 2021-08-08.
- Final test: fixed cutoff 2021-08-09, first subsequent observed day through 2021-08-22. Only evaluate after selecting policies/configurations on development.
- No order is known within a day. Supervised examples use only strictly earlier days; all items on the first target day are relevant at evaluation. Ties in past context use item ID and are an acknowledged approximation.
- Tiny randomly initialized GPT-2: 2 layers, 4 heads, embedding dimension 64, context 32. Item IDs are tokens; no text checkpoint or tokenizer.
- Reward: fraction watched clipped to [0,1]. Three independently initialized conditional neural regressors; report held-out RMSE, MAE, R2, completion AUC and calibration against mean/item-mean baselines. No candidate's observed duration/completion enters input.
- REINFORCE: four independent trajectories per prompt, five steps, leave-one-out baseline, exact per-state KL to frozen supervised policy. Generated tokens enter the context; these are imagined actions, not simulated actual consumption.
- Control: supervised replay plus KL, with zero policy reward; same updates and selection rule. This separates the RL reward contribution from continued supervised training.
- Arms: raw completion (KL .01), completion with stronger KL (.15), conservative completion minus .5 ensemble standard deviation minus .3 repeat indicator plus .1 bounded novelty (KL .15), continuation proxy mixed with completion (KL .15). Each uses supervised replay weight .2. 80 updates; evaluate checkpoints 20/40/80 on policy-selection week. Select per arm by validation NDCG subject to predicted completion >= baseline; if none pass, select maximum NDCG and record infeasibility. Repeat selected arm settings with seeds 17 and 2026.
- Evaluate full-vocabulary top 10 without post-hoc masking seen items, and a separately labeled seen-filtered sensitivity. Include unknown next-day targets in metric denominators. Report eligible/active/empty-history counts and target coverage.
- Behavior: catalog coverage, top-one concentration, popularity, known-item repetition, inter-user top-10 overlap, genre diversity; five-step greedy rollout under appended-action and frozen-context histories.
- No causal off-policy value claim: logging propensities, impression negatives and real session boundaries are unavailable. Model-predicted reward is a diagnostic, not observed policy value.
- Test results never choose hyperparameters or a preferred arm. Paired user bootstrap is conditional on fitted models and the chosen temporal split.
