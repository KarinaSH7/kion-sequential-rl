# Checks and implementation decisions

- Same-day targets cannot see one another; histories use strict date inequality.
- Seven-day continuation labels are masked when the complete horizon is unavailable within the fitting/evaluation period.
- Unknown target IDs remain distinct and count in metric denominators.
- REINFORCE uses independent trajectories and a leave-one-out baseline, with a unit test against the exact expected bandit gradient.
- Reward predictors and the reference policy are frozen. Policy dropout is disabled during sampling and probability evaluation.
- A proposed repeat flag was removed from the reward predictor before RL: KION has one row per user–item pair, so this input is constant zero during supervised fitting and cannot be learned. Repetition is instead penalized explicitly in the conservative objective. Reward models were refit after this correction; the baseline checkpoint was preserved. No final-test metric informed this change.
- A separate replay + KL control isolates continued supervised optimization from the reward-gradient contribution.
