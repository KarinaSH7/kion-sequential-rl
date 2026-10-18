"""Evaluate locked policies. Does not tune or update a model."""

import json
import time
import numpy as np
import pandas as pd
import torch
from .data import ROOT
from .source import dump, sha256
from .train import setup, reward_models, get_data, CKPT
from .evaluate import load_queries, diagnostics, reward_quality, rollout_diagnostics
from .models import make_policy


def main():
    started = time.time()
    output = ROOT / "results"
    device, cat = setup()
    locked = json.loads((output / "locked_policies.json").read_text())
    lock_hash = sha256(output / "locked_policies.json")
    queries = load_queries("test")
    rms = reward_models(cat, device)
    active = [q for q in queries if q["targets"]]
    oracle_recall = []
    oracle_ndcg = []
    for q in active:
        known = sum(t >= 2 for t in q["targets"])
        total = len(q["targets"])
        oracle_recall.append(known / total)
        oracle_ndcg.append(
            sum(1 / np.log2(i + 2) for i in range(min(known, 10)))
            / sum(1 / np.log2(i + 2) for i in range(min(total, 10)))
        )
    dump(
        output / "catalog_oracle.json",
        {
            "macro_recall_ceiling": float(np.mean(oracle_recall)),
            "ndcg10_ceiling": float(np.mean(oracle_ndcg)),
        },
    )
    constants = np.load(output / "reward_baselines.npz")
    rm, cont, cal, pred, sd = reward_quality(
        rms, get_data("test"), device, constants["item_means"], float(constants["mean"])
    )
    d = get_data("test")
    mask = d["cont_mask"].astype(bool)
    cont["train_mean_baseline_brier"] = float(
        np.mean((d["continuation"][mask] - float(constants["continuation_mean"])) ** 2)
    )
    pd.DataFrame(rm).to_csv(output / "reward_test_metrics.csv", index=False)
    dump(output / "continuation_test.json", cont)
    shuffled = dict(d)
    shuffled["x"] = d["x"][np.random.default_rng(42).permutation(len(d["x"]))]
    shuffled_rows, _, _, _, _ = reward_quality(
        rms, shuffled, device, constants["item_means"], float(constants["mean"])
    )
    dump(output / "reward_history_ablation.json", shuffled_rows[-1])
    pd.DataFrame(cal).to_csv(output / "reward_test_calibration.csv", index=False)
    np.savez_compressed(
        output / "reward_test_predictions.npz",
        prediction=pred,
        disagreement=sd,
        observed=d["reward"],
        continuation=d["continuation"],
        continuation_mask=mask,
        user=d["user"],
    )
    configs = [("baseline", 42, CKPT / "baseline.pt")]
    for arm, c in locked["arms"].items():
        for seed in locked["rl_seeds"]:
            configs.append((arm, seed, CKPT / f"{arm}_{seed}_step{c['step']}.pt"))
    results = []
    rollouts = []
    examples = []
    for arm, seed, path in configs:
        model = make_policy(cat["vocab_size"]).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        stem = output / "predictions" / f"{arm}_{seed}"
        metrics = diagnostics(model, rms, queries, cat, device, stem)
        results.append({"arm": arm, "seed": seed, "checkpoint_sha256": sha256(path), **metrics})
        frame, samples = rollout_diagnostics(model, rms, queries, cat, device)
        frame["arm"] = arm
        frame["seed"] = seed
        rollouts.append(frame)
        for sample in samples:
            sample.update(arm=arm, seed=seed)
        examples += samples
        print(results[-1], flush=True)
    pd.DataFrame(results).to_csv(output / "test_metrics.csv", index=False)
    pd.concat(rollouts).to_csv(output / "rollout_users.csv", index=False)
    pd.concat(rollouts).groupby(["arm", "seed", "mode", "step"])[
        [
            "completion",
            "continuation",
            "disagreement",
            "seen_in_real_history",
            "repeat_in_generated",
            "surprisal",
        ]
    ].mean().reset_index().to_csv(output / "rollout_summary.csv", index=False)
    dump(output / "rollout_examples.json", examples)
    reference = pd.read_csv(output / "predictions/baseline_42_users.csv")
    active = reference.active.to_numpy(dtype=bool)
    intervals = []
    for arm, seed, _ in configs[1:]:
        df = pd.read_csv(output / f"predictions/{arm}_{seed}_users.csv")
        for metric in ["ndcg10", "recall10", "predicted_completion"]:
            m = active if metric != "predicted_completion" else np.ones(len(df), dtype=bool)
            delta = (df[metric] - reference[metric]).to_numpy()[m]
            rng = np.random.default_rng(42)
            samples = np.array(
                [delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(2000)]
            )
            lo, hi = np.quantile(samples, [0.025, 0.975])
            intervals.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "metric": metric,
                    "delta": delta.mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_users": len(delta),
                }
            )
    pd.DataFrame(intervals).to_csv(output / "paired_bootstrap.csv", index=False)
    dump(
        output / "final_manifest.json",
        {
            "locked_policies_sha256": lock_hash,
            "policies": len(configs),
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 42,
            "test_used_for_selection": False,
            "evaluation_wall_s": time.time() - started,
        },
    )


if __name__ == "__main__":
    main()
