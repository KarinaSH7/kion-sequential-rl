"""Full-catalog ranking and explicit model-based behavior diagnostics."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from .data import ROOT
from .models import policy_logits, ensemble, append_action


def load_queries(split):
    return json.loads((ROOT / f"data/processed/{split}_queries.json").read_text())


def rank_metrics(ranked, targets, k=10):
    relevant = set(targets)
    if not relevant:
        return 0.0, 0.0
    hits = np.array([int(i in relevant) for i in ranked[:k]])
    return float(hits.sum() / len(relevant)), float(
        (hits / np.log2(np.arange(len(hits)) + 2)).sum()
        / sum(1 / np.log2(np.arange(min(k, len(relevant))) + 2))
    )


@torch.no_grad()
def rank_policy(model, queries, device, batch=256, filter_seen=False):
    model.eval()
    output = []
    for start in range(0, len(queries), batch):
        qs = queries[start : start + batch]
        x = torch.tensor([q["context"] for q in qs], device=device)
        scores = policy_logits(model, x)
        if filter_seen:
            for j, q in enumerate(qs):
                scores[j, q["seen"]] = -1e9
        output.append(scores.topk(10, dim=1).indices.cpu().numpy())
    return np.concatenate(output)


@torch.no_grad()
def diagnostics(model, rewards, queries, catalog, device, out=None):
    top = rank_policy(model, queries, device)
    active = np.array([bool(q["targets"]) for q in queries])
    per = np.array([rank_metrics(t, q["targets"]) for t, q in zip(top, queries)])
    seen = np.array([np.isin(t, q["seen"]).mean() for t, q in zip(top, queries)])
    reward = []
    unc = []
    cont = []
    for st in range(0, len(queries), 128):
        qs = queries[st : st + 128]
        x = torch.tensor([q["context"] for q in qs], device=device).repeat_interleave(10, 0)
        a = torch.tensor(top[st : st + 128].reshape(-1), device=device)
        mu, sd = ensemble(rewards, x, a)
        reward.extend(mu[:, 0].cpu().numpy())
        unc.extend(sd[:, 0].cpu().numpy())
        cont.extend(mu[:, 1].cpu().numpy())
    counts = np.bincount(top[:, 0], minlength=catalog["vocab_size"])
    genres = {m["token"]: set(m["genres"]) for m in catalog["metadata"]}
    diversity = np.mean(
        [
            np.mean(
                [
                    1
                    - len(genres[int(t[i])] & genres[int(t[j])])
                    / max(1, len(genres[int(t[i])] | genres[int(t[j])]))
                    for i in range(10)
                    for j in range(i)
                ]
            )
            for t in top
        ]
    )
    rng = np.random.default_rng(42)
    pairs = rng.integers(0, len(top), size=(1000, 2))
    overlap = np.mean([len(set(top[a]) & set(top[b])) / 10 for a, b in pairs if a != b])
    known = sum(sum(t >= 2 for t in q["targets"]) for q in queries)
    total = sum(len(q["targets"]) for q in queries)
    result = {
        "queries": len(queries),
        "active_queries": int(active.sum()),
        "recall10": float(per[active, 0].mean()),
        "ndcg10": float(per[active, 1].mean()),
        "ndcg10_all": float(per[:, 1].mean()),
        "catalog_coverage10": len(np.unique(top)) / (catalog["vocab_size"] - 2),
        "top1_share": float(counts.max() / len(top)),
        "top1_item_token": int(counts.argmax()),
        "repeat10": float(seen.mean()),
        "pair_overlap10": float(overlap),
        "genre_diversity10": float(diversity),
        "predicted_completion10": float(np.mean(reward)),
        "reward_disagreement10": float(np.mean(unc)),
        "predicted_continuation10": float(np.mean(cont)),
        "mean_train_popularity10": float(np.array(catalog["popularity"])[top].mean()),
        "target_catalog_coverage": known / max(1, total),
    }
    priors = np.load(ROOT / "results/reward_baselines.npz")["item_means"]
    result["item_prior_completion10"] = float(priors[top].mean())
    films = np.zeros(catalog["vocab_size"])
    for item in catalog["metadata"]:
        films[item["token"]] = item["content_type"] == "film"
    result["film_share10"] = float(films[top].mean())
    filtered = rank_policy(model, queries, device, filter_seen=True)
    fp = np.array([rank_metrics(t, q["targets"]) for t, q in zip(filtered, queries)])
    result.update(
        filtered_recall10=float(fp[active, 0].mean()), filtered_ndcg10=float(fp[active, 1].mean())
    )
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "user_id": [q["user_id"] for q in queries],
                "active": active,
                "recall10": per[:, 0],
                "ndcg10": per[:, 1],
                "repeat10": seen,
                "predicted_completion": np.array(reward).reshape(-1, 10).mean(1),
            }
        ).to_csv(str(out) + "_users.csv", index=False)
        np.savez_compressed(str(out) + "_top10.npz", top10=top, filtered=filtered)
    return result


@torch.no_grad()
def reward_quality(models, data, device, item_means, global_mean):
    preds = []
    std = []
    for st in range(0, len(data["y"]), 512):
        x = torch.tensor(data["x"][st : st + 512], device=device)
        a = torch.tensor(data["y"][st : st + 512], device=device)
        mu, sd = ensemble(models, x, a)
        preds.append(mu.cpu().numpy())
        std.append(sd.cpu().numpy())
    pred = np.concatenate(preds)
    sd = np.concatenate(std)
    y = data["reward"]
    rows = []
    for name, p in [
        ("mean", np.full(len(y), global_mean)),
        ("item_mean", item_means[data["y"]]),
        ("reward_ensemble", pred[:, 0]),
    ]:
        row = {
            "model": name,
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)),
            "auc_completion80": float(roc_auc_score(y >= 0.8, p)),
        }
        rows.append(row)
    m = data["cont_mask"].astype(bool)
    continuation = {
        "n": int(m.sum()),
        "base_rate": float(data["continuation"][m].mean()),
        "brier": float(np.mean((pred[m, 1] - data["continuation"][m]) ** 2)),
        "test_prevalence_oracle_brier": float(np.var(data["continuation"][m])),
        "auc": float(roc_auc_score(data["continuation"][m], pred[m, 1])),
    }
    cal = []
    for lo in np.arange(0, 1, 0.1):
        ix = (pred[:, 0] >= lo) & (pred[:, 0] < lo + 0.1)
        if ix.any():
            cal.append(
                {
                    "bin": float(lo),
                    "n": int(ix.sum()),
                    "predicted": float(pred[ix, 0].mean()),
                    "observed": float(y[ix].mean()),
                    "rmse": float(np.sqrt(np.mean((y[ix] - pred[ix, 0]) ** 2))),
                }
            )
    return rows, continuation, cal, pred, sd


@torch.no_grad()
def rollout_diagnostics(model, rewards, queries, catalog, device):
    rows = []
    samples = []
    # Fixed user sample, selected without looking at outcomes.
    idx = np.sort(
        np.random.default_rng(42).choice(len(queries), min(1024, len(queries)), replace=False)
    )
    for mode in ["append_action", "frozen_context"]:
        for st in range(0, len(idx), 128):
            qs = [queries[i] for i in idx[st : st + 128]]
            x = torch.tensor([q["context"] for q in qs], device=device)
            histories = [set(q["seen"]) for q in qs]
            generated = []
            for step in range(5):
                logp = policy_logits(model, x).log_softmax(-1)
                a = logp.argmax(-1)
                mu, sd = ensemble(rewards, x, a)
                aa = a.cpu().numpy()
                for j, q in enumerate(qs):
                    rows.append(
                        {
                            "mode": mode,
                            "step": step + 1,
                            "user_id": q["user_id"],
                            "token": int(aa[j]),
                            "completion": float(mu[j, 0]),
                            "continuation": float(mu[j, 1]),
                            "disagreement": float(sd[j, 0]),
                            "seen_in_real_history": float(aa[j] in histories[j]),
                            "repeat_in_generated": float(any(aa[j] == z[j] for z in generated)),
                            "surprisal": float(-logp[j, a[j]]),
                        }
                    )
                generated.append(aa)
                if mode == "append_action":
                    x = append_action(x, a)
            if st == 0:
                for j, q in enumerate(qs[:5]):
                    samples.append(
                        {
                            "mode": mode,
                            "user_id": q["user_id"],
                            "context": q["context"],
                            "generated": [int(a[j]) for a in generated],
                        }
                    )
    return pd.DataFrame(rows), samples
