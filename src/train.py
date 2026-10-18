"""Supervised checkpoints and on-policy REINFORCE with a frozen response model."""

import argparse
import copy
import json
import time
import platform
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .data import ROOT
from .source import dump
from .models import make_policy, RewardModel, seed_all, policy_logits, ensemble, append_action
from .evaluate import load_queries, rank_policy, rank_metrics, diagnostics, reward_quality

OUT = ROOT / "results"
CKPT = OUT / "checkpoints"
ARMS = {
    "replay_kl_control": {"beta": 0.15, "kind": "replay"},
    "completion_weak_kl": {"beta": 0.01, "kind": "completion"},
    "completion_strong_kl": {"beta": 0.15, "kind": "completion"},
    "conservative": {"beta": 0.15, "kind": "conservative"},
    "continuation_mix": {"beta": 0.15, "kind": "continuation"},
}


def setup():
    torch.set_num_threads(4)
    device = torch.device("cpu")
    CKPT.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((OUT / "catalog.json").read_text())
    return device, catalog


def get_data(split):
    return dict(np.load(ROOT / f"data/processed/{split}.npz"))


def reward_models(catalog, device):
    models = []
    for seed in [42, 17, 2026]:
        m = RewardModel(catalog["vocab_size"]).to(device)
        m.load_state_dict(
            torch.load(CKPT / f"reward_{seed}.pt", map_location=device, weights_only=True)
        )
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
    return models


def baseline(catalog, device):
    m = make_policy(catalog["vocab_size"]).to(device)
    m.load_state_dict(torch.load(CKPT / "baseline.pt", map_location=device, weights_only=True))
    m.eval()
    return m


def supervised():
    device, cat = setup()
    seed_all(42)
    start = time.time()
    train, dev = get_data("train"), get_data("dev")
    queries = load_queries("dev")
    model = make_policy(cat["vocab_size"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    history = []
    best = -1
    x = torch.tensor(train["x"], device=device)
    y = torch.tensor(train["y"], device=device)
    for epoch in range(1, 9):
        model.train()
        order = torch.randperm(len(y))
        losses = []
        for batch in order.split(128):
            opt.zero_grad()
            loss = F.cross_entropy(policy_logits(model, x[batch]), y[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            losses.append(float(loss.detach()))
        ranked = rank_policy(model, queries, device)
        score = np.mean(
            [rank_metrics(r, q["targets"])[1] for r, q in zip(ranked, queries) if q["targets"]]
        )
        history.append(
            {
                "stage": "baseline",
                "epoch": epoch,
                "train_ce": np.mean(losses),
                "dev_ndcg10": score,
                "elapsed_s": time.time() - start,
            }
        )
        if score > best:
            best = score
            torch.save(model.state_dict(), CKPT / "baseline.pt")
            best_epoch = epoch
        print(history[-1], flush=True)
    dump(
        OUT / "baseline_selection.json",
        {
            "epoch": best_epoch,
            "dev_ndcg10": best,
            "parameters": sum(p.numel() for p in model.parameters()),
            "config": model.config.to_dict(),
        },
    )
    pd.DataFrame(history).to_csv(OUT / "baseline_training.csv", index=False)
    fit_rewards(train, dev, cat, device, start)


def fit_rewards(train, dev, cat, device, start):
    rmhistory = []
    tx = torch.tensor(train["x"], device=device)
    ta = torch.tensor(train["y"], device=device)
    ty = torch.tensor(
        np.stack([train["reward"], train["continuation"]], 1), dtype=torch.float32, device=device
    )
    tm = torch.tensor(train["cont_mask"], dtype=torch.float32, device=device)
    dx = torch.tensor(dev["x"], device=device)
    da = torch.tensor(dev["y"], device=device)
    for seed in [42, 17, 2026]:
        seed_all(seed)
        m = RewardModel(cat["vocab_size"]).to(device)
        optimizer = torch.optim.AdamW(m.parameters(), lr=0.002, weight_decay=0.01)
        best = np.inf
        for epoch in range(1, 9):
            m.train()
            losses = []
            for batch in torch.randperm(len(ta)).split(256):
                optimizer.zero_grad()
                p = m(tx[batch], ta[batch])
                loss = F.mse_loss(p[:, 0], ty[batch, 0]) + 0.3 * (
                    F.binary_cross_entropy(p[:, 1], ty[batch, 1], reduction="none") * tm[batch]
                ).sum() / tm[batch].sum().clamp(min=1)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            m.eval()
            with torch.no_grad():
                pred = (
                    torch.cat([m(dx[b], da[b]) for b in torch.arange(len(da)).split(512)])
                    .cpu()
                    .numpy()
                )
            mse = np.mean((pred[:, 0] - dev["reward"]) ** 2)
            rmhistory.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": np.mean(losses),
                    "dev_rmse": float(np.sqrt(mse)),
                }
            )
            if mse < best:
                best = mse
                torch.save(m.state_dict(), CKPT / f"reward_{seed}.pt")
        print("reward", seed, "best_dev_rmse", np.sqrt(best), flush=True)
    pd.DataFrame(rmhistory).to_csv(OUT / "reward_training.csv", index=False)
    means = np.full(cat["vocab_size"], train["reward"].mean())
    sums = np.bincount(train["y"], weights=train["reward"], minlength=len(means))
    counts = np.bincount(train["y"], minlength=len(means))
    means = (sums + 20 * train["reward"].mean()) / (counts + 20)
    np.savez(
        OUT / "reward_baselines.npz",
        item_means=means,
        mean=train["reward"].mean(),
        continuation_mean=np.average(train["continuation"], weights=train["cont_mask"]),
    )
    rms = reward_models(cat, device)
    rows, cont, cal, _, _ = reward_quality(rms, dev, device, means, train["reward"].mean())
    pd.DataFrame(rows).to_csv(OUT / "reward_dev_metrics.csv", index=False)
    dump(OUT / "continuation_dev.json", cont)
    pd.DataFrame(cal).to_csv(OUT / "reward_dev_calibration.csv", index=False)
    dump(
        OUT / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "threads": 4,
            "stage_wall_s": time.time() - start,
        },
    )


def reward_value(mu, sd, repeat, novelty, kind):
    if kind == "replay":
        return torch.zeros_like(mu[:, 0])
    if kind == "conservative":
        return mu[:, 0] - 0.5 * sd[:, 0] - 0.3 * repeat + 0.1 * novelty
    if kind == "continuation":
        return 0.5 * mu[:, 0] + 0.5 * mu[:, 1] - 0.3 * repeat
    return mu[:, 0]


def leave_one_out(returns, group=4):
    values = returns.reshape(-1, group)
    return (values - (values.sum(1, keepdim=True) - values) / (group - 1)).reshape(-1)


def rl_run(arm, seed, steps, device, cat, ref, rms, queries, train):
    seed_all(seed)
    model = copy.deepcopy(ref)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    # Dropout stays disabled for coherent on-policy probabilities; gradients remain enabled.
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    prompts = load_queries("rl")
    novelty = torch.tensor(cat["novelty"], device=device)
    logs = []
    valid = []
    t0 = time.time()
    for step in range(1, steps + 1):
        ids = np.random.choice(len(prompts), 16, replace=False)
        qs = [prompts[i] for i in ids]
        x = torch.tensor([q["context"] for q in qs], device=device).repeat_interleave(4, 0)
        seen = torch.zeros(len(qs), cat["vocab_size"], dtype=torch.bool, device=device)
        for j, q in enumerate(qs):
            seen[j, q["seen"]] = True
        seen = seen.repeat_interleave(4, 0)
        rewards = []
        logps = []
        kls = []
        for h in range(5):
            logp = policy_logits(model, x).log_softmax(-1)
            p = logp.exp()
            with torch.no_grad():
                ref_logp = policy_logits(ref, x).log_softmax(-1)
            a = torch.multinomial(p, 1).squeeze(1)
            logps.append(logp.gather(1, a[:, None]).squeeze(1))
            kls.append((p * (logp - ref_logp)).sum(1))
            with torch.no_grad():
                mu, sd = ensemble(rms, x, a)
                repeat = seen.gather(1, a[:, None]).squeeze(1).float()
                rewards.append(reward_value(mu, sd, repeat, novelty[a], ARMS[arm]["kind"]))
            seen = seen.scatter(1, a[:, None], True)
            x = append_action(x, a)
        # Full trajectory return with independent leave-one-out baseline is unbiased.
        discount = torch.tensor([0.95**h for h in range(5)], device=device)
        returns = (torch.stack(rewards, 1) * discount).sum(1) / discount.sum()
        advantage = leave_one_out(returns).detach()
        policy_loss = -(torch.stack(logps, 1).sum(1) * advantage).mean()
        kl = torch.stack(kls, 1).mean()
        replay = np.random.choice(len(train["y"]), 128, replace=False)
        sx = torch.tensor(train["x"][replay], device=device)
        sy = torch.tensor(train["y"][replay], device=device)
        ce = F.cross_entropy(policy_logits(model, sx), sy)
        loss = policy_loss + ARMS[arm]["beta"] * kl + 0.2 * ce
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        logs.append(
            {
                "arm": arm,
                "seed": seed,
                "step": step,
                "rollout_objective": float(returns.mean()),
                "kl": float(kl.detach()),
                "replay_ce": float(ce.detach()),
                "policy_loss": float(policy_loss.detach()),
                "elapsed_s": time.time() - t0,
            }
        )
        if step in [20, 40, 80] or step == steps:
            path = CKPT / f"{arm}_{seed}_step{step}.pt"
            torch.save(model.state_dict(), path)
            if seed == 42:
                metrics = diagnostics(model, rms, queries, cat, device)
                valid.append({"arm": arm, "seed": seed, "step": step, **metrics})
            print("rl", logs[-1], flush=True)
    return logs, valid


def rl():
    device, cat = setup()
    ref = baseline(cat, device)
    for p in ref.parameters():
        p.requires_grad_(False)
    # deepcopy inherits requires_grad=False, so enable only the trainable policy below.
    rms = reward_models(cat, device)
    queries = load_queries("tune")
    train = get_data("train")
    reference = diagnostics(ref, rms, queries, cat, device)
    dump(OUT / "baseline_tune.json", reference)
    logs = []
    valid = []
    selected = {}
    for arm in ARMS:
        ll, vv = rl_run(arm, 42, 80, device, cat, ref, rms, queries, train)
        logs += ll
        valid += vv
        feasible = [
            v for v in vv if v["predicted_completion10"] >= reference["predicted_completion10"]
        ]
        chosen = max(feasible or vv, key=lambda v: (v["ndcg10"], -v["step"]))
        selected[arm] = {
            "step": chosen["step"],
            "reward_constraint_feasible": bool(feasible),
            "settings": ARMS[arm],
        }
    dump(
        OUT / "locked_policies.json",
        {
            "arms": selected,
            "reference": reference,
            "selection_rule": "max validation NDCG under predicted completion >= baseline; fall back to max NDCG if infeasible",
            "baseline_seed": 42,
            "rl_seeds": [42, 17, 2026],
        },
    )
    pd.DataFrame(valid).to_csv(OUT / "validation_policies.csv", index=False)
    for arm, choice in selected.items():
        for seed in [17, 2026]:
            ll, _ = rl_run(arm, seed, choice["step"], device, cat, ref, rms, queries, train)
            logs += ll
    pd.DataFrame(logs).to_csv(OUT / "rl_training.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["supervised", "reward", "rl"])
    a = p.parse_args()
    if a.stage == "reward":
        device, cat = setup()
        fit_rewards(get_data("train"), get_data("dev"), cat, device, time.time())
    else:
        {"supervised": supervised, "rl": rl}[a.stage]()
