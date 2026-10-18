"""Chronological examples; no invented within-day target ordering."""

import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .source import download, sha256, dump, REVISION

ROOT = Path(__file__).resolve().parents[1]
TRAIN_END = np.datetime64("2021-07-12")
MAXLEN = 32


def context(tokens):
    out = np.zeros(MAXLEN, dtype=np.int64)
    v = np.asarray(tokens[-MAXLEN:], dtype=np.int64)
    if len(v):
        out[-len(v) :] = v
    return out


def first_day_targets(days, tokens, start, end):
    idx = np.flatnonzero((days >= start) & (days < end))
    return tokens[days == days[idx[0]]].tolist() if len(idx) else []


def prepare():
    download()
    with zipfile.ZipFile(ROOT / "data/raw/data_en.zip") as z:
        with z.open("data_en/interactions.csv") as f:
            df = pd.read_csv(f)
    assert len(df) == 5_476_251
    assert not df.duplicated(["user_id", "item_id"]).any()
    df["day"] = pd.to_datetime(df.last_watch_dt)
    counts = df[df.day < "2021-07-12"].groupby("user_id").size()
    eligible = counts[counts >= 5].index.to_numpy()
    ids = np.sort(
        np.random.default_rng(42).choice(eligible, min(12000, len(eligible)), replace=False)
    )
    df = df[df.user_id.isin(ids)].sort_values(["user_id", "day", "item_id"])
    train = df[df.day < "2021-07-12"]
    freq = (
        train.item_id.value_counts()
        .rename_axis("item_id")
        .reset_index(name="count")
        .sort_values(["count", "item_id"], ascending=[False, True])
    )
    item_ids = freq.head(1000).item_id.tolist()
    mapping = {i: n + 2 for n, i in enumerate(item_ids)}
    vocab = len(mapping) + 2
    pop = np.zeros(vocab, dtype=np.float32)
    pop[2:] = freq.head(1000)["count"].to_numpy()
    novelty = -np.log((pop + 1) / (pop.sum() + vocab))
    novelty /= novelty[2:].max()
    items = pd.read_csv(ROOT / "data/raw/items.csv").set_index("item_id")
    metadata = []
    for i in item_ids:
        row = items.loc[i]
        metadata.append(
            {
                "token": mapping[i],
                "item_id": i,
                "title": str(row.title),
                "genres": str(row.genres).split(", "),
                "content_type": str(row.content_type),
            }
        )
    queries = {name: [] for name in ["dev", "tune", "test", "rl"]}
    splits = {
        "dev": ("2021-07-12", "2021-07-26"),
        "tune": ("2021-08-02", "2021-08-09"),
        "test": ("2021-08-09", "2021-08-23"),
        "rl": ("2021-07-26", "2021-07-26"),
    }
    examples = {"train": [], "dev": [], "test": []}
    excluded = {k: 0 for k in splits}
    for user, g in df.groupby("user_id", sort=True):
        days = g.day.to_numpy().astype("datetime64[D]")
        tokens = g.item_id.map(mapping).fillna(1).to_numpy(dtype=np.int64)
        pct = g.watched_pct.to_numpy(dtype=float)
        for name, (a, b) in splits.items():
            start, end = np.datetime64(a), np.datetime64(b)
            hist = tokens[days < start]
            if not len(hist):
                excluded[name] += 1
                continue
            # Unknown target IDs must remain distinct in metric denominators.
            raw_target = first_day_targets(days, g.item_id.to_numpy(), start, end)
            queries[name].append(
                {
                    "user_id": int(user),
                    "context": context(hist).tolist(),
                    "seen": sorted(set(int(t) for t in hist if t >= 2)),
                    "targets": sorted(set(mapping.get(i, -int(i) - 1) for i in raw_target)),
                }
            )
        for d in np.unique(days):
            hist = tokens[days < d]
            if not len(hist):
                continue
            split = (
                "train"
                if d < TRAIN_END
                else "dev"
                if d < np.datetime64("2021-07-26")
                else "test"
                if d >= np.datetime64("2021-08-09")
                else None
            )
            if split is None:
                continue
            after = days[(days > d) & (days <= d + np.timedelta64(7, "D"))]
            boundary = (
                TRAIN_END
                if split == "train"
                else np.datetime64("2021-07-26")
                if split == "dev"
                else np.datetime64("2021-08-23")
            )
            ret_mask = d + np.timedelta64(7, "D") < boundary
            for j in np.flatnonzero(days == d):
                if tokens[j] < 2 or not np.isfinite(pct[j]):
                    continue
                examples[split].append(
                    (
                        context(hist),
                        tokens[j],
                        np.clip(pct[j] / 100, 0, 1),
                        float(len(after) > 0),
                        float(ret_mask),
                        int(user),
                        str(d),
                        str(days[days < d].max()),
                    )
                )
    cache = ROOT / "data/processed"
    cache.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, rows in examples.items():
        total = len(rows)
        if name == "train" and total > 80000:
            take = np.sort(np.random.default_rng(42).choice(total, 80000, replace=False))
            rows = [rows[i] for i in take]
        x, y, r, c, m, u, d, h = zip(*rows)
        np.savez_compressed(
            cache / f"{name}.npz",
            x=np.stack(x),
            y=y,
            reward=r,
            continuation=c,
            cont_mask=m,
            user=u,
            day=d,
            hist_max=h,
        )
        assert all(a < b for a, b in zip(h, d))
        stats[name] = {
            "examples": len(rows),
            "before_cap": total,
            "reward_mean": float(np.mean(r)),
            "continuation_labeled": int(sum(m)),
        }
    for name, rows in queries.items():
        dump(cache / f"{name}_queries.json", rows)
    config = {
        "vocab_size": vocab,
        "item_ids": item_ids,
        "popularity": pop.tolist(),
        "novelty": novelty.tolist(),
        "metadata": metadata,
    }
    dump(cache / "catalog.json", config)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    dump(out / "catalog.json", config)
    pd.Series(ids, name="user_id").to_csv(out / "cohort.csv", index=False)
    manifest = {
        "source_revision": REVISION,
        "source_sha256": sha256(ROOT / "data/raw/data_en.zip"),
        "cohort_users": len(ids),
        "eligible_users": len(eligible),
        "cohort_rows": len(df),
        "train_rows": len(train),
        "examples": stats,
        "query_counts": {k: len(v) for k, v in queries.items()},
        "active_queries": {k: sum(bool(q["targets"]) for q in v) for k, v in queries.items()},
        "empty_histories": excluded,
        "vocab_size": vocab,
        "train_target_catalog_coverage": float(train.item_id.isin(item_ids).mean()),
        "max_context": MAXLEN,
        "tie_policy": "strictly earlier days; item_id tie-break within historical days",
    }
    dump(out / "data_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
