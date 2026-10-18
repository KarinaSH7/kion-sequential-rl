"""Data chronology and independently recomputed ranking metrics."""

import json
import zipfile
import numpy as np
import pandas as pd
from .data import ROOT
from .source import dump
from .evaluate import load_queries


def run():
    checks = {}
    bounds = {
        "train": (None, "2021-07-12"),
        "dev": ("2021-07-12", "2021-07-26"),
        "test": ("2021-08-09", "2021-08-23"),
    }
    for split, (start, end) in bounds.items():
        d = np.load(ROOT / f"data/processed/{split}.npz")
        assert (d["hist_max"] < d["day"]).all()
        assert (d["day"] < end).all()
        if start:
            assert (d["day"] >= start).all()
        valid = d["cont_mask"].astype(bool)
        assert (
            (d["day"].astype("datetime64[D]") + np.timedelta64(7, "D"))[valid] < np.datetime64(end)
        ).all()
        assert ((d["reward"] >= 0) & (d["reward"] <= 1)).all()
        checks[split] = {
            "examples": len(d["y"]),
            "strict_history": True,
            "continuation_censoring": True,
        }
    # Rebuild fixed-cutoff contexts from source rows, not from saved metadata.
    with zipfile.ZipFile(ROOT / "data/raw/data_en.zip") as archive:
        with archive.open("data_en/interactions.csv") as stream:
            raw = pd.read_csv(stream, usecols=["user_id", "item_id", "last_watch_dt"])
    ids = pd.read_csv(ROOT / "results/cohort.csv").user_id
    raw = raw[raw.user_id.isin(ids)].sort_values(["user_id", "last_watch_dt", "item_id"])
    catalog = json.loads((ROOT / "results/catalog.json").read_text())
    mapping = {item: i + 2 for i, item in enumerate(catalog["item_ids"])}
    cutoffs = {"dev": "2021-07-12", "rl": "2021-07-26", "tune": "2021-08-02", "test": "2021-08-09"}
    source_groups = {int(user): g for user, g in raw.groupby("user_id")}
    for split, cutoff in cutoffs.items():
        for q in load_queries(split):
            g = source_groups[q["user_id"]]
            history = (
                g.loc[g.last_watch_dt < cutoff, "item_id"]
                .map(mapping)
                .fillna(1)
                .to_numpy(dtype=int)
            )
            expected = np.zeros(32, dtype=int)
            expected[-min(32, len(history)) :] = history[-32:]
            assert np.array_equal(expected, q["context"])
            assert set(q["seen"]) == set(history[history >= 2])
            ends = {
                "dev": "2021-07-26",
                "rl": "2021-07-26",
                "tune": "2021-08-09",
                "test": "2021-08-23",
            }
            future = g[(g.last_watch_dt >= cutoff) & (g.last_watch_dt < ends[split])]
            first = future[future.last_watch_dt == future.last_watch_dt.min()]
            expected_targets = {mapping.get(int(i), -int(i) - 1) for i in first.item_id}
            assert set(q["targets"]) == expected_targets

        checks[split + "_source_contexts"] = True
    # Independent metric implementation on stored final rankings, if available.
    queries = load_queries("test")
    checked = []
    for path in (ROOT / "results/predictions").glob("*_top10.npz"):
        predictions = np.load(path)["top10"]
        reported = pd.read_csv(str(path).replace("_top10.npz", "_users.csv"))
        assert np.array_equal(reported.user_id, [q["user_id"] for q in queries])
        values = []
        for ranking, q in zip(predictions, queries):
            target = set(q["targets"])
            dcg = 0.0
            hits = 0
            for rank, item in enumerate(ranking, 1):
                if item in target:
                    hits += 1
                    dcg += 1 / np.log2(rank + 1)
            ideal = sum(1 / np.log2(rank + 1) for rank in range(1, min(10, len(target)) + 1))
            values.append((hits / max(1, len(target)), dcg / ideal if ideal else 0))
        assert np.allclose(np.array(values), reported[["recall10", "ndcg10"]].values, atol=1e-12)
        checked.append(path.name)
    checks["independent_metric_files"] = checked
    dump(ROOT / "results/audit.json", checks)
    print(checks)


if __name__ == "__main__":
    run()
