#!/usr/bin/env python3
"""Train candidate classifiers from a CSV file of precomputed features."""

import argparse
import csv

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def read_candidates(path, features):
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"group_id", "label", *features}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("input has no candidate rows")
    groups, labels, values = [], [], []
    for row in rows:
        if not row["group_id"]:
            raise ValueError("group_id must not be empty")
        if row["label"] not in ("0", "1"):
            raise ValueError("label must be 0 or 1")
        groups.append(row["group_id"])
        labels.append(int(row["label"]))
        values.append([float(row[name]) if row[name].strip() else np.nan for name in features])
    x = np.asarray(values, dtype=np.float64)
    if np.isinf(x).any():
        raise ValueError("feature values must be finite or missing")
    return x, np.asarray(labels), np.asarray(groups)


def grouped_split(x, y, groups, seed):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val, test = next(outer.split(x, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed + 1)
    train, validation = next(inner.split(x[train_val], y[train_val], groups[train_val]))
    return train_val[train], train_val[validation], test


def report(name, model, x, y, splits):
    print(f"\n{name}")
    for split_name, indices in zip(("train", "validation", "test"), splits):
        scores = model.predict_proba(x[indices])[:, 1]
        print(f"  {split_name}: AUROC={roc_auc_score(y[indices], scores):.4f} "
              f"average_precision={average_precision_score(y[indices], scores):.4f}")
    _, validation, test = splits
    validation_scores = model.predict_proba(x[validation])[:, 1]
    count = max(1, int(np.ceil(len(validation_scores) * 0.10)))
    threshold = np.sort(validation_scores)[-count]
    admitted = model.predict_proba(x[test])[:, 1] >= threshold
    recall = y[test][admitted].sum() / y[test].sum()
    print(f"  test at validation 10% threshold: admission={admitted.mean():.4f} "
          f"positive_recall={recall:.4f} threshold={threshold:.6g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV with group_id, label, and numeric feature columns")
    parser.add_argument("--features", nargs="+", required=True, help="feature column names")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(set(args.features)) != len(args.features) or {"group_id", "label"} & set(args.features):
        parser.error("features must be distinct and exclude group_id and label")

    x, y, groups = read_candidates(args.csv, args.features)
    splits = grouped_split(x, y, groups, args.seed)
    for name, indices in zip(("train", "validation", "test"), splits):
        if np.unique(y[indices]).size != 2:
            raise ValueError(f"{name} split must contain both labels")
        print(f"{name}: {len(indices)} candidates, {len(np.unique(groups[indices]))} groups")

    train, _, _ = splits
    positives = int(y[train].sum())
    models = {
        "Logistic regression": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=1.0, objective="binary:logistic", eval_metric="aucpr",
            scale_pos_weight=(len(train) - positives) / positives,
            tree_method="hist", random_state=args.seed, n_jobs=-1,
        ),
    }
    for name, model in models.items():
        model.fit(x[train], y[train])
        report(name, model, x, y, splits)


if __name__ == "__main__":
    main()
