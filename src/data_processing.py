import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ------------------------
# Global config
# ------------------------
WINDOW_SIZE = 10
STRIDE      = 1  # step size for sliding windows

# experiment splits (adjust if you change them later)
TRAIN_EXPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 18]
VAL_EXPS   = [11, 13]
TEST_EXPS  = [12, 14, 16, 17]

TARGET_COL = "successful_part"


# ------------------------
# Helpers
# ------------------------
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _clear_dir(path: Path):
    if path.exists():
        for p in path.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------
# 1) Combine experiment data with metadata + drop cols
# ----------------------------------------------------
def experiment_encoding():
    base_dir = _project_root()

    data_dir = base_dir / "data" / "data_raw"
    out_dir  = base_dir / "data" / "data_id"
    _clear_dir(out_dir)

    meta = pd.read_csv(data_dir / "train.csv")
    meta = meta.rename(columns={"No": "experiment_id"})

    # columns to remove from raw experiments
    drop_cols = [
        "M1_CURRENT_PROGRAM_NUMBER",
        "M1_sequence_number",
        "M1_CURRENT_FEEDRATE",
        "Machining_Process",
    ]

    for i in range(1, 19):
        exp_file = data_dir / f"experiment_{i:02d}.csv"
        df = pd.read_csv(exp_file)

        # drop unwanted columns
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # grab metadata row for this experiment
        row = meta.loc[meta["experiment_id"] == i].iloc[0]

        # ---------- NEW TARGET: successful_part ----------
        # successful_part = 1 if machining_finalized == yes AND passed_visual_inspection == yes
        # otherwise 0 (includes unfinished or failed visual inspection)
        passed   = str(row["passed_visual_inspection"]).lower()
        finished = str(row["machining_finalized"]).lower()

        if finished == "yes" and passed == "yes":
            label = 1
        else:
            label = 0

        df[TARGET_COL] = label

        # keep tool_condition as a binary feature (worn=1, unworn=0)
        tc = str(row["tool_condition"]).lower()
        df["tool_condition"] = 1 if tc == "worn" else 0

        # metadata features
        df["experiment_id"] = i
        df["time_step"] = np.arange(len(df), dtype=int)

        out_name = out_dir / f"experiment_{i:02d}_idd.csv"
        df.to_csv(out_name, index=False)

    print("All experiments encoded with successful_part and tool_condition.")


# ----------------------------------------------------
# 2) Check missing values (sanity check only)
# ----------------------------------------------------
def checking_missing_values():
    base_dir = _project_root()
    data_dir = base_dir / "data" / "data_id"

    total_missing_all = 0
    for i in range(1, 19):
        exp_file = data_dir / f"experiment_{i:02d}_idd.csv"
        df = pd.read_csv(exp_file)
        total_missing_all += df.isnull().sum().sum()

    print(f"\nTotal missing values across all experiments (data_id): {total_missing_all}\n")


# ----------------------------------------------------
# 3) Ensure target is int, save cleaned CSVs
# ----------------------------------------------------
def categorical_encoding():
    base_dir = _project_root()

    data_dir = base_dir / "data" / "data_id"
    out_dir  = base_dir / "data" / "data_cleaned"
    _clear_dir(out_dir)

    for i in range(1, 19):
        exp_file = data_dir / f"experiment_{i:02d}_idd.csv"
        df = pd.read_csv(exp_file)

        # ensure successful_part is int 0/1
        df[TARGET_COL] = df[TARGET_COL].astype(int)

        # ensure tool_condition is int 0/1
        df["tool_condition"] = df["tool_condition"].astype(int)

        out_name = out_dir / f"experiment_{i:02d}_cleaned.csv"
        df.to_csv(out_name, index=False)

    print("All experiments cleaned and encoded (successful_part, tool_condition).")


# ----------------------------------------------------
# 4) Feature importance with LightGBM (on TRAIN_EXPS)
# ----------------------------------------------------
def compute_feature_importance():
    base_dir    = _project_root()
    cleaned_dir = base_dir / "data" / "data_cleaned"

    dfs = []
    for i in TRAIN_EXPS:
        f = cleaned_dir / f"experiment_{i:02d}_cleaned.csv"
        df = pd.read_csv(f)
        dfs.append(df)

    train_all = pd.concat(dfs, axis=0, ignore_index=True)

    y = train_all[TARGET_COL]

    drop_cols = [
        TARGET_COL,
        "experiment_id",
        "time_ms",
        "time_step",
    ]
    drop_cols = [c for c in drop_cols if c in train_all.columns]

    feature_cols = [c for c in train_all.columns if c not in drop_cols]
    X = train_all[feature_cols]

    lgb_train = lgb.Dataset(X, label=y)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbose": -1,
    }

    model = lgb.train(params, lgb_train, num_boost_round=300)

    importances = model.feature_importance()
    feat_imp = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return feat_imp


# ----------------------------------------------------
# 5) Keep only top-K features
# ----------------------------------------------------
def drop_features(top_k: int = 25):
    base_dir     = _project_root()
    cleaned_dir  = base_dir / "data" / "data_cleaned"
    filtered_dir = base_dir / "data" / "data_filtered"
    _clear_dir(filtered_dir)

    feat_imp = compute_feature_importance()
    top_features = feat_imp.head(top_k)["feature"].tolist()

    # always keep target + metadata columns if present
    keep_cols = top_features + [TARGET_COL, "experiment_id", "time_ms", "time_step"]

    for i in range(1, 19):
        f = cleaned_dir / f"experiment_{i:02d}_cleaned.csv"
        df = pd.read_csv(f)

        df_reduced = df[[c for c in keep_cols if c in df.columns]]

        out_path = filtered_dir / f"experiment_{i:02d}_filtered.csv"
        df_reduced.to_csv(out_path, index=False)

    print(f"All experiments reduced to top {top_k} features (plus target/metadata).")


# ----------------------------------------------------
# 6) Normalize filtered features (fit scaler on TRAIN_EXPS)
# ----------------------------------------------------
def normalize_filtered():
    base_dir     = _project_root()
    filtered_dir = base_dir / "data" / "data_filtered"
    norm_dir     = base_dir / "data" / "data_normalized"
    _clear_dir(norm_dir)

    # collect training experiments
    train_dfs = []
    for i in TRAIN_EXPS:
        f = filtered_dir / f"experiment_{i:02d}_filtered.csv"
        df = pd.read_csv(f)
        train_dfs.append(df)

    # find common columns across train exps
    common_cols = set(train_dfs[0].columns)
    for df in train_dfs[1:]:
        common_cols &= set(df.columns)

    drop_cols = {TARGET_COL, "experiment_id", "time_ms", "time_step"}
    feature_cols = [c for c in common_cols if c not in drop_cols]

    train_all = pd.concat([df[feature_cols] for df in train_dfs],
                          axis=0, ignore_index=True)

    scaler = StandardScaler()
    scaler.fit(train_all[feature_cols])

    # apply scaler to ALL experiments
    for i in range(1, 19):
        f = filtered_dir / f"experiment_{i:02d}_filtered.csv"
        df = pd.read_csv(f)

        df_scaled = df.copy()
        # only scale columns that we actually fitted
        cols_to_scale = [c for c in feature_cols if c in df.columns]
        df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])

        out_path = norm_dir / f"experiment_{i:02d}_normalized.csv"
        df_scaled.to_csv(out_path, index=False)

    print("All experiments normalized using training experiments' stats.")


# ----------------------------------------------------
# 7) Create windowed CSV datasets (train/val/test)
# ----------------------------------------------------
def _windows_to_csv_rows(exp_ids, norm_dir, window_size, feature_cols, target_col=TARGET_COL):
    rows = []

    for i in exp_ids:
        f = norm_dir / f"experiment_{i:02d}_normalized.csv"
        df = pd.read_csv(f)

        # sort by time if present
        if "time_ms" in df.columns:
            df = df.sort_values("time_ms")

        feats  = df[feature_cols].values
        labels = df[target_col].values
        n = len(df)

        # column names for flattened window
        win_feature_cols = [
            f"{feat}_t{t}"
            for t in range(window_size)
            for feat in feature_cols
        ]

        # sliding windows with stride
        for start in range(0, n - window_size + 1, STRIDE):
            end = start + window_size
            window_feats = feats[start:end, :]     # (W, F)
            flat = window_feats.reshape(-1)        # (W*F,)

            row = dict(zip(win_feature_cols, flat))
            row[target_col] = labels[end - 1]      # label at last step
            row["experiment_id"] = df["experiment_id"].iloc[end - 1]
            rows.append(row)

    return pd.DataFrame(rows)


def make_windowed_datasets():
    base_dir = _project_root()
    norm_dir = base_dir / "data" / "data_normalized"
    out_dir  = base_dir / "data" / "data_windowed_csv"
    _clear_dir(out_dir)

    # infer common feature columns across ALL normalized experiments
    dfs = []
    for i in range(1, 19):
        f = norm_dir / f"experiment_{i:02d}_normalized.csv"
        dfs.append(pd.read_csv(f))

    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)

    drop_cols = {TARGET_COL, "experiment_id", "time_ms", "time_step"}
    feature_cols = [c for c in dfs[0].columns if c in common_cols and c not in drop_cols]

    train_df = _windows_to_csv_rows(TRAIN_EXPS, norm_dir, WINDOW_SIZE, feature_cols)
    val_df   = _windows_to_csv_rows(VAL_EXPS,   norm_dir, WINDOW_SIZE, feature_cols)
    test_df  = _windows_to_csv_rows(TEST_EXPS,  norm_dir, WINDOW_SIZE, feature_cols)

    train_path = out_dir / f"train_windows_w{WINDOW_SIZE}.csv"
    val_path   = out_dir / f"val_windows_w{WINDOW_SIZE}.csv"
    test_path  = out_dir / f"test_windows_w{WINDOW_SIZE}.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Saved windowed datasets:")
    print("  ", train_path)
    print("  ", val_path)
    print("  ", test_path)


# ----------------------------------------------------
# Full pipeline
# ----------------------------------------------------
if __name__ == "__main__":
    experiment_encoding()
    checking_missing_values()
    categorical_encoding()
    drop_features(top_k=25)
    normalize_filtered()
    make_windowed_datasets()
