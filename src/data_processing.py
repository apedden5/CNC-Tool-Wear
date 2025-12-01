import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Global Variables
WINDOW_SIZE = 10
STRIDE      = 1

# Experiment splits to make the training, testing, and validation sets
TRAIN_EXPS = [1, 2, 3, 11, 12, 13, 4, 5, 6, 7]
VAL_EXPS   = [14, 15, 8, 9]
TEST_EXPS  = [17, 18, 10, 16]

# Global definition for our target variable.
TARGET_COL = "successful_part"

# Clears any past data files before running pre-processing
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


# A method to combine experiment data with metadata + drop cols
def experiment_encoding():
    base_dir = _project_root()

    data_dir = base_dir / "data" / "data_raw"
    out_dir  = base_dir / "data" / "data_id"
    _clear_dir(out_dir)

    meta = pd.read_csv(data_dir / "train.csv")
    meta = meta.rename(columns={"No": "experiment_id"})

    # Remove these columns from the experiment data as they are seen as unnescasary information.
    drop_cols = [
        "M1_CURRENT_PROGRAM_NUMBER",
        "M1_sequence_number",
        "M1_CURRENT_FEEDRATE",
        "Machining_Process",
    ]

    for i in range(1, 19):
        exp_file = data_dir / f"experiment_{i:02d}.csv"
        df = pd.read_csv(exp_file)

        # Removing the unwanted columns specified above.
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # Specifying the data from the target.csv that corresponds with the experiment.csv
        row = meta.loc[meta["experiment_id"] == i].iloc[0]

        # Data Engineering for the target variable "successful_part"
        # successful_part = 1 if machining_finalized == yes AND passed_visual_inspection == yes
        # otherwise 0 (includes unfinished or failed visual inspection)
        passed   = str(row["passed_visual_inspection"]).lower()
        finished = str(row["machining_finalized"]).lower()

        if finished == "yes" and passed == "yes":
            label = 1
        else:
            label = 0

        df[TARGET_COL] = label

        # Converting "tool_worn" to be binary for the model to train.
        tc = str(row["tool_condition"]).lower()
        df["tool_condition"] = 1 if tc == "worn" else 0

        # Adding time_step and experiment_id to each row of the experiment.csv
        df["experiment_id"] = i
        df["time_step"] = np.arange(len(df), dtype=int)

        out_name = out_dir / f"experiment_{i:02d}_idd.csv"
        df.to_csv(out_name, index=False)

    print("All experiments encoded with successful_part and tool_condition.")


# A method to ensure there are no missing values in the dataset
# In this case we already knew there were no missing values, this simply checks and confirms
# this assumption prior to completing more pre-processing of the dataset.
def checking_missing_values():
    base_dir = _project_root()
    data_dir = base_dir / "data" / "data_id"

    total_missing_all = 0
    for i in range(1, 19):
        exp_file = data_dir / f"experiment_{i:02d}_idd.csv"
        df = pd.read_csv(exp_file)
        total_missing_all += df.isnull().sum().sum()

    print(f"\nTotal missing values across all experiments (data_id): {total_missing_all}\n")


# This method ensures that the binary categorical variables are stored as integers for the training models.
# If they are not, we convert them to be simple binary.
def categorical_encoding():
    base_dir = _project_root()

    data_dir = base_dir / "data" / "data_id"
    out_dir  = base_dir / "data" / "data_cleaned"
    _clear_dir(out_dir)

    for i in range(1, 19):
        exp_file = data_dir / f"experiment_{i:02d}_idd.csv"
        df = pd.read_csv(exp_file)

        # Ensure successful_part is int 0/1
        df[TARGET_COL] = df[TARGET_COL].astype(int)

        # Ensure tool_condition is int 0/1
        df["tool_condition"] = df["tool_condition"].astype(int)

        out_name = out_dir / f"experiment_{i:02d}_cleaned.csv"
        df.to_csv(out_name, index=False)

    print("All experiments cleaned and encoded (successful_part, tool_condition).")

# A method to complete rudimentary feature importance of the dataset to remove
# parameters which have little impact on the target variable.
def compute_feature_importance():
    base_dir    = _project_root()
    cleaned_dir = base_dir / "data" / "data_cleaned"

    # To determine feature importance, we concatenate all training experiments into one df
    # to determine their effect.
    dfs = []
    for i in TRAIN_EXPS:
        f = cleaned_dir / f"experiment_{i:02d}_cleaned.csv"
        df = pd.read_csv(f)
        dfs.append(df)

    train_all = pd.concat(dfs, axis=0, ignore_index=True)

    y = train_all[TARGET_COL]   # defining the target variable (successful_part)

    # dropping any features which may leak info regarding the outcome of the experiment.
    drop_cols = [
        TARGET_COL,
        "experiment_id",
        "time_ms",
        "time_step",
    ]
    drop_cols = [c for c in drop_cols if c in train_all.columns]

    feature_cols = [c for c in train_all.columns if c not in drop_cols]
    X = train_all[feature_cols]

    lgb_train = lgb.Dataset(X, label=y)     # using lightgbm to quickly determine which features are important, we then train the models based on this outcome.

    # Defining the parameters for the lgb model.
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

    # Training the lgb model to determine feature importance.
    model = lgb.train(params, lgb_train, num_boost_round=300)

    importances = model.feature_importance()

    # Returns a dataframe of important features in order from most to least to be used for later pre-processing.
    feat_imp = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return feat_imp

# A method to keep only the top k features according to the feature selection method (compute_feature_importance)
def drop_features(top_k: int = 25):
    base_dir     = _project_root()
    cleaned_dir  = base_dir / "data" / "data_cleaned"
    filtered_dir = base_dir / "data" / "data_filtered"
    _clear_dir(filtered_dir)

    # First call the method compute_feature_importance to determine the top features of the dataset.
    feat_imp = compute_feature_importance()
    top_features = feat_imp.head(top_k)["feature"].tolist()

    # Keep the top features as well as the metadata from train.csv for later identification of rows and time steps.
    keep_cols = top_features + [TARGET_COL, "experiment_id", "time_ms", "time_step"]

    # For each experiment, reduce the dolumns to only the important features and metadata.
    for i in range(1, 19):
        f = cleaned_dir / f"experiment_{i:02d}_cleaned.csv"
        df = pd.read_csv(f)

        df_reduced = df[[c for c in keep_cols if c in df.columns]]

        out_path = filtered_dir / f"experiment_{i:02d}_filtered.csv"
        df_reduced.to_csv(out_path, index=False)

    print(f"All experiments reduced to top {top_k} features (plus target/metadata).")

# A method to normalize the dataset according to a normalization based on only the training data (no data leakage with validation or test).
def normalize_filtered():
    base_dir     = _project_root()
    filtered_dir = base_dir / "data" / "data_filtered"
    norm_dir     = base_dir / "data" / "data_normalized"
    _clear_dir(norm_dir)

    # Collect all training experiments in one dataframe to compute standard scaler.
    train_dfs = []
    for i in TRAIN_EXPS:
        f = filtered_dir / f"experiment_{i:02d}_filtered.csv"
        df = pd.read_csv(f)
        train_dfs.append(df)

    # Intersect all collumns from the training experiements (combine datasets)
    common_cols = set(train_dfs[0].columns)
    for df in train_dfs[1:]:
        common_cols &= set(df.columns)

    # Remove columns which do not need to be normalized (categorical/time specific)
    drop_cols = {TARGET_COL, "experiment_id", "time_ms", "time_step"}
    feature_cols = [c for c in common_cols if c not in drop_cols]

    train_all = pd.concat([df[feature_cols] for df in train_dfs],
                          axis=0, ignore_index=True)

    # Fit standard scaler to ONLY TRAINING data feature collumns.
    scaler = StandardScaler()
    scaler.fit(train_all[feature_cols])

    # Apply the scaler to all experiments including validation and test sets.
    for i in range(1, 19):
        f = filtered_dir / f"experiment_{i:02d}_filtered.csv"
        df = pd.read_csv(f)

        df_scaled = df.copy()

        # Only applies scale to features we are going to be using for training (based on feature importance).
        cols_to_scale = [c for c in feature_cols if c in df.columns]
        df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])

        out_path = norm_dir / f"experiment_{i:02d}_normalized.csv"
        df_scaled.to_csv(out_path, index=False)

    print("All experiments normalized using training experiments' stats.")


# A method to create windowed CSV datasets for each experiment.
def _windows_to_csv_rows(exp_ids, norm_dir, window_size, feature_cols, target_col=TARGET_COL):
    rows = []

    # For each experiment, window the timesteps.
    for i in exp_ids:
        f = norm_dir / f"experiment_{i:02d}_normalized.csv"
        df = pd.read_csv(f)

        # Ensure the time steps are still in order (important for time series training).
        if "time_ms" in df.columns:
            df = df.sort_values("time_ms")

        feats  = df[feature_cols].values
        labels = df[target_col].values
        n = len(df)

        # Column names for flattened window
        win_feature_cols = [
            f"{feat}_t{t}"
            for t in range(window_size)
            for feat in feature_cols
        ]

        # Creates sliding window frames across the entire experiment appending the 
        # target label to the end of every window alongside experiment_id. 
        # experiment_id is added to make sure there is no windows that cross between experiements. 
        for start in range(0, n - window_size + 1, STRIDE):
            end = start + window_size
            window_feats = feats[start:end, :]     # (W, F)
            flat = window_feats.reshape(-1)        # (W*F,)

            row = dict(zip(win_feature_cols, flat))
            row[target_col] = labels[end - 1]      # label at last step
            row["experiment_id"] = df["experiment_id"].iloc[end - 1]
            rows.append(row)

    return pd.DataFrame(rows)

# A method to combine the windowed experiments into single train, test, and validation datasets.
def make_windowed_datasets():
    base_dir = _project_root()
    norm_dir = base_dir / "data" / "data_normalized"
    out_dir  = base_dir / "data" / "data_windowed_csv"
    _clear_dir(out_dir)

    # Loading all normalized experiments to determine the intersecting features (ones defined as having high importance)
    dfs = []
    for i in range(1, 19):
        f = norm_dir / f"experiment_{i:02d}_normalized.csv"
        dfs.append(pd.read_csv(f))

    # Computing the intersection of these experiements to determine the common collumns that will be in our final datasets.
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)

    # Removing labels and metadata from the feature set.
    drop_cols = {TARGET_COL, "experiment_id", "time_ms", "time_step"}
    feature_cols = [c for c in dfs[0].columns if c in common_cols and c not in drop_cols]

    # Creating the windowed rows for the train, test, and validation test sets.
    train_df = _windows_to_csv_rows(TRAIN_EXPS, norm_dir, WINDOW_SIZE, feature_cols)
    val_df   = _windows_to_csv_rows(VAL_EXPS,   norm_dir, WINDOW_SIZE, feature_cols)
    test_df  = _windows_to_csv_rows(TEST_EXPS,  norm_dir, WINDOW_SIZE, feature_cols)

    # Defining the output file paths for these datasets.
    train_path = out_dir / f"train_windows_w{WINDOW_SIZE}.csv"
    val_path   = out_dir / f"val_windows_w{WINDOW_SIZE}.csv"
    test_path  = out_dir / f"test_windows_w{WINDOW_SIZE}.csv"

    # Saving the final windowed data to the output file paths where they will be accessed by each deep learning model.
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Saved windowed datasets:")
    print("  ", train_path)
    print("  ", val_path)
    print("  ", test_path)

# Full pipeline to pre-process the data.
if __name__ == "__main__":
    experiment_encoding()
    checking_missing_values()
    categorical_encoding()
    drop_features(top_k=25)
    normalize_filtered()
    make_windowed_datasets()
