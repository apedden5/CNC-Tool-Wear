import pandas as pd
from pathlib import Path
import os


def load_experiments():
    experiments = []

    for i in range(1, 19):
        df = pd.read_csv(f"data_raw/experiment_{i:02d}.csv")
        experiments.append(df)
    dftrain = pd.read_csv(f"data_raw/train.csv")
    return experiments, dftrain

# The method to combine the experiement data with the experiment metadata.
def experiment_encoding():
    base_dir = Path(__file__).resolve().parents[1]

    data_dir = base_dir / "data/data_raw"
    out_dir = base_dir / "data/data_id"

    os.makedirs(out_dir, exist_ok=True)

    meta = pd.read_csv(data_dir / "train.csv")
    meta = meta.rename(columns={"No": "experiment_id"})

    for i in range(1, 19):
        exp_file = data_dir / f"experiment_{i:02d}.csv"
        df = pd.read_csv(exp_file)

        tool_condition = meta.loc[meta["experiment_id"] == i, "tool_condition"].values[0]
        feedrate = meta.loc[meta["experiment_id"] == i, "feedrate"].values[0]
        clamp_pressure = meta.loc[meta["experiment_id"] == i, "clamp_pressure"].values[0]

        df["experiment_id"] = i
        df["tool_condition"] = tool_condition
        df["feedrate"] = feedrate
        df["clamp_pressure"] = clamp_pressure

        out_name = out_dir / f"experiment_{i:02d}_idd.csv"
        df.to_csv(out_name, index=False)

    print("All experiments encoded successfully")

# The method to normalize, remove missing values, and smooth the overall data.

def checking_missing_values():
    base_dir = Path(__file__).resolve().parents[1]

    data_dir = base_dir / "data/data_id"
    out_dir = base_dir / "data/data_cleaned"
    os.makedirs(out_dir, exist_ok=True)

    ##### Checking for missing values across all experiments #####
    total_missing_all = 0  # accumulator

    for i in range(1, 19):
        exp_file = data_dir / f"experiment_{i:02d}_idd.csv"
        df = pd.read_csv(exp_file)

        # Count missing values for this experiment
        total_missing_all += df.isnull().sum().sum()

    # Final summary print
    print(f"\nTotal missing values across all experiments: {total_missing_all}\n")

def categorical_encoding():
    base_dir = Path(__file__).resolve().parents[1]

    data_dir = base_dir / "data/data_id"
    out_dir = base_dir / "data/data_cleaned"
    os.makedirs(out_dir, exist_ok=True)

    ##### Encoding the categorical variables (machine_process and tool_condition) #####
    for i in range(1, 19):
        exp_file = data_dir / f"experiment_{i:02d}_idd.csv"
        df = pd.read_csv(exp_file)

        # Convert 'tool_condition' to binary
        df['tool_condition'] = df['tool_condition'].astype(str).str.lower()
        df['tool_condition'] = df['tool_condition'].map({'worn': 1,'unworn': 0,}).astype(int)

        # One-hot encode 'machine_process'
        df = pd.get_dummies(df, columns=['Machining_Process'], drop_first=True)

        # Save cleaned data
        out_name = out_dir / f"experiment_{i:02d}_cleaned.csv"
        df.to_csv(out_name, index=False)

    print("All experiments cleaned successfully")

from pathlib import Path
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def plot_feature_importance():
    base_dir = Path(__file__).resolve().parents[1]

    cleaned_dir = base_dir / "data" / "data_cleaned"

    # ----- Choose training experiments -----
    train_exps = [1,2,3,4,5,6,7,8,9,10,11,12]

    dfs = []
    for i in train_exps:
        f = cleaned_dir / f"experiment_{i:02d}_cleaned.csv"
        df = pd.read_csv(f)
        dfs.append(df)

    train_all = pd.concat(dfs, axis=0, ignore_index=True)

    # ----- Separate features and label -----
    y = train_all["tool_condition"]

    drop_cols = ["tool_condition", "experiment_id", "time_ms"]
    feature_cols = [c for c in train_all.columns if c not in drop_cols]

    X = train_all[feature_cols]

    # ----- Train RF for feature importance -----
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    rf.fit(X, y)

    # ----- Create importance ranking -----
    importances = rf.feature_importances_
    feat_imp = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
    )

    # ----- Plot Top 25 in terminal -----
    top_k = 25
    top = feat_imp.head(top_k).iloc[::-1]  # reverse for barh

    plt.figure(figsize=(9, 10))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top Feature Importances (RandomForest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()     # <-- ONLY shows the graph in terminal

    print("Feature importance plot displayed.")
    return feat_imp

def drop_features():
    base_dir = Path(__file__).resolve().parents[1]

    cleaned_dir = base_dir / "data" / "data_cleaned"
    filtered_dir = base_dir / "data" / "data_filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    # ---- Get sorted feature importances ----
    feat_imp = plot_feature_importance()  # gets DF returned from earlier

    # ---- Extract TOP-25 feature names ----
    top25_features = feat_imp.head(25)["feature"].tolist()
    print("\nTop 25 selected features:")
    print(top25_features)

    # ---- Always keep label + experiment meta ----
    keep_cols = top25_features + ["tool_condition", "experiment_id", "time_ms"]

    # ---- Apply filtering to ALL experiments ----
    for i in range(1, 19):
        f = cleaned_dir / f"experiment_{i:02d}_cleaned.csv"
        df = pd.read_csv(f)

        # Keep only columns that actually exist in df
        df_reduced = df[[c for c in keep_cols if c in df.columns]]

        out_path = filtered_dir / f"experiment_{i:02d}_filtered.csv"
        df_reduced.to_csv(out_path, index=False)

        print(f"Experiment {i:02d} reduced → saved to {out_path.name}")

    print("\nAll experiments reduced to Top-25 features successfully.")

def normalization():
    base_dir = Path(__file__).resolve().parents[1]

    data_dir = base_dir / "data/data_cleaned"
    out_dir = base_dir / "data/data_normalized"
    os.makedirs(out_dir, exist_ok=True)

    
