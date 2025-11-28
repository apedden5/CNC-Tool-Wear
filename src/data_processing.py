import pandas as pd
from pathlib import Path
import os


def load_experiments():
    experiments = []

    for i in range(1, 19):
        df = pd.read_csv(f"data/experiment_{i:02d}.csv")
        experiments.append(df)
    dftrain = pd.read_csv(f"data/train.csv")
    return experiments, dftrain

# The method to encode experiments with IDs and tool conditions
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

