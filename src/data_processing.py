
import os
import pandas as pd
from pathlib import Path

def load_experiments():
    # Resolve project root:    CNC-Tool-Wear/
    base = Path(__file__).resolve().parent.parent

    # Path to raw data:        CNC-Tool-Wear/data/data_raw/
    data_path = base / "data" / "data_raw"

    experiments = []
    for i in range(1, 19):
        fp = data_path / f"experiment_{i:02d}.csv"
        experiments.append(pd.read_csv(fp))

    dftrain = pd.read_csv(data_path / "train.csv")

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
def data_cleaning():
    base_dir = Path(__file__).resolve().parents[1]

    data_dir = base_dir / "data/data_id"
    out_dir = base_dir / "data/data_cleaned"

    os.makedirs(out_dir, exist_ok=True)

    # Using linear interpolation to fill missing values
    for i in range(1, 19):
        exp_file = data_dir / f"experiment_{i:02d}_idd.csv"
        df = pd.read_csv(exp_file)

        df = df.interpolate(method="linear", limit_direction="both")
