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


def process_and_save_experiments():
    base_dir = Path(__file__).resolve().parents[1]

    data_dir = base_dir / "data/data_raw"
    out_dir = base_dir / "data/data_id"

    os.makedirs(out_dir, exist_ok=True)

    meta = pd.read_csv(data_dir / "train.csv")
    meta = meta.rename(columns={"No": "experiment_id"})

    for i in range(1, 19):
        exp_file = data_dir / f"experiment_{i:02d}.csv"
        df = pd.read_csv(exp_file)

        tool_condition = meta.loc[
            meta["experiment_id"] == i, "tool_condition"
        ].values[0]

        df["experiment_id"] = i
        df["tool_condition"] = tool_condition

        out_name = out_dir / f"experiment_{i:02d}_idd.csv"
        df.to_csv(out_name, index=False)

    print("All experiments processed and saved successfully.")

process_and_save_experiments()