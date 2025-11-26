import pandas as pd

def load_experiments():
    experiments = []

    for i in range(1, 19):
        df = pd.read_csv(f"data/experiment_{i:02d}.csv")
        experiments.append(df)
    dftrain = pd.read_csv(f"data/train.csv")
    return experiments, dftrain