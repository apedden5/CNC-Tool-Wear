import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd
import math


def plot_time_all_experiments(experiments, feature_name, y_label, colour, title):
    
    rows = 3
    cols = 6

    plt.figure(figsize=(18, 10))
    plt.suptitle(title, fontsize=30, y=1.02)

    for i in range(len(experiments)):
        df = experiments[i]

        
        signal = df[feature_name].values
        
    
        time = np.arange(len(signal)) * 0.1   

        # Plot
        plt.subplot(rows, cols, i+1)
        plt.plot(time, signal, color=colour)

        plt.title(f"{feature_name} - Exp {i+1}", fontsize=10)
        plt.xlabel("Time (s)", fontsize=8)
        plt.ylabel(y_label, fontsize=8)
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_feedrate_clamp_joint(df, title="Feedrate vs Clamp Pressure by Tool Condition"):

    sns.set(style="whitegrid")

    g = sns.jointplot(
        data=df,
        x="feedrate",
        y="clamp_pressure",
        hue="tool_condition",
        palette="Set2",
        kind="scatter",
        height=8
    )

    
    g.plot_joint(sns.kdeplot, levels=5, alpha=0.4)

    
    plt.suptitle(title, fontsize=18, y=1.03)

    return g




def compare_worn_unworn_multi_feature(data_path, features, title=None):

    
    unworn_exps = [1, 2, 3, 4, 5, 6, 11, 12, 16]
    worn_exps   = [7, 8, 9, 10, 13, 14, 15, 17, 18]

    # Load all experiment files
    all_files = list(Path(data_path).glob("experiment_*_idd.csv"))

    df_worn_list = []
    df_unworn_list = []

    for f in all_files:
        df = pd.read_csv(f)
        exp_id = int(f.name.split("_")[1])

        if exp_id in unworn_exps:
            df_unworn_list.append(df)
        elif exp_id in worn_exps:
            df_worn_list.append(df)

    df_unworn = pd.concat(df_unworn_list, ignore_index=True)
    df_worn   = pd.concat(df_worn_list, ignore_index=True)

    # --- Plotting ---
    n = len(features)
    cols = 2  # you can change to 3 for 3-wide layout
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]

        unworn_mean = df_unworn[feature].mean()
        worn_mean = df_worn[feature].mean()

        ax.bar(["Unworn", "Worn"], [unworn_mean, worn_mean],
               color=["steelblue", "firebrick"])

        ax.set_title(feature, fontsize=12)
        ax.set_ylabel("Mean Value")
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots (if features % cols != 0)
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=18, y=1.02)

    plt.tight_layout()
    plt.show()