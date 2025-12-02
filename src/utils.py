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

def plot_feedrate_clamp_joint_successful(df, title="Feedrate vs Clamp Pressure by Successful Part"):

    sns.set(style="whitegrid")

    
    df = df.copy()
    df["successful_label"] = df["successful_part"].map({1: "successful", 0: "not successful"})

    g = sns.jointplot(
        data=df,
        x="feedrate",
        y="clamp_pressure",
        hue="successful_label",      
        palette="Set1",
        kind="scatter",
        height=8
    )

    g.plot_joint(sns.kdeplot, levels=5, alpha=0.4)

    plt.suptitle(title, fontsize=18, y=1.03)

    return g



def compare_four_group_multi_feature(data_path, features, title=None):


   
    all_files = list(Path(data_path).glob("experiment_*_cleaned.csv"))
    dfs = [pd.read_csv(f) for f in all_files]
    df = pd.concat(dfs, ignore_index=True)

   
    unworn_df = df[df["tool_condition"] == 0]
    worn_df   = df[df["tool_condition"] == 1]
    success_df = df[df["successful_part"] == 1]
    fail_df    = df[df["successful_part"] == 0]

    
    n = len(features)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]

        
        vals = [
            unworn_df[feature].mean(),
            worn_df[feature].mean(),
            success_df[feature].mean(),
            fail_df[feature].mean()
        ]

        labels = ["Unworn", "Worn", "Successful", "Not Successful"]
        colors = ["steelblue", "firebrick", "green", "orange"]

        ax.bar(labels, vals, color=colors)
        ax.set_title(feature, fontsize=12)
        ax.set_ylabel("Mean Value")
        ax.grid(axis="y", alpha=0.3)

    
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=18, y=1.02)

    plt.tight_layout()
    plt.show()

def plot_feature_importance(feat_imp, top_n=30):
    
    df = feat_imp.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 10))
    plt.barh(df["feature"], df["importance"], color="purple")
    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.title("Feature Importance", fontsize=18)
    plt.tight_layout()
    plt.show()
