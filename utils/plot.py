import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot_histograms(data,
                    target, target_figsize,
                    dependent_layout, dependent_figsize,
                    include_boxplots=False, boxplot_figsize=None):
    print(f"Distribution of target {target} and dependent variables:")
    data[target].hist(figsize=target_figsize, grid=False).set_title(target)
    data.drop([target], axis=1).hist(layout=dependent_layout, figsize=dependent_figsize, sharey=True, grid=False)

    if include_boxplots:
        data.plot(kind='box', subplots=True, figsize=boxplot_figsize);
        
    plt.tight_layout();


def plot_scores(scores_dict):
    fig, axes = plt.subplots(len(scores_dict.keys()), 1,
                    figsize=(5, len(scores_dict.keys())*3))
    fig.tight_layout(pad=5)
    for i, (function, scores) in enumerate(scores_dict.items()):
        try:
            ax = axes[i]
        except TypeError:
            ax = axes
        ax.text(0.02, 0,
    f"""
    Mean: {round(np.mean(scores), 2)}
    Std: {round(np.std(scores), 2)}
    Max: {round(np.max(scores), 2)}
    75th Percentile: {round(np.percentile(scores, 75), 2)}
    Median: {round(np.median(scores), 2)}
    25th Percentile: {round(np.percentile(scores, 25), 2)}
    Min: {round(np.min(scores), 2)}
    """)
        sns.distplot(scores, ax=ax)
        ax.set_title(function)
        ax.set_xlabel("Accuracy Score")
        ax.set_xbound(0, 1)