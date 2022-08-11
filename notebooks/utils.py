from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def sample_counter(counts: Counter, size: int) -> Counter:
    words, counts = zip(*counts.items())
    population = np.repeat(words, counts)
    return Counter(np.random.choice(population, size=size, replace=False))


def tsplot(
    y,
    x=None,
    n=50,
    percentile_min=1,
    percentile_max=99,
    color="C0",
    location="median",
    line_color="k",
    axis=0,
    ax=None,
    label=None,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()
    # calculate the lower and upper percentile groups, skipping 50 percentile
    perc1 = np.nanpercentile(
        y, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=axis
    )
    perc2 = np.nanpercentile(
        y, np.linspace(50, percentile_max, num=n + 1)[1:], axis=axis
    )

    if x is None:
        x = np.arange(y.shape[1])

    if "alpha" in kwargs:
        alpha = kwargs.pop("alpha")
    else:
        alpha = 1 / n
    # fill lower and upper percentile groups
    for p1, p2 in zip(perc1, perc2):
        ax.fill_between(x, p1, p2, alpha=alpha, color=color, edgecolor=None)

    if location == "mean":
        ax.plot(x, np.nanmean(y, axis=axis), color=line_color, label=label)
    elif location == "median":
        ax.plot(x, np.nanmedian(y, axis=axis), color=line_color, label=label)
    else:
        raise ValueError(f"Location `{location}` is not supported.")

    return ax
