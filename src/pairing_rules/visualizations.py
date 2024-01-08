from copy import deepcopy
from math import pi

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def make_radio_chart(data: dict[float], title: str, subtitle: str, color: str) -> None:
    """
    Create a radar chart to visually represent multi-dimensional data.

    :param data: A dictionary where keys are categories and values are corresponding data points.
    :param title: The main title of the radar chart.
    :param subtitle: A subtitle to be included below the main title.
    :param color: The color of the radar chart.
    """
    data = deepcopy(data)
    weight = data["weight"]
    del data["weight"]

    categories = list(data.keys())
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 10))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1])

    ax = fig.add_subplot(gs[0], polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color="grey", size=11)

    ax.set_rlabel_position(0)
    plt.yticks(
        [0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=0
    )
    plt.ylim(0, 1)

    values = list(data.values())
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle="solid")
    ax.fill(angles, values, color=color, alpha=0.4)

    title_split = str(title).split(",")
    new_title = []
    for number, word in enumerate(title_split):
        if (number % 2) == 0 and number > 0:
            updated_word = "\n" + word.strip()
            new_title.append(updated_word)
        else:
            updated_word = word.strip()
            new_title.append(updated_word)
    new_title = ", ".join(new_title)

    title_incl_subtitle = new_title + "\n" + "(" + str(subtitle) + ")"

    plt.title(title_incl_subtitle, size=16, y=1.1)
    add_weight_line(gs, 1, weight, color)
    plt.tight_layout()


def add_weight_line(gs: GridSpec, n: int, value: float, color: str) -> None:
    """
    Add a reference line to a radar chart to indicate the weight of a specific feature.

    :param gs: The GridSpec specifying the layout of subplots in the radar chart.
    :param n: The position of the subplot in the radar chart where the weight line will be added.
    :param value: The value representing the weight of the feature. The line will be positioned accordingly.
    :param color: The color of the reference line and marker.
    """
    ax = plt.subplot(gs[n])
    ax.set_xlim(-1, 2)
    ax.set_ylim(0, 3)

    xmin = 0
    xmax = 1
    y = 1
    height = 0.2

    plt.hlines(y, xmin, xmax)
    plt.vlines(xmin, y - height / 2.0, y + height / 2.0)
    plt.vlines(xmax, y - height / 2.0, y + height / 2.0)

    plt.plot(value, y, "ko", ms=10, mfc=color)

    plt.text(
        xmin - 0.1,
        y,
        "Light-Bodied",
        horizontalalignment="right",
        fontsize=11,
        color="grey",
    )
    plt.text(
        xmax + 0.1,
        y,
        "Full-Bodied",
        horizontalalignment="left",
        fontsize=11,
        color="grey",
    )
    plt.axis("off")
