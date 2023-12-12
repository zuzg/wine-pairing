from copy import deepcopy
from math import pi

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def make_radio_chart(data: dict[float], title: str, subtitle: str, color: str) -> None:
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
    """Add a line to the radio chart to indicate the weight of the food."""
    ax = plt.subplot(gs[n])
    ax.set_xlim(-1, 2)
    ax.set_ylim(0, 3)

    xmin = 0
    xmax = 1
    y = 1
    height = 0.2

    # draw a lines (horizontal & vertical)
    plt.hlines(y, xmin, xmax)
    plt.vlines(xmin, y - height / 2.0, y + height / 2.0)
    plt.vlines(xmax, y - height / 2.0, y + height / 2.0)

    # draw a point on the line
    px = value
    plt.plot(px, y, "ko", ms=10, mfc=color)

    # add text
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
