import numpy as np
from pylab import plt

def make_model_dict(name, model, model_class):
    return {
        "name": name,
        "model": model,
        "threshold": -1,
        "auc": -1,
        "maxsss": -1,
        "frozen model": None,
        "class": model_class
    }

def plotit(x, title, colors, cmap="Blues"):
    x = np.asarray(x)
    unique_vals = [0, 1, -9999]
    n_val = len(unique_vals)
    base = plt.get_cmap(cmap, n_val)
    plt.imshow(x, cmap=cmap, interpolation='nearest')
    
    x_mapped = np.full(x.shape, np.nan)

    for i, val in enumerate(unique_vals):
        x_mapped[x == val] = i
    
    discrete_cmap = ListedColormap(colors)

    bounds = np.arange(-0.5, n_val + 0.5, 1)
    norm = BoundaryNorm(bounds, discrete_cmap.N)

    im = plt.imshow(
        x_mapped,
        cmap=discrete_cmap,
        norm=norm,
        interpolation="nearest"
    )

    cbar = plt.colorbar(im, ticks=np.arange(n_val))
    cbar.ax.set_yticklabels(unique_vals)

    plt.title(title, fontweight="bold")
    plt.savefig(title, dpi=300, bbox_inches='tight')
    plt.show()