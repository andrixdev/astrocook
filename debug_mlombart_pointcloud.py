import numpy as np, matplotlib.pyplot as plt

d = np.loadtxt(r"output\maximelombart\1-frame-amr\maximelombart-amr-0145.txt")
plt.style.use("dark_background")
fig, ax = plt.subplots(2, 3, figsize=(16, 10))
h, t = 50000, 1500
for r, c, v in zip(ax, [5, 4], ["v3", "v2"]):
    for a, (i, j, k, n) in zip(r, [(0, 1, 2, "XY"), (0, 2, 1, "XZ"), (1, 2, 0, "YZ")]):
        m = abs(d[:, k] - 5e5) < t
        sc = a.scatter(d[m, i], d[m, j], s=1, c=d[m, c], cmap="inferno", vmin=d[:, c].min(), vmax=d[:, c].max())
        a.set_xlim(5e5 - h, 5e5 + h); a.set_ylim(5e5 - h, 5e5 + h)
        a.set_aspect("equal"); a.set_title(f"{n} {v} ±{h} | |{'xyz'[k]}-5e5|<{t}")
    fig.colorbar(sc, ax=r, label=v)
plt.show()
