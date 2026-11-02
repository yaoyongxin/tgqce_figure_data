"""Fig. 4: measuring the embedding observables for B = 5, U_f = 1.5 under a trapped-ion noise model scaled by
lambda = 1 (upper row) and 1/3 (lower row).

    python plot_fig4.py        # writes Fig4.pdf

(a) CNOT count of the AVQDS circuit and of the compressed circuits; (b, e) double occupancy, (c, f) maximum element
error of the single-particle density matrix, (d, g) error of the double occupancy, each at 1e5 shots per time with
shot noise only, with the noise model, and with the noise model and mirror-circuit mitigation (median and 5-95%
quantiles of 500 repetitions). The AVQDS reference curve is read from ../fig3_B357_avqds/avqds_data/B5U1.5.dat.
"""
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GROUPS = ("shot", "noisy", "mitigated")

colors = np.asarray([[0, 0, 0], [230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66], [0, 114, 178],
                     [213, 94, 0], [204, 121, 167]]) / 255
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
COL = {"shot": "C3", "noisy": "C6", "mitigated": "C5"}
LAB = {"shot": "shot noise only", "noisy": "noise model", "mitigated": "mitigated"}


def panels(ax_d, ax_r, ax_e, D, fd, lam):
    """d estimates, max rho error and d error of one noise strength (rows of hw_data_*.dat)."""
    t = D[:, 0]
    col = {g: 5 + 9 * k for k, g in enumerate(GROUPS)}       # first column of each group: d median
    ax_d.plot(fd[:, 0], fd[:, 1], "-", color="C3", lw=1, label="AVQDS")
    for a, off in ((ax_d, 0), (ax_r, 3), (ax_e, 6)):
        for k, g in enumerate(GROUPS):
            c = col[g] + off
            dx = (k - 1) * 0.08
            a.errorbar(t + dx, D[:, c], yerr=[D[:, c] - D[:, c + 1], D[:, c + 2] - D[:, c]], fmt="o", ms=3, lw=1,
                       capsize=1.5, color=COL[g], label=LAB[g] if a is ax_d else None)
    ax_d.set_ylim(0.08, 0.26)
    ax_d.text(0.96, 0.95, rf"$\lambda={lam}$", transform=ax_d.transAxes, ha="right", va="top")
    ax_d.set_ylabel(r"$\langle\uparrow\downarrow\rangle$")
    ax_r.set_ylabel(r"$\max_{ab}|\rho_{ab}-\rho^\mathrm{AVQDS}_{ab}|$", fontsize="small", labelpad=2)
    ax_e.set_ylabel(r"$|\langle\uparrow\downarrow\rangle-\langle\uparrow\downarrow\rangle_\mathrm{AVQDS}|$", fontsize="small",
                    labelpad=2)
    for a, ym in ((ax_r, 0.052), (ax_e, 0.032)):
        a.axhline(0.01, color="k", ls="--", lw=0.8)
        a.set_ylim(0, ym)


def main():
    sets = (("hw_data_lambda1.dat", "1"), ("hw_data_lambda1over3.dat", "1/3"))
    fd = np.loadtxt(HERE.parent / "fig3_B357_avqds" / "avqds_data" / "B5U1.5.dat")   # t, d_AVQDS, N_cx, ...
    fig = plt.figure(figsize=(7, 4.3))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1])
    a = fig.add_subplot(gs[:, 0])
    D = np.loadtxt(HERE / sets[0][0])
    a.plot(fd[:, 0], fd[:, 2], "-", color="C3", label="AVQDS")
    a.plot(D[:, 0], D[:, 2], "s", color="C5", ms=4, label="compressed")
    a.set_yscale("log")
    a.set_ylabel(r"$N_\mathrm{cx}$")
    a.set_ylim(50, 3000)
    a.set_xlabel("$t$")
    a.set_xticks([0, 1, 2, 3, 4, 5])
    a.set_xlim(-0.1, 5.3)
    a.legend(fontsize="x-small", frameon=False, loc="center right")
    a.text(0.04, 0.985, "(a)", transform=a.transAxes, va="top")
    labels = iter("bcdefg")
    for r, (name, lam) in enumerate(sets):
        D = np.loadtxt(HERE / name)
        axs = [fig.add_subplot(gs[r, k]) for k in (1, 2, 3)]
        panels(*axs, D, fd, lam)
        for x in axs:
            x.set_xticks([0, 1, 2, 3, 4, 5])
            x.set_xlim(-0.1, 5.3)
            if r == 1:
                x.set_xlabel("$t$")
            else:
                x.set_xticklabels([])
            x.text(0.04, 0.97, f"({next(labels)})", transform=x.transAxes, va="top")
    h, l = fig.axes[1].get_legend_handles_labels()
    fig.axes[5].legend(h[1:], l[1:], loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize="x-small", frameon=False,
                       handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout(w_pad=0.6, h_pad=0.4)
    fig.savefig(HERE / "Fig4.pdf")


if __name__ == "__main__":
    main()
