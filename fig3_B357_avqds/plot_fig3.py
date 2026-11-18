"""Fig. 3: t-GQCE with B = 3, 5, 7 bath orbitals per spin, embedding states propagated by exact diagonalization
(lines) and by AVQDS (circles), compared with B = 1 and nonequilibrium DMFT.

    python plot_fig3.py        # writes Fig3.pdf

Rows: quench U: 0.05 -> 1.5 (weak) and 0.05 -> 2.5 (strong).
Columns: double occupancy, infidelity 1 - F of the AVQDS state with the spin-up/spin-down entanglement entropy of the
exact embedding state as inset, and the CNOT count N_cx of the AVQDS circuit.
"""
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=np.asarray([
    [0, 0, 0], [230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66], [0, 114, 178], [213, 94, 0],
    [204, 121, 167]]) / 255)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# double occupancy d(t): td-gGA with exact embedding states (B = 1, 3, 5, 7) and nonequilibrium DMFT
data = {f.stem: np.loadtxt(f) for f in sorted(HERE.glob("dt_u*.dat"))}

# AVQDS: t, d_AVQDS, N_cx, n_gates, 1-F, S_imp, S_updn
avqds, ncnot, infid, sud = {}, {}, {}, {}
for b in (3, 5, 7):
    for u in ("1.5", "2.5"):
        t, d, ncx, ng, inf, s, su = np.loadtxt(HERE / "avqds_data" / f"B{b}U{u}.dat", unpack=True)
        avqds[(b, u)] = (t, d)
        ncnot[(b, u)] = ncx
        infid[(b, u)] = (t, np.where(inf > 1e-10, inf, np.nan))   # round-off values masked for the log scale
        sud[(b, u)] = (t, su)


def thin(t, dt=0.1):
    """Indices of points spaced at least dt apart in time (for marker plotting)."""
    idx, last = [], -np.inf
    for i, ti in enumerate(t):
        if ti - last >= dt:
            idx.append(i)
            last = ti
    return np.asarray(idx)


fig, axes = plt.subplots(2, 3, figsize=(8, 5), sharex=True)
for row, (tag, u) in enumerate((("u15", "1.5"), ("u25", "2.5"))):
    ax = axes[row, 0]
    ax.plot(*data[f'dt_{tag}_tdmft'][:, :2].T, '-', label='DMFT')
    ax.plot(*data[f'dt_{tag}_B1'][:, :2].T, '--', label='B=1')
    ax.plot(*data[f'dt_{tag}_B3'][:, :2].T, '-', label='B=3')
    ax.plot(*data[f'dt_{tag}_B5'][:, :2].T, '-', label='B=5')
    ax.plot(*data[f'dt_{tag}_B7'][:, :2].T, '-', color='C7', label='B=7')
    for b, c in ((3, 'C2'), (5, 'C3'), (7, 'C7')):
        t, d = avqds[(b, u)]
        i = thin(t)
        ax.plot(t[i], d[i], 'o', color=c, mfc='none', ms=3, zorder=0)
    ax.set_ylabel(r'$\langle \uparrow\downarrow \rangle$')
axes[0, 0].set_xlim(-0.05, 5.05)
axes[0, 0].set_xticks([0, 1, 2, 3, 4, 5])
axes[0, 0].set_ylim(0.02, 0.27)
axes[0, 0].legend(loc='upper left', bbox_to_anchor=(0.3, 1), fontsize='small')
axes[1, 0].sharey(axes[0, 0])
for ax in axes[1]:
    ax.set_xlabel('$t$')

for row, u in ((0, '1.5'), (1, '2.5')):
    for b, c in ((3, 'C2'), (5, 'C3'), (7, 'C7')):
        t, inf = infid[(b, u)]
        axes[row, 1].plot(t, inf, 'o', color=c, mfc='none', ms=3, label=f'AVQDS B={b}')
    axes[row, 1].set_yscale('log')
    axes[row, 1].set_ylabel(r'$1-\mathcal{F}$')
    axes[row, 1].text(0.05, 0.92, f'$U={u}$', transform=axes[row, 1].transAxes, ha='left', va='top')
    ins = axes[row, 1].inset_axes([0.5, 0.09, 0.46, 0.4])
    for b, c in ((3, 'C2'), (5, 'C3'), (7, 'C7')):
        ins.plot(*sud[(b, u)], '-', color=c, lw=1)
    ins.set_xlim(-0.05, 5.05)
    ins.set_xticks([0, 1, 2, 3, 4, 5])
    ins.tick_params(labelsize=6, length=2, pad=1)
    ins.set_ylabel(r'$S_{\uparrow\downarrow}$', fontsize=7, labelpad=1)
axes[1, 1].sharey(axes[0, 1])

for row, u in ((0, '1.5'), (1, '2.5')):
    for b, c in ((3, 'C2'), (5, 'C3'), (7, 'C7')):
        axes[row, 2].plot(avqds[(b, u)][0], ncnot[(b, u)], 'o', color=c, mfc='none', ms=3, label=f'AVQDS B={b}')
    axes[row, 2].set_ylabel(r'$N_\textrm{cx}$')
axes[0, 2].legend(loc='upper left', fontsize='small')
axes[1, 2].sharey(axes[0, 2])

for ax, lab in zip(axes.flat, 'abcdef'):
    ax.text(-0.15, 0.98, f'({lab})', transform=ax.transAxes, ha='left', va='bottom')

plt.tight_layout()
fig.savefig(HERE / 'Fig3.pdf', bbox_inches='tight')
