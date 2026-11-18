"""Fig. 2: t-GQCE with B = 1 (conventional time-dependent Gutzwiller approximation, single-qubit embedding state).

    python plot_fig2.py        # writes Fig2.pdf

(a) double occupancy after the quench U: 0 -> U_c^dyn/2, (b) oscillation period T versus U_f,
(c) double occupancy for U_f = U_c^dyn: statevector, IBMQ_Quito noise model and IBMQ_Quito device,
with the Bloch-sphere trace of the embedding state as an inset (requires QuTiP).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from qutip import Bloch

HERE = Path(__file__).resolve().parent
plt.rcParams['text.usetex'] = True


def load_csv(name):
    """t, <n_up n_dn> from a two-column csv with one header line."""
    return np.loadtxt(HERE / name, delimiter=',', skiprows=1, unpack=True)


t_a, d_a = np.loadtxt(HERE / 'a_double_occupancy_Uf_half_Uc_dyn.dat', unpack=True)
u_b, period_b = np.loadtxt(HERE / 'b_period_vs_Uf.dat', unpack=True)
sv = load_csv('c_statevector.csv')
nm = load_csv('c_noise_model.csv')
dev = load_csv('c_ibmq_quito.csv')

# Bloch vector of the noiseless embedding state, every 50th of the first 700 time points
w = np.loadtxt(HERE / 'c_statevector_wavefunction.dat')[:700:50, 1:5]
states = w.view(np.complex128).reshape(-1, 2)
pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
bloch_points = np.array([[np.vdot(psi, p @ psi).real for psi in states] for p in pauli])

fig = plt.figure(figsize=(4.5, 4.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.6], hspace=0.33, wspace=0.35)
ax_dt = fig.add_subplot(gs[0, 0])
ax_pu = fig.add_subplot(gs[0, 1])
axes = fig.add_subplot(gs[1, :])

ax_dt.plot(t_a, d_a, '-k')
ax_dt.set_xlabel(r'$t$')
ax_dt.set_ylabel(r'$\langle\uparrow\downarrow\rangle$')
ax_dt.set_xlim(0, 10)

ax_pu.plot(u_b, period_b, '-b')
ax_pu.set_xlabel(r'$U_\textrm{f}$')
ax_pu.set_ylabel(r'$T$')
ax_pu.set_yticks([2, 4, 6, 8, 10])
U_c_dyn = 16 / (3 * np.pi)                    # U_c^BR / 2 with U_c^BR = 32 / (3 pi)
ax_pu.axvline(U_c_dyn, color='r', linestyle='--', label=r'$U_\textrm{c}^\textrm{dyn}$')
ax_pu.legend(loc="upper left", handlelength=1.5, frameon=False)

axes.plot(*sv, '-k', label='Statevector')
axes.plot(*nm, '-b', label='Noise Model')
axes.plot(*dev, '-r', label='IBMQ_Quito')
axes.legend(loc="lower left", handlelength=0.6, frameon=False)
axes.set_xlabel(r'$t$')
axes.set_ylabel(r'$\langle\uparrow\downarrow\rangle$')
axes.text(0.4, 0.6, r'$U_\textrm{f} = U_\textrm{c}^\textrm{dyn}$', transform=axes.transAxes, ha='center', va='center')

ax_dt.text(-0.32, 0.95, '(a)', transform=ax_dt.transAxes, ha='center', va='center', fontsize=12)
ax_pu.text(-0.23, 0.95, '(b)', transform=ax_pu.transAxes, ha='center', va='center', fontsize=12)
axes.text(-0.13, 0.95, '(c)', transform=axes.transAxes, ha='center', va='center', fontsize=12)

fig.tight_layout()

# Bloch-sphere inset inside panel (c)
bbox = axes.get_position()
ax3d = fig.add_axes([bbox.x0 + 0.52 * bbox.width, bbox.y0 + 0.12 * bbox.height, 0.40 * bbox.width, 0.78 * bbox.height],
                    projection="3d")
b = Bloch(fig=fig, axes=ax3d)
b.point_marker_size = 1
cmap = plt.get_cmap('plasma')
n = bloch_points.shape[1]
b.point_color = [cmap(i / n) for i in range(n)]
b.add_points(bloch_points)
b.font_size = 12
b.vector_label_fontsize = 12
b.point_label_fontsize = 12
b.make_sphere()

fig.savefig(HERE / 'Fig2.pdf')
