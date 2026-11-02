"""Compress AVQDS embedding states into brickwork circuits of general two-qubit gates by environment sweeps
(each gate in turn set to the unitary that maximizes the overlap with all other gates fixed: polar
decomposition of its 4x4 environment; Evenbly-Vidal). Dense statevectors (12 qubits); the overlap never
decreases during a sweep, and unlike the gradient fits of aqc_compress.py it does not stall on
small-amplitude parts of the state that the one-particle density matrix is sensitive to.

The depth grows layer by layer (the new layer starts as identity and is put at the end or the start of
the circuit, whichever sweeps to the better fidelity), with --restarts random kicks of the warm start per depth.
The starting state is the dominant basis configuration of the target (X gates).

    python sweep_compress.py RESULTS_ASIM.h5 OUT_PREFIX --times ... [--lmax 16] [--sweeps 3000]

Writes OUT_PREFIX_t<t>.json (per depth: fidelity, d, |dd|, max |rho - rho_AVQDS|, CNOTs) and
OUT_PREFIX_t<t>.qpy (circuits named L<depth>). Qubit q is qubit q of the AVQDS/pyhemb convention
(qubit 0 = most significant bit); a 4x4 gate on (q, q+1) acts on |a_q b_{q+1}> as in mps_compress.py.
"""
import argparse, json, os, sys, time
import numpy as np
import h5py
from qiskit import QuantumCircuit, qpy
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mps_compress import to_qiskit_order, docc_op, ncx_depth
from rdm1 import exact_rho


def apply(t, q, U):
    """Apply the 4x4 U to axes (q, q+1) of the state tensor t (shape [2]*n)."""
    t = np.moveaxis(t, [q, q + 1], [0, 1])
    sh = t.shape
    return np.moveaxis((U @ t.reshape(4, -1)).reshape(sh), [0, 1], [q, q + 1])


def env(phi, chi, q):
    """M with <chi|U|phi> = Tr(U M) for U on (q, q+1)."""
    a = np.moveaxis(phi, [q, q + 1], [0, 1]).reshape(4, -1)
    b = np.moveaxis(chi, [q, q + 1], [0, 1]).reshape(4, -1)
    return a @ b.conj().T


def sweep(gates, s, psi):
    """One forward sweep over all gates (list order = time order); returns |overlap|^2 after it."""
    chis = [psi]
    for q, U in reversed(gates):
        chis.append(apply(chis[-1], q, U.conj().T))
    chis = chis[::-1]                                    # chis[k] = G_{k+1}^dag ... G_N^dag psi
    phi = s
    for k, (q, _) in enumerate(gates):
        W, S, Vh = np.linalg.svd(env(phi, chis[k + 1], q))
        U = Vh.conj().T @ W.conj().T
        gates[k] = (q, U)
        phi = apply(phi, q, U)
    return abs(np.vdot(psi, phi))**2


def optimize(gates, s, psi, nsweep, tol=1e-12):
    f_old = -1.0
    for it in range(nsweep):
        f = sweep(gates, s, psi)
        if f - f_old < tol*max(1 - f, 1e-12) and it > 10:
            break
        f_old = f
    return f, it + 1


def layer(l, n):
    return [(q, np.eye(4, dtype=complex)) for q in range(l % 2, n - 1, 2)]


def kick(gates, eps, rng):
    from scipy.linalg import expm
    out = []
    for q, U in gates:
        h = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))
        out.append((q, expm(1j*eps*(h + h.conj().T)/2) @ U))
    return out


def circuit(k, gates, n):
    qc = QuantumCircuit(n)
    for q in range(n):
        if (k >> (n - 1 - q)) & 1:                       # AVQDS order: qubit 0 = most significant bit
            qc.x(q)
    for q, U in gates:
        qc.append(UnitaryGate(U, check_input=False), [q + 1, q])
    return qc


def compress(psi, n, nb, lmax, nsweep, restarts, seed=0, log=print):
    psi_q = to_qiskit_order(psi, n)
    blocks = [list(range(nb + 1)), list(range(nb + 1, n))]
    rho_ex = [exact_rho(psi_q, b, n) for b in blocks]
    dop = docc_op(n, nb)
    d_ex = float(Statevector(psi_q).expectation_value(dop).real)
    k = int(np.argmax(np.abs(psi)))
    s = np.zeros(2**n, complex); s[k] = 1; s = s.reshape([2]*n)
    T = psi.reshape([2]*n)
    rng = np.random.default_rng(seed)
    gates, rows, circs = [], [], []
    p0, p1 = 1, 1                                        # brickwork parity of the first and last layer
    for L in range(1, lmax + 1):
        tw = time.time()
        best = None
        # new layer at the end or at the start, whichever sweeps to the better fidelity
        trials = [(gates + layer(p1 + 1, n), p0 if gates else p1 + 1, p1 + 1)]
        if gates:
            trials.append((layer(p0 + 1, n) + gates, p0 + 1, p1))
        for g0, a0, a1 in trials:
            for r in range(restarts + 1):
                g = kick(g0, 0.02, rng) if r else list(g0)
                f, it = optimize(g, s, T, nsweep)
                if best is None or f > best[0]:
                    best = (f, g, it, a0 % 2, a1 % 2)
        f, gates, it, p0, p1 = best
        qc = circuit(k, gates, n)
        sv = Statevector(qc)
        fid = float(abs(np.vdot(psi_q, sv.data))**2)
        assert abs(fid - f) < 1e-8, (fid, f)
        d = float(sv.expectation_value(dop).real)
        drho = float(max(np.abs(exact_rho(sv.data, b, n) - rho_ex[j]).max() for j, b in enumerate(blocks)))
        nc, dep = ncx_depth(qc)
        rows.append(dict(layers=L, fidelity=fid, d=d, dd=d - d_ex, drho=drho, ncx=nc, depth2q=dep, sweeps=it))
        qc.name = f"L{L}"
        circs.append(qc)
        log(f"  L={L:2d}  1-F={1-fid:.2e}  dd={d-d_ex:+.1e}  drho={drho:.4f}  CX={nc:4d}  sweeps={it}  {time.time()-tw:.0f}s")
    return d_ex, rows, circs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5"); ap.add_argument("out")
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--lmax", type=int, default=16)
    ap.add_argument("--sweeps", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=2)
    a = ap.parse_args()
    with h5py.File(a.h5, "r") as f:
        t_all = f["/0/t"][()]; d_all = f["/0/observables"][:, 0]
        st = f["/0/state"]; n = st.shape[1].bit_length() - 1; nb = (n - 2)//2
        for tt in a.times:
            i = int(np.argmin(abs(t_all - tt)))
            psi = st[i]/np.linalg.norm(st[i])
            print(f"t = {t_all[i]:.4f} (step {i}): d_AVQDS {d_all[i]:.6f}", flush=True)
            d_ex, rows, circs = compress(psi, n, nb, a.lmax, a.sweeps, a.restarts, log=lambda s: print(s, flush=True))
            json.dump(dict(t=float(t_all[i]), step=i, d_avqds=d_ex, n_qubits=n, rows=rows),
                      open(f"{a.out}_t{t_all[i]:.2f}.json", "w"), indent=1)
            with open(f"{a.out}_t{t_all[i]:.2f}.qpy", "wb") as fq:
                qpy.dump(circs, fq)


if __name__ == "__main__":
    main()
