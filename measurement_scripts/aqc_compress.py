"""Compress AVQDS embedding states into shallow brickwork circuits with AQC-Tensor (MPS based).

For each time t, the AVQDS state is read from results_asim.h5 and loaded as the target MPS. A brickwork
of general two-qubit blocks (KAK, <= 3 CNOTs each) on the qubit chain 0-1-...-(n-1) is fitted to it,
layer by layer: each depth starts from the optimum of the previous one plus a near-identity layer.
The quality measure is the error in the double occupancy d = <n_imp,up n_imp,dn> (qubits B, B+1).

    python aqc_compress.py RESULTS_ASIM.h5 OUT_PREFIX --times 0 0.2 ... [--lmax 12] [--dtol 1e-3]

Writes OUT_PREFIX_t<t>.json per time: layers, fidelity, d, |d - d_AVQDS|, CNOT count, and the optimized
parameters of each depth (enough to rebuild the circuits). Qubit q here is qubit q of the AVQDS/pyhemb
convention (bath up 0..B-1, impurity up B, impurity down B+1, bath down B+2..2B+1).
"""
import argparse, json, time
from functools import partial
import numpy as np
import h5py
import quimb.tensor
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator


def to_qiskit_order(psi, n):
    """AVQDS/qutip order (qubit 0 = most significant bit) -> qiskit order (qubit 0 = least significant)."""
    return np.ascontiguousarray(psi.reshape([2]*n).transpose(list(range(n - 1, -1, -1))).reshape(-1))


def docc_op(n, nb):
    """Physical double occupancy d = 1/2 - n_B n_{B+1} (n = (1 - Z)/2), as incar['observables'] of pyhemb:
    in the qubit encoding one impurity spin is particle-hole transformed, so d = <n_B (1 - n_{B+1})>
    at half filling, i.e. d = 1/4 + (Z_B + Z_{B+1} - Z_B Z_{B+1})/4."""
    return SparsePauliOp.from_sparse_list([("", [], 0.25), ("Z", [nb], 0.25), ("Z", [nb + 1], 0.25),
                                           ("ZZ", [nb, nb + 1], -0.25)], num_qubits=n)


def brick_layer(qc, layer, n, angle=1e-3):
    """One brickwork layer of near-identity two-qubit gates (only the connectivity matters)."""
    for q in range(layer % 2, n - 1, 2):
        qc.rzz(angle, q, q + 1)


def ncx(qc):
    t = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    return int(t.count_ops().get("cx", 0))


def compress(psi_q, n, nb, lmax, dtol, settings, log=print, jitters=(0.0,), seed=0, ftol=1e-3):
    target_qc = QuantumCircuit(n)
    target_qc.append(StatePreparation(psi_q), range(n))
    target_qc = transpile(target_qc, basis_gates=["cx", "u"], optimization_level=1)
    target = tensornetwork_from_circuit(target_qc, settings)
    dop = docc_op(n, nb)
    d_ref = float(Statevector(psi_q).expectation_value(dop).real)

    # start: dominant basis configuration of the target
    k = int(np.argmax(np.abs(psi_q)))
    start = QuantumCircuit(n)
    for q in range(n):
        if (k >> q) & 1:
            start.x(q)
    prev = start
    rows = []
    for L in range(1, lmax + 1):
        qc = prev.copy()
        brick_layer(qc, L - 1, n)
        ansatz, x0 = generate_ansatz_from_circuit(qc, qubits_initially_zero=True)
        obj = MaximizeStateFidelity(target, ansatz, settings)
        tw = time.time()
        # restarts from the warm start with small random kicks (they help to leave the previous
        # depth's local optimum); keep the best objective
        best, nfev = None, 0
        for r, jit in enumerate(jitters):
            xs = np.asarray(x0) + jit*np.random.default_rng(seed + 100*L + r).normal(size=len(x0))
            res = minimize(obj, xs, jac=True, method="L-BFGS-B", options=dict(maxiter=2000, ftol=1e-12, gtol=1e-9))
            nfev += int(res.nfev)
            if best is None or res.fun < best.fun:
                best = res
        res = best
        circ = ansatz.assign_parameters(res.x)
        sv = Statevector(circ)
        fid = float(abs(np.vdot(psi_q, sv.data))**2)
        d = float(sv.expectation_value(dop).real)
        nc = ncx(circ)
        rows.append(dict(layers=L, fidelity=fid, d=d, dd=abs(d - d_ref), ncx=nc, nfev=nfev,
                         sec=round(time.time() - tw, 1), params=[float(v) for v in res.x]))
        log(f"  L={L:2d}  1-F={1-fid:.2e}  d={d:.5f}  |dd|={abs(d-d_ref):.1e}  CX={nc:4d}  nfev={nfev}  {time.time()-tw:.0f}s")
        prev = circ.decompose()          # warm start for the next depth
        if abs(d - d_ref) < dtol and 1 - fid < ftol:
            break
    return d_ref, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5"); ap.add_argument("out")
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--lmax", type=int, default=12)
    ap.add_argument("--dtol", type=float, default=1e-3)
    ap.add_argument("--jitters", type=float, nargs="+", default=[0.0, 0.02, 0.02, 0.05],
                    help="one restart per value: random kick of the warm-start parameters")
    ap.add_argument("--ftol", type=float, default=1e-3, help="stop when |dd| < dtol and 1-F < ftol")
    a = ap.parse_args()
    settings = QuimbSimulator(partial(quimb.tensor.CircuitMPS, cutoff=1e-12), autodiff_backend="jax")
    with h5py.File(a.h5, "r") as f:
        t_all = f["/0/t"][()]; d_all = f["/0/observables"][:, 0]; ng = f["/0/ngates"][()]
        st = f["/0/state"]
        n = st.shape[1].bit_length() - 1; nb = (n - 2)//2
        lq = np.arange(1, ng.shape[1] + 1); ncx_avqds = ng @ (2*(lq - 1))
        for tt in a.times:
            i = int(np.argmin(abs(t_all - tt)))
            psi_q = to_qiskit_order(st[i], n)
            print(f"t = {t_all[i]:.4f} (step {i}): d_AVQDS (log) {d_all[i]:.6f}, AVQDS N_cx {ncx_avqds[i]}", flush=True)
            d_ref, rows = compress(psi_q, n, nb, a.lmax, a.dtol, settings, log=lambda s: print(s, flush=True), jitters=a.jitters, ftol=a.ftol)
            assert abs(d_ref - d_all[i]) < 1e-8, (d_ref, d_all[i])
            json.dump(dict(t=float(t_all[i]), step=i, d_avqds=float(d_all[i]), ncx_avqds=int(ncx_avqds[i]),
                           n_qubits=n, rows=rows), open(f"{a.out}_t{t_all[i]:.2f}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
