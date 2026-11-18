"""MPS-based compression of AVQDS embedding states: analytic staircase layers (Ran, PRA 101, 032310
(2020)), optionally refined with AQC-Tensor.

Layer construction: truncate the current state to a bond-dimension-2 MPS (left-canonical). Such an MPS is
prepared exactly from |0...0> by a staircase of two-qubit unitaries U_n (qubits n-2, n-1), ..., U_2
(qubits 0, 1) followed by a one-qubit U_1 (qubit 0), each the unitary completion of an MPS tensor, with
the bond index carried by the qubit to the left. Disentangle, |psi> <- C^dag |psi>, and repeat:
|psi> ~ C_1 C_2 ... C_K |0>. The circuit of K layers is then optionally refined by AQC-Tensor
(maximize the fidelity with the exact target MPS, all gates free, warm start).

Qubit order here is the AVQDS/pyhemb one (qubit 0 = most significant bit of the state index). The
qiskit circuits use the same qubit labels (UnitaryGate on [q+1, q] for a 4x4 matrix acting on |a_q b_{q+1}>).

    python mps_compress.py RESULTS_ASIM.h5 OUT_PREFIX --times ... [--kmax 8] [--refine] [--qpy]

--qpy also writes the circuits of every K (refined ones when --refine) to OUT_PREFIX_t<t>.qpy.
"""
import argparse, json, time
from functools import partial
import numpy as np
import h5py
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector, SparsePauliOp


def complete_unitary(cols):
    """4x4 (or 2x2) unitary whose first columns are the orthonormal columns `cols` (dim x k)."""
    dim, k = cols.shape
    rng = np.random.default_rng(0)
    m = np.concatenate([cols, rng.normal(size=(dim, dim - k)) + 1j*rng.normal(size=(dim, dim - k))], axis=1)
    q, r = np.linalg.qr(m)
    q[:, :k] = cols            # QR may change phases of the given columns; restore them exactly
    # re-orthogonalize the completion against the fixed columns
    for j in range(k, dim):
        v = q[:, j] - q[:, :j] @ (q[:, :j].conj().T @ q[:, j])
        q[:, j] = v/np.linalg.norm(v)
    return q


def chi2_layer(psi, n):
    """Staircase gates [(q, U), ...] (apply in list order) preparing the bond-dimension-2 truncation
    of psi from |0...0>; returns gates and the fidelity of that truncation."""
    # left-canonical MPS with bond dimension <= 2 by an SVD sweep with truncation
    tens, rest, dl = [], psi.reshape(1, -1), 1
    for k in range(n - 1):
        m = rest.reshape(dl*2, -1)
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        d = min(2, len(s))
        tens.append(u[:, :d].reshape(dl, 2, d))
        rest = (s[:d, None]*vh[:d])
        dl = d
    last = rest.reshape(dl, 2, 1)
    last = last/np.linalg.norm(last)
    tens.append(last)
    # A_k: (D_{k-1}, 2, D_k) left-canonical; build unitaries with bond carried by qubit k-1
    gates = []
    # last site: vector over (alpha_{n-2}, s_{n-1}) on qubits (n-2, n-1)
    A = tens[n - 1]                                  # (D, 2, 1)
    v = np.zeros((2, 2), complex); v[:A.shape[0], :] = A[:, :, 0]
    gates.append((n - 2, complete_unitary(v.reshape(4, 1))))
    for k in range(n - 2, 0, -1):                    # sites k = n-2 .. 1, unitary on qubits (k-1, k)
        A = tens[k]                                  # (D_{k-1}, 2, D_k)
        cols = np.zeros((2, 2, A.shape[2]), complex); cols[:A.shape[0], :, :] = A
        cols = cols.reshape(4, A.shape[2])           # isometry: |beta>_k -> |alpha>_{k-1}|s>_k
        # input basis |0>_{k-1}|beta>_k is column index beta (since 2*0 + beta)
        gates.append((k - 1, complete_unitary(cols)))
    A = tens[0]                                      # (1, 2, D_0): |beta>_0 -> |s>_0
    gates.append((None, complete_unitary(A[0, :, :])))
    return gates


def apply_gates(psi, n, gates, dagger=False):
    """Apply staircase gates (list order) or their inverse (reverse order, U^dag) to a dense state."""
    seq = reversed(gates) if dagger else gates
    t = psi.reshape([2]*n)
    for q, U in seq:
        U = U.conj().T if dagger else U
        if q is None:                                # one-qubit gate on qubit 0
            t = np.tensordot(U, t, axes=([1], [0]))
        else:
            t = np.moveaxis(t, [q, q + 1], [0, 1])
            sh = t.shape
            t = (U @ t.reshape(4, -1)).reshape(sh)
            t = np.moveaxis(t, [0, 1], [q, q + 1])
    return t.reshape(-1)


def layers_circuit(layer_gates, n):
    """qiskit circuit C_1 ... C_K |0>: C_K is applied first."""
    qc = QuantumCircuit(n)
    for gates in reversed(layer_gates):
        for q, U in gates:
            if q is None:
                qc.append(UnitaryGate(U), [0])
            else:
                qc.append(UnitaryGate(U), [q + 1, q])
    return qc


def to_qiskit_order(psi, n):
    return np.ascontiguousarray(psi.reshape([2]*n).transpose(list(range(n - 1, -1, -1))).reshape(-1))


def docc_op(n, nb):
    """Physical double occupancy in pyhemb's qubit encoding: d = 1/4 + (Z_B + Z_{B+1} - Z_B Z_{B+1})/4."""
    return SparsePauliOp.from_sparse_list([("", [], 0.25), ("Z", [nb], 0.25), ("Z", [nb + 1], 0.25),
                                           ("ZZ", [nb, nb + 1], -0.25)], num_qubits=n)


def ncx_depth(qc):
    t = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    return int(t.count_ops().get("cx", 0)), int(t.depth(lambda g: g.operation.num_qubits == 2))


def refine(qc, psi, n, settings, log):
    """AQC-Tensor refinement of all gates of qc towards the exact state psi (AVQDS order).
    qiskit-quimb maps qiskit qubit q to quimb site q with site 0 the most significant, i.e. the AVQDS
    order, so the target MPS is built directly from psi."""
    import quimb.tensor as qtn
    from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    target = qtn.CircuitMPS(n, psi0=qtn.MatrixProductState.from_dense(psi, dims=[2]*n), cutoff=1e-12)
    ansatz, x0 = generate_ansatz_from_circuit(transpile(qc, basis_gates=["cx", "u"], optimization_level=1),
                                              qubits_initially_zero=True)
    obj = MaximizeStateFidelity(target, ansatz, settings)
    res = minimize(obj, np.asarray(x0), jac=True, method="L-BFGS-B", options=dict(maxiter=3000, ftol=1e-13, gtol=1e-10))
    return ansatz.assign_parameters(res.x), int(res.nfev)


def compress(psi, n, nb, kmax, ftarget, refine_settings=None, log=print, circuits=None):
    psi_q = to_qiskit_order(psi, n)
    dop = docc_op(n, nb)
    d_ref = float(Statevector(psi_q).expectation_value(dop).real)
    rows, layer_gates, phi = [], [], psi.copy()
    for K in range(1, kmax + 1):
        tw = time.time()
        gates = chi2_layer(phi, n)
        layer_gates.append(gates)
        phi = apply_gates(phi, n, gates, dagger=True)
        fid_layers = float(abs(phi[0])**2)           # |<0|C_K^dag ... C_1^dag|psi>|^2
        qc = layers_circuit(layer_gates, n)
        sv = Statevector(qc)
        fid = float(abs(np.vdot(psi_q, sv.data))**2)
        assert abs(fid - fid_layers) < 1e-8, (fid, fid_layers)
        d = float(sv.expectation_value(dop).real)
        nc, dep = ncx_depth(qc)
        row = dict(layers=K, fidelity=fid, d=d, dd=abs(d - d_ref), ncx=nc, depth2q=dep)
        msg = f"  K={K}  1-F={1-fid:.2e}  |dd|={abs(d-d_ref):.1e}  CX={nc:4d}  2q-depth={dep:4d}"
        if refine_settings is not None:
            qr, nfev = refine(qc, psi, n, refine_settings, log)
            svr = Statevector(qr)
            fr = float(abs(np.vdot(psi_q, svr.data))**2)
            dr = float(svr.expectation_value(dop).real)
            ncr, depr = ncx_depth(qr)
            row.update(fidelity_ref=fr, d_ref=dr, dd_ref=abs(dr - d_ref), ncx_ref=ncr, depth2q_ref=depr, nfev=nfev)
            qc = qr
            msg += f"  | refined: 1-F={1-fr:.2e}  |dd|={abs(dr-d_ref):.1e}  CX={ncr}"
        rows.append(row)
        if circuits is not None:
            qc.name = f"K{K}"
            circuits.append(qc)
        log(msg + f"  {time.time()-tw:.0f}s")
        if 1 - max(row["fidelity"], row.get("fidelity_ref", 0)) < ftarget:
            break
    return d_ref, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5"); ap.add_argument("out")
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--ftarget", type=float, default=1e-4)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--qpy", action="store_true")
    a = ap.parse_args()
    settings = None
    if a.refine:
        import quimb.tensor
        from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
        settings = QuimbSimulator(partial(quimb.tensor.CircuitMPS, cutoff=1e-12), autodiff_backend="jax")
    with h5py.File(a.h5, "r") as f:
        t_all = f["/0/t"][()]; d_all = f["/0/observables"][:, 0]; ng = f["/0/ngates"][()]
        st = f["/0/state"]; n = st.shape[1].bit_length() - 1; nb = (n - 2)//2
        lq = np.arange(1, ng.shape[1] + 1); ncx_avqds = ng @ (2*(lq - 1))
        for tt in a.times:
            i = int(np.argmin(abs(t_all - tt)))
            psi = st[i]/np.linalg.norm(st[i])
            print(f"t = {t_all[i]:.4f} (step {i}): d_AVQDS {d_all[i]:.6f}, AVQDS N_cx {ncx_avqds[i]}", flush=True)
            circs = [] if a.qpy else None
            d_ref, rows = compress(psi, n, nb, a.kmax, a.ftarget, settings, log=lambda s: print(s, flush=True), circuits=circs)
            assert abs(d_ref - d_all[i]) < 1e-6, (d_ref, d_all[i])
            json.dump(dict(t=float(t_all[i]), step=i, d_avqds=float(d_all[i]), ncx_avqds=int(ncx_avqds[i]),
                           n_qubits=n, rows=rows), open(f"{a.out}_t{t_all[i]:.2f}.json", "w"), indent=1)
            if circs:
                from qiskit import qpy
                with open(f"{a.out}_t{t_all[i]:.2f}.qpy", "wb") as fq:
                    qpy.dump(circs, fq)


if __name__ == "__main__":
    main()
