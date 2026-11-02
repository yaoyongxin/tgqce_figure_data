"""Compile the compressed circuits for Quantinuum and IBM Heron and estimate the error in d with noise models.

For each time t and each circuit of the compression runs (brickwork AQC, aqc_t*.json, rebuilt by replaying
the warm-start chain; MPS staircase + AQC refinement, circ/mps_t*.qpy from mps_compress.py --qpy; brickwork by
environment sweeps, swp/swp_t*.qpy from sweep_compress.py):

- Quantinuum (all-to-all): pytket FullPeepholeOptimise to TK2 + exact DecomposeTK2 into ZZPhase (arbitrary
  angle), rebase to {ZZPhase, PhasedX, Rz}. Counts N_ZZ, N_1q (PhasedX; Rz are virtual), 2q depth, and
  HQC = 5 + shots/5000 (N_1q + 10 N_2q + 5 N_meas) per circuit (Quantinuum's cost formula, check the current one).
  Noise model (assumed, H2-class published specs): depolarizing 1q 3e-5, 2q 1e-3 (average infidelities),
  symmetric readout 1.5e-3, and memory error: Z dephasing with probability p_mem on every qubit for every
  two-qubit layer (idle or not; transport and waiting), --pmem values (default 0, 2e-4 and 1e-3; the
  spec-sheet order of magnitude is 1e-4 - 5e-4 per qubit per depth-1 layer).
- IBM Heron (FakeFez, heavy-hex, CZ): qiskit transpile, optimization_level 3, best of a few seeds (fewest CZ).
  Noise: AerSimulator.from_backend (gate depolarizing + thermal relaxation from the fake backend's
  calibration snapshot, idle relaxation via delays), readout from the snapshot.

All qubits are measured in Z. d = 1/2 - P(q_B = 1, q_{B+1} = 1) (pyhemb encoding) is computed from the exact
outcome distribution (density-matrix simulation, no shot noise) with independent readout flips applied
analytically, raw and post-selected on the bit counts of the two spin blocks (qubits 0..B and B+1..2B+1),
which are sharp in the target state (3 and 3 for B = 5).
Shots for a statistical error sigma: N = P11 (1 - P11) / (sigma^2 * acceptance).

    python hw_compile.py DIR --times 0 0.2 ... [--fmax 1e-2] [--platforms quantinuum ibm] [--pmem 0 2e-4 1e-3]
        [--methods aqc mps swp]

writes DIR/hw_t<t>.json; with a subset of --platforms the other platforms' entries of an existing file are kept.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import h5py
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aqc_compress import to_qiskit_order, brick_layer, docc_op

H5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AVQDS", "{case}", "results_asim.h5")
P1Q, P2Q, PMEAS = 3e-5, 1e-3, 1.5e-3          # assumed Quantinuum H2-class average infidelities


def aqc_circuits(js, psi_q, n):
    """Rebuild the brickwork circuits of aqc_compress.py by replaying its warm-start chain."""
    from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
    k = int(np.argmax(np.abs(psi_q)))
    qc = QuantumCircuit(n)
    for q in range(n):
        if (k >> q) & 1:
            qc.x(q)
    out = []
    for L, row in enumerate(js["rows"], 1):
        qc = qc.copy()
        brick_layer(qc, L - 1, n)
        ans, x0 = generate_ansatz_from_circuit(qc, qubits_initially_zero=True)
        circ = ans.assign_parameters(row["params"])
        out.append((L, circ))
        qc = circ.decompose()
    return out


def load_circuits(dir, t, psi_q, n, methods=("aqc", "mps", "swp")):
    """Candidate circuits [(method, depth, circuit)] of the compression runs at time t (file label t:.2f)."""
    cands = []
    fa = os.path.join(dir, f"aqc_t{t:.2f}.json")
    if "aqc" in methods and os.path.exists(fa):
        cands += [("aqc", L, c) for L, c in aqc_circuits(json.load(open(fa)), psi_q, n)]
    for meth, sub in (("mps", "circ"), ("swp", "swp")):
        f = os.path.join(dir, sub, f"{meth}_t{t:.2f}.qpy")
        if meth in methods and os.path.exists(f):
            with open(f, "rb") as fq:
                cands += [(meth, int(c.name[1:]), c) for c in qpy.load(fq)]
    return cands


def quantinuum_compile(qc):
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    from pytket.passes import (DecomposeBoxes, FullPeepholeOptimise, NormaliseTK2, DecomposeTK2, AutoRebase,
                               RemoveRedundancies, SquashRzPhasedX, SequencePass)
    from pytket import OpType
    tk = qiskit_to_tk(transpile(qc, basis_gates=["cx", "u"], optimization_level=0))
    SequencePass([DecomposeBoxes(), FullPeepholeOptimise(allow_swaps=False, target_2qb_gate=OpType.TK2),
                  NormaliseTK2(), DecomposeTK2(allow_swaps=False, ZZPhase_fidelity=1.0),
                  AutoRebase({OpType.ZZPhase, OpType.PhasedX, OpType.Rz}), SquashRzPhasedX(),
                  RemoveRedundancies()]).apply(tk)
    n2 = tk.n_gates_of_type(OpType.ZZPhase)
    n1 = tk.n_gates_of_type(OpType.PhasedX)
    dep = tk.depth_2q()
    qk = transpile(tk_to_qiskit(tk), basis_gates=["rzz", "u"], optimization_level=0)
    return qk, dict(n2q=int(n2), n1q=int(n1), depth2q=int(dep), n_rzz=int(qk.count_ops().get("rzz", 0)))


# npj QI 12, 102 (2026) (chen2025quantum), following Self, Benedetti & Amaro (their ref. 58), H1-2 randomized
# benchmarking: Pauli-channel probabilities E_1q = (1 - p1) rho + p1/3 sum_P P rho P, E_2q = (1 - p2) rho + p2/15 sum_P2 ...,
# bit flips p_init after initialization and p_meas before measurement; all scaled by lam (lam = 1: the device).
NPJ = dict(p1=4e-4, p2=3e-3, pinit=4e-4, pmeas=3e-3)


def noise_params(model="h2", lam=1.0):
    """Error rates of the Quantinuum noise models: "h2" (P1Q, P2Q average infidelities, PMEAS, no init error) or
    "npj" (NPJ Pauli-channel probabilities, scaled by lam). Returns qiskit depolarizing parameters dep1, dep2
    (E = (1 - dep) rho + dep I/d), pinit and pmeas."""
    if model == "h2":
        return dict(dep1=2*P1Q, dep2=4/3*P2Q, pinit=0.0, pmeas=PMEAS)
    if model == "npj":
        return dict(dep1=4/3*lam*NPJ["p1"], dep2=16/15*lam*NPJ["p2"], pinit=lam*NPJ["pinit"], pmeas=lam*NPJ["pmeas"])
    raise ValueError(model)


def quantinuum_noise(pmem=0.0, model="h2", lam=1.0):
    """Aer noise model; the init bit flip acts on `reset` (prepend with_init_reset to the circuit)."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
    q = noise_params(model, lam)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(q["dep1"], 1), ["u"])
    nm.add_all_qubit_quantum_error(depolarizing_error(q["dep2"], 2), ["rzz"])
    if q["pinit"] > 0:
        nm.add_all_qubit_quantum_error(pauli_error([("X", q["pinit"]), ("I", 1 - q["pinit"])]), ["reset"])
    if pmem > 0:
        nm.add_all_qubit_quantum_error(pauli_error([("Z", pmem), ("I", 1 - pmem)]), ["id"])
    return nm


def with_init_reset(qc):
    """qc preceded by a reset of every qubit (carries the initialization error of the npj model)."""
    out = qc.copy_empty_like()
    for q in range(qc.num_qubits):
        out.reset(q)
    return out.compose(qc)


def with_memory_layers(qc):
    """Insert one `id` (carrying the memory error) per qubit per two-qubit layer (ASAP layering of the
    two-qubit gates): before a two-qubit gate in layer L its qubits catch up on idle layers, after it they get
    the layer itself; at the end every qubit is padded to the full two-qubit depth."""
    out = qc.copy_empty_like()
    clock = [0]*qc.num_qubits
    for ins in qc.data:
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if ins.operation.num_qubits == 2:
            L = 1 + max(clock[q] for q in qs)
            for q in qs:
                for _ in range(L - 1 - clock[q]):
                    out.id(q)
            out.append(ins)
            for q in qs:
                out.id(q)
                clock[q] = L
        else:
            out.append(ins)
    D = max(clock)
    for q in range(qc.num_qubits):
        for _ in range(D - clock[q]):
            out.id(q)
    return out, D


def ibm_compile(qc, backend, seeds=(0, 1, 2, 3)):
    best = None
    base = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    for s in seeds:
        t = transpile(base, backend=backend, optimization_level=3, seed_transpiler=s)
        ncz = t.count_ops().get("cz", 0)
        if best is None or ncz < best[1]:
            best = (t, ncz)
    t = best[0]
    dep = t.depth(lambda g: g.operation.num_qubits == 2)
    lay = t.layout
    return t, dict(ncz=int(best[1]), depth2q=int(dep),
                   routed_swaps=lay.initial_index_layout(filter_ancillas=True)[:qc.num_qubits] != lay.final_index_layout())


def probs(qc, qubits, sim):
    """Exact outcome distribution of `qubits` (list order = logical qubits 0..n-1), shape [2]*n with axis q
    = logical qubit q, without readout error."""
    c = qc.copy()
    c.save_probabilities(qubits)
    p = np.asarray(sim.run(c).result().data()["probabilities"])   # little endian: index = sum b_q 2^q
    n = len(qubits)
    return p.reshape([2]*n).transpose(list(range(n - 1, -1, -1)))


def readout(p, eps):
    """Independent symmetric readout flips, eps[q] for logical qubit q."""
    for q, e in enumerate(eps):
        M = np.array([[1 - e, e], [e, 1 - e]])
        p = np.moveaxis(np.tensordot(M, p, axes=([1], [q])), 0, q)
    return p


def d_estimates(p, nb, sectors):
    """d raw, d post-selected on (N block 1, N block 2) = sectors, acceptance, P11 (post-selected)."""
    n = p.ndim
    d_raw = 0.5 - p.take(1, nb).take(1, nb).sum()
    idx = np.indices([2]*n)
    n1 = idx[:nb + 1].sum(0); n2 = idx[nb + 1:].sum(0)
    keep = (n1 == sectors[0]) & (n2 == sectors[1])
    acc = p[keep].sum()
    p11 = p[keep & (idx[nb] == 1) & (idx[nb + 1] == 1)].sum()/acc
    return float(d_raw), float(0.5 - p11), float(acc), float(p11)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--case", default="B5U1.5")
    ap.add_argument("--fmax", type=float, default=1e-2, help="skip circuits with 1-F above this")
    ap.add_argument("--platforms", nargs="+", default=["quantinuum", "ibm"])
    ap.add_argument("--methods", nargs="+", default=["aqc", "mps", "swp"],
                    help="compression runs to (re)compute; rows of the others in an existing hw_t*.json are kept")
    ap.add_argument("--pmem", type=float, nargs="+", default=[0.0, 2e-4, 1e-3],
                    help="Quantinuum memory error per qubit per two-qubit layer; the first one fills d_noisy")
    a = ap.parse_args()
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeFez
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
    fez = FakeFez()
    sim_ibm = AerSimulator.from_backend(fez, method="density_matrix")
    sim_ibm_ideal = AerSimulator(method="statevector")
    sim_q = {pm: AerSimulator(method="density_matrix", noise_model=quantinuum_noise(pm)) for pm in a.pmem}
    durations = fez.target.durations()
    sched = PassManager([ALAPScheduleAnalysis(durations), PadDelay(target=fez.target)])
    with h5py.File(H5.format(case=a.case), "r") as f:
        t_all = f["/0/t"][()]; st = f["/0/state"]; d_all = f["/0/observables"][:, 0]
        n = st.shape[1].bit_length() - 1; nb = (n - 2)//2
        for tt in a.times:
            i = int(np.argmin(abs(t_all - tt)))
            psi_q = to_qiskit_order(st[i]/np.linalg.norm(st[i]), n)
            dop = docc_op(n, nb)
            idx = np.arange(2**n); w = np.abs(psi_q)**2       # conserved block numbers of the target (qiskit order)
            bl = lambda qs: sum((idx >> q) & 1 for q in qs)
            sectors = (int(np.round(w @ bl(range(nb + 1)))), int(np.round(w @ bl(range(nb + 1, n)))))
            leak = float(w[(bl(range(nb + 1)) != sectors[0]) | (bl(range(nb + 1, n)) != sectors[1])].sum())
            assert leak < 1e-3, leak     # AVQDS conserves only the block parities exactly; leakage ~1e-4 at t = 6
            d_ex = float(Statevector(psi_q).expectation_value(dop).real)
            cands = load_circuits(a.dir, t_all[i], psi_q, n, a.methods)
            print(f"t = {t_all[i]:.2f}: d_AVQDS {d_ex:.5f}, leakage out of {sectors} {leak:.1e}, {len(cands)} circuits", flush=True)
            fout = os.path.join(a.dir, f"hw_t{t_all[i]:.2f}.json")
            old = {}                     # existing rows: kept, or updated for the platforms computed now
            if os.path.exists(fout):
                old = {(r["method"], r["layers"]): r for r in json.load(open(fout))["rows"]}
                if set(a.platforms) == {"quantinuum", "ibm"}:
                    old = {k: v for k, v in old.items() if k[0] not in a.methods}
            rows = [r for k, r in old.items() if k[0] not in a.methods]
            for meth, L, qc in cands:
                tw = time.time()
                sv = Statevector(qc)
                fid = float(abs(np.vdot(psi_q, sv.data))**2)
                d_c = float(sv.expectation_value(dop).real)
                if 1 - fid > a.fmax:
                    continue
                row = old.get((meth, L), {})
                row.update(method=meth, layers=L, fidelity=fid, d=d_c, dd=d_c - d_ex)
                if "quantinuum" in a.platforms:
                    qk, info = quantinuum_compile(qc)
                    p = probs(qk, list(range(n)), AerSimulator(method="statevector"))
                    d_qi, d_qi_ps, acc_i, _ = d_estimates(p, nb, sectors)
                    qm, D = with_memory_layers(qk)
                    mem = {}
                    for pm in a.pmem:
                        p = readout(probs(qm, list(range(n)), sim_q[pm]), [PMEAS]*n)
                        d_qn, d_qn_ps, acc, p11 = d_estimates(p, nb, sectors)
                        mem[f"{pm:g}"] = dict(d_noisy=d_qn, d_noisy_ps=d_qn_ps, acceptance=acc, p11_ps=p11)
                    row["quantinuum"] = dict(info, d_ideal=d_qi, acc_ideal=acc_i, memory_layers=D, pmem=a.pmem[0],
                                             **mem[f"{a.pmem[0]:g}"], memory=mem,
                                             hqc_per_1000_shots=5 + 1000/5000*(info["n1q"] + 10*info["n2q"] + 5*n))
                if "ibm" not in a.platforms:
                    rows.append(row)
                    q = row["quantinuum"]
                    print(f"  {meth} {L:2d}  1-F={1-fid:.1e} dd={d_c-d_ex:+.1e} | Q: ZZ={q['n2q']:4d} d2q={q['depth2q']:4d} ps err "
                          + "  ".join(f"pmem {k}: {v['d_noisy_ps']-d_ex:+.4f} (acc {v['acceptance']:.2f})" for k, v in q["memory"].items())
                          + f"  {time.time()-tw:.0f}s", flush=True)
                    continue
                # IBM Heron
                tq, info = ibm_compile(qc, fez)
                phys = tq.layout.final_index_layout()[:n]
                p = probs(tq, phys, sim_ibm_ideal)
                d_ii = d_estimates(p, nb, sectors)[0]
                ts = sched.run(tq)
                eps = [fez.target["measure"][(q,)].error for q in phys]
                p = readout(probs(ts, phys, sim_ibm), eps)
                d_in, d_in_ps, acc_b, p11_b = d_estimates(p, nb, sectors)
                p = readout(probs(tq, phys, sim_ibm), eps)
                d_in_nd, d_in_nd_ps, _, _ = d_estimates(p, nb, sectors)
                row["ibm_fez"] = dict(info, d_ideal=d_ii, d_noisy=d_in, d_noisy_ps=d_in_ps, acceptance=acc_b, p11_ps=p11_b,
                                      d_noisy_no_idle=d_in_nd, d_noisy_no_idle_ps=d_in_nd_ps,
                                      duration_us=float(tq.estimate_duration(fez.target, unit="s")*1e6),
                                      physical_qubits=[int(q) for q in phys], readout=eps)
                rows.append(row)
                q, b = row["quantinuum"], row["ibm_fez"]
                print(f"  {meth} {L:2d}  1-F={1-fid:.1e} dd={d_c-d_ex:+.1e} | Q: ZZ={q['n2q']:4d} d2q={q['depth2q']:4d} "
                      f"err={q['d_noisy']-d_ex:+.4f} ps {q['d_noisy_ps']-d_ex:+.4f} acc {q['acceptance']:.2f} "
                      f"(ideal {q['d_ideal']-d_c:+.0e}) HQC/1k={q['hqc_per_1000_shots']:.0f} | "
                      f"IBM: CZ={b['ncz']:4d} d2q={b['depth2q']:4d} err={b['d_noisy']-d_ex:+.4f} ps {b['d_noisy_ps']-d_ex:+.4f} "
                      f"acc {b['acceptance']:.2f} (no idle {b['d_noisy_no_idle']-d_ex:+.4f}, ideal {b['d_ideal']-d_c:+.0e}) "
                      f"{b['duration_us']:.1f}us  {time.time()-tw:.0f}s", flush=True)
            json.dump(dict(t=float(t_all[i]), step=i, d_avqds=d_ex, n_qubits=n,
                           noise_quantinuum=dict(p1=P1Q, p2=P2Q, pmeas=PMEAS, pmem=a.pmem),
                           ibm_backend="FakeFez", avqds_leakage=leak, sectors=sectors, rows=rows), open(fout, "w"), indent=1)


if __name__ == "__main__":
    main()
