"""One-particle density matrix of the AVQDS embedding states from Z-basis measurements after orbital rotations,
for the compressed circuits chosen by summarize_hw.py, with the Quantinuum and IBM noise models of hw_compile.py.

rho_ab = <a_a^dag a_b> for the modes of each spin block (openfermion Jordan-Wigner, mode = qubit, occupied = |1>;
block up: qubits 0..B (bath up 0..B-1, impurity up B), block dn: qubits B+1..2B+1 (impurity dn B+1, bath dn)).
Cross-block elements vanish (the block numbers are conserved).

Measurement: setting s appends V_s = exp(-i K_s), K_s = sum_ab k_ab a_a^dag a_b inside each block (the same k for
both blocks), then all qubits are measured in Z. V_s conserves the block numbers, so the post-selection on
(N_up, N_dn) of hw_compile.py applies to every setting. <n_m>_s = sum_cd conj(u_mc) u_md rho_cd, u = exp(-i k),
and rho follows from a least-squares fit over the settings (checked against the exact rho). Settings: the
identity (Z basis, also gives d) and, for each of the B round-robin rounds of (B+1)/2 disjoint mode pairs, two
50:50 pair mixings k = (pi/4) (e^{i phi} a_i^dag a_j + h.c.), phi = 0 and pi/2: 2B + 1 settings
(11 for B = 5). V_s commutes term by term (disjoint pairs), so one Trotter step of its Pauli form is exact.

    python rdm1.py DIR --times ... [--tol 0.01] [--pmem 2e-4] [--drho 2e-3] [--methods swp] [--platforms quantinuum]
                   [--out rdm1] [--noise h2|npj --lam 1 --pmem 0] [--pin rdm1mit]    (writes DIR/<out>_t<t>.json)

The circuit per platform: the cheapest with d error bound < tol among those with compression error in rho <= drho.

Mitigation (mirror circuit, as in noise-estimation circuits): the noisy post-selected state of each block is modeled
as rho_noisy = f rho + (1 - f) I/2 (depolarizing within the sector). The mirror circuit C U^dag (C = X-prep + U,
barrier in between so the compiler keeps both halves) ideally returns the reference configuration c, whose
occupations are 0/1: n_q = f_m c_q + (1 - f_m)/2 gives f_m per block, the forward circuit has f_U = sqrt(f_m), and
setting s (n2q_s two-qubit gates after compilation, n2q_U for U alone) gets f_s = f_U^(n2q_s/n2q_U). The occupations
of each setting are rescaled, n -> (n - (1 - f_s)/2)/f_s, before the fit. d: P11 -> (P11 - (1 - F)/4)/F with
F = f_s,up f_s,dn of the identity setting. "oracle" = one f per block fitted to the known answer (best case).
Also writes DIR/<out>_probs_<platform>_t<t>.npz: exact outcome distributions (axes = logical qubits) of the
settings and the mirror circuit (last), noisy (with readout error) and noiseless, for finite-shot sampling (budget.py).
"""
import argparse, json, os, sys, time
import numpy as np
import h5py
from scipy.linalg import expm
from qiskit import QuantumCircuit, qpy
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aqc_compress import to_qiskit_order
from hw_compile import (H5, PMEAS, load_circuits, quantinuum_compile, quantinuum_noise, with_memory_layers,
                        noise_params, with_init_reset,
                        ibm_compile, probs, readout, d_estimates)
from summarize_hw import PLAT, choose


def hop(i, j, n):
    """a_i^dag a_j (qubit = mode, openfermion JW) as a SparsePauliOp."""
    if i == j:
        return SparsePauliOp.from_sparse_list([("", [], 0.5), ("Z", [i], -0.5)], num_qubits=n)
    lo, hi = min(i, j), max(i, j)
    zs = "Z"*(hi - lo - 1); mid = list(range(lo + 1, hi))
    s = 1 if i < j else -1                     # a_j^dag a_i = (a_i^dag a_j)^dag
    terms = [("X" + zs + "X", 0.25), ("X" + zs + "Y", 0.25j*s), ("Y" + zs + "X", -0.25j*s), ("Y" + zs + "Y", 0.25)]
    return SparsePauliOp.from_sparse_list([(p, [lo] + mid + [hi], c) for p, c in terms], num_qubits=n)


def round_robin(m):
    """m - 1 rounds of m/2 disjoint pairs covering all pairs of m (even) modes."""
    idx = list(range(m)); rounds = []
    for _ in range(m - 1):
        rounds.append([(min(idx[k], idx[m - 1 - k]), max(idx[k], idx[m - 1 - k])) for k in range(m//2)])
        idx = [idx[0]] + [idx[-1]] + idx[1:-1]
    return rounds


def settings(m):
    """One-body generators k (m x m) of the measurement settings."""
    ks = [np.zeros((m, m), complex)]
    for pairs in round_robin(m):
        for phi in (0.0, np.pi/2):
            k = np.zeros((m, m), complex)
            for i, j in pairs:
                k[i, j] = np.pi/4*np.exp(1j*phi); k[j, i] = np.conj(k[i, j])
            ks.append(k)
    return ks


def rotation_circuit(k, blocks, n):
    """V = exp(-i sum_blocks sum_ab k_ab a^dag_a a_b) as a circuit (exact: all terms commute)."""
    qc = QuantumCircuit(n)
    if not np.any(k):
        return qc
    op = SparsePauliOp.from_sparse_list([("", [], 0.0)], num_qubits=n)
    for b in blocks:
        for x in range(len(b)):
            for y in range(len(b)):
                if abs(k[x, y]) > 0:
                    op = op + k[x, y]*hop(b[x], b[y], n)
    op = op.simplify(atol=1e-14)
    assert np.allclose(op.coeffs.imag, 0, atol=1e-12)
    qc.append(PauliEvolutionGate(SparsePauliOp(op.paulis, op.coeffs.real), time=1.0, synthesis=LieTrotter(reps=1)), range(n))
    return qc


def design(ks):
    """Linear map rho (m x m, as real vector [Re, Im]) -> occupations <n_m>_s for all settings."""
    m = ks[0].shape[0]
    rows = []
    for k in ks:
        u = expm(-1j*k)
        for a in range(m):
            w = np.outer(u[a].conj(), u[a])            # <n_a> = sum_cd w_cd rho_cd
            rows.append(np.concatenate([w.real.ravel(), -w.imag.ravel()]))
    return np.array(rows)


def fit_rho(A, occ, m):
    x = np.linalg.lstsq(A, occ, rcond=None)[0]
    rho = (x[:m*m] + 1j*x[m*m:]).reshape(m, m)
    return (rho + rho.conj().T)/2


def occupations(p, blocks, sectors, nb, post_select):
    """<n_q> for all logical qubits from the distribution p (axes = logical qubits), raw or post-selected."""
    n = p.ndim
    idx = np.indices([2]*n)
    if post_select:
        keep = (idx[:nb + 1].sum(0) == sectors[0]) & (idx[nb + 1:].sum(0) == sectors[1])
        p = np.where(keep, p, 0.0)
    acc = p.sum()
    occ = np.array([p[idx[q] == 1].sum() for q in range(n)])/acc
    return occ, float(acc)


def exact_rho(psi_q, block, n):
    sv = Statevector(psi_q)
    return np.array([[sv.expectation_value(hop(a, b, n)) for b in block] for a in block])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--case", default="B5U1.5")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--pmem", type=float, default=2e-4)
    ap.add_argument("--drho", type=float, default=2e-3,
                    help="choose only among circuits with compression error in rho <= this (if any)")
    ap.add_argument("--methods", nargs="+", default=["aqc", "mps", "swp"], help="compression methods to choose from")
    ap.add_argument("--platforms", nargs="+", default=["quantinuum", "ibm"])
    ap.add_argument("--out", default="rdm1")
    ap.add_argument("--noise", default="h2", choices=["h2", "npj"], help="Quantinuum noise model (hw_compile.py)")
    ap.add_argument("--lam", type=float, default=1.0, help="npj model: scale of all error rates")
    ap.add_argument("--pin", help="use the circuit (method, layers) of DIR/<PIN>_t<t>.json instead of choosing")
    a = ap.parse_args()
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeFez
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
    fez = FakeFez()
    sim_ibm = AerSimulator.from_backend(fez, method="density_matrix")
    sim_q = AerSimulator(method="density_matrix", noise_model=quantinuum_noise(a.pmem, a.noise, a.lam))
    nq = noise_params(a.noise, a.lam)
    sched = PassManager([ALAPScheduleAnalysis(fez.target.durations()), PadDelay(target=fez.target)])
    with h5py.File(H5.format(case=a.case), "r") as f:
        t_all = f["/0/t"][()]; st = f["/0/state"]
        n = st.shape[1].bit_length() - 1; nb = (n - 2)//2
        blocks = [list(range(nb + 1)), list(range(nb + 1, n))]
        m = nb + 1
        ks = settings(m)
        A = design(ks)
        assert np.linalg.matrix_rank(A) == m*m, np.linalg.matrix_rank(A)   # a Hermitian m x m matrix: m^2 real numbers
        for tt in a.times:
            i = int(np.argmin(abs(t_all - tt)))
            psi_q = to_qiskit_order(st[i]/np.linalg.norm(st[i]), n)
            hw = json.load(open(os.path.join(a.dir, f"hw_t{t_all[i]:.2f}.json")))
            sectors = tuple(hw["sectors"])
            rho_ex = [exact_rho(psi_q, b, n) for b in blocks]
            # check of the settings and the fit on the exact state
            occ = []
            for k in ks:
                sv = Statevector(psi_q).evolve(rotation_circuit(k, blocks, n))
                pr = np.abs(sv.data)**2
                occ.append([pr[(np.arange(2**n) >> q) & 1 == 1].sum() for q in range(n)])
            occ = np.array(occ)
            for bi, b in enumerate(blocks):
                r = fit_rho(A, occ[:, b].ravel(), m)
                assert np.abs(r - rho_ex[bi]).max() < 1e-10, np.abs(r - rho_ex[bi]).max()
            out = dict(t=float(t_all[i]), step=i, n_qubits=n, blocks=blocks, sectors=sectors, n_settings=len(ks),
                       noise=dict(model=a.noise, lam=a.lam, pmem=a.pmem, **nq),
                       rho_avqds=[[[ [z.real, z.imag] for z in row] for row in r] for r in rho_ex], platforms={})
            print(f"t = {t_all[i]:.2f}: {len(ks)} settings, exact-state fit checked", flush=True)
            for x in hw["rows"]:
                if "memory" in x.get("quantinuum", {}):
                    x["quantinuum"].update(x["quantinuum"]["memory"][f"{a.pmem:g}"])
            for p, name, key, g in PLAT:
                if p not in a.platforms:
                    continue
                if a.pin:
                    pr_ = json.load(open(os.path.join(a.dir, f"{a.pin}_t{t_all[i]:.2f}.json")))["platforms"][p]
                    meth, L = pr_["method"], pr_["layers"]
                else:
                    row, ok = choose([x for x in hw["rows"] if x["method"] in a.methods], p, key, a.tol, a.drho)
                    meth, L = row["method"], row["layers"]
                qc = {(m_, L_): c for m_, L_, c in load_circuits(a.dir, t_all[i], psi_q, n, (meth,))}[(meth, L)]
                rho_c = [exact_rho(Statevector(qc).data, b, n) for b in blocks]
                occ_raw, occ_ps, acc, n2, P, P0 = [], [], [], [], [], []
                tw = time.time()
                U = qc.copy_empty_like()
                for ins in qc.data:
                    if ins.operation.name != "x":
                        U.append(ins)
                mirror = qc.copy(); mirror.barrier(); mirror = mirror.compose(U.inverse())
                conf = np.abs(Statevector(mirror).data)**2
                assert conf.max() > 1 - 1e-8
                conf = np.array([(int(np.argmax(conf)) >> q) & 1 for q in range(n)])
                for k in ks + ["mirror", "U"]:
                    circ = (mirror if k == "mirror" else qc) if isinstance(k, str) else qc.compose(rotation_circuit(k, blocks, n))
                    if p == "quantinuum":
                        qk, info = quantinuum_compile(circ)
                        qm, _ = with_memory_layers(qk)
                        pr = readout(probs(with_init_reset(qm), list(range(n)), sim_q), [nq["pmeas"]]*n)
                        n2.append(dict(n2q=info["n2q"], n1q=info["n1q"], depth2q=info["depth2q"]))
                    else:
                        tq, info = ibm_compile(circ, fez)
                        phys = tq.layout.final_index_layout()[:n]
                        pr = readout(probs(sched.run(tq), phys, sim_ibm), [fez.target["measure"][(q,)].error for q in phys])
                        n2.append(dict(ncz=info["ncz"], depth2q=info["depth2q"]))
                    P.append(pr)
                    P0.append(np.abs(Statevector(circ).data.reshape([2]*n).transpose(list(range(n - 1, -1, -1))))**2)
                    if not isinstance(k, str) and k is ks[0]:
                        p11 = d_estimates(pr, nb, sectors)[3]
                    o, _ = occupations(pr, blocks, sectors, nb, False); occ_raw.append(o)
                    o, ac = occupations(pr, blocks, sectors, nb, True); occ_ps.append(o); acc.append(ac)
                info_m, info_u = n2.pop(-2), n2.pop()
                occ_m, occ_u = occ_ps.pop(-2), occ_ps.pop(); occ_raw = occ_raw[:-2]; acc_m = acc.pop(-2); acc.pop()
                occ_raw, occ_ps = np.array(occ_raw), np.array(occ_ps)
                res = dict(method=meth, layers=L, settings=n2, acceptance=acc)
                for lab, oc in (("raw", occ_raw), ("ps", occ_ps)):
                    res[f"rho_{lab}"] = [[[[z.real, z.imag] for z in row] for row in fit_rho(A, oc[:, b].ravel(), m)]
                                         for b in blocks]
                    res[f"err_{lab}"] = float(max(np.abs(fit_rho(A, oc[:, b].ravel(), m) - rho_ex[bi]).max()
                                                  for bi, b in enumerate(blocks)))
                res["err_compression"] = float(max(np.abs(rho_c[bi] - rho_ex[bi]).max() for bi in range(2)))
                res["bias_ps"] = float(max(np.abs(fit_rho(A, occ_ps[:, b].ravel(), m) - rho_c[bi]).max()
                                           for bi, b in enumerate(blocks)))
                # mirror-circuit rescaling
                k2 = "n2q" if p == "quantinuum" else "ncz"
                f_m = [1 - 2*np.mean(np.abs(occ_m[b] - conf[b])) for b in blocks]
                f_u = np.sqrt(f_m)
                f_s = np.array([[fu**(s[k2]/info_u[k2]) for fu in f_u] for s in n2])      # settings x blocks
                occ_mit = occ_ps.copy()
                for bi, b in enumerate(blocks):
                    occ_mit[:, b] = (occ_ps[:, b] - (1 - f_s[:, [bi]])/2)/f_s[:, [bi]]
                rho_mit = [fit_rho(A, occ_mit[:, b].ravel(), m) for b in blocks]
                rho_ps = [fit_rho(A, occ_ps[:, b].ravel(), m) for b in blocks]
                f_or = []
                for bi in range(2):                     # rho_ps = f rho_ex + (1 - f) I/2, least squares in f
                    x = (rho_ex[bi] - np.eye(m)/2).ravel(); y = (rho_ps[bi] - np.eye(m)/2).ravel()
                    f_or.append(float(np.real(np.vdot(x, y))/np.vdot(x, x).real))
                rho_or = [(rho_ps[bi] - (1 - f_or[bi])/2*np.eye(m))/f_or[bi] for bi in range(2)]
                F = f_s[0, 0]*f_s[0, 1]
                d_ex = 0.5 - float(Statevector(psi_q).probabilities_dict(qargs=[nb, nb + 1]).get("11", 0.0))
                res.update(mirror=dict(n2q=info_m[k2], acceptance=acc_m, f_m=f_m, f_u=list(f_u),
                                       f_identity=list(f_s[0]), f_oracle=f_or),
                           occ_ps=occ_ps.tolist(), occ_mirror=occ_m.tolist(), n2q_U=info_u[k2],
                           rho_mit=[[[[z.real, z.imag] for z in row] for row in r] for r in rho_mit],
                           err_mit=float(max(np.abs(rho_mit[bi] - rho_ex[bi]).max() for bi in range(2))),
                           err_oracle=float(max(np.abs(rho_or[bi] - rho_ex[bi]).max() for bi in range(2))),
                           d_avqds=d_ex, d_ps=0.5 - p11, d_mit=0.5 - (p11 - (1 - F)/4)/F)
                out["platforms"][p] = res
                np.savez_compressed(os.path.join(a.dir, f"{a.out}_probs_{p}_t{t_all[i]:.2f}.npz"),
                                    noisy=np.array(P[:-1]), noiseless=np.array(P0[:-1]), conf=conf,
                                    f_exponent=np.array([s[k2]/info_u[k2] for s in n2]))
                print(f"  {name}: {meth} {L}, 2q gates per setting {min(s[k2] for s in n2)}-{max(s[k2] for s in n2)}, "
                      f"max |rho - rho_AVQDS|: compression {res['err_compression']:.4f}, raw {res['err_raw']:.4f}, "
                      f"ps {res['err_ps']:.4f}, mirror-mitigated {res['err_mit']:.4f}, oracle {res['err_oracle']:.4f}; "
                      f"f_u {f_u[0]:.4f} {f_u[1]:.4f}, f_oracle {f_or[0]:.4f} {f_or[1]:.4f}; "
                      f"d - d_AVQDS: ps {res['d_ps'] - d_ex:+.4f}, mitigated {res['d_mit'] - d_ex:+.4f}; "
                      f"acceptance {min(acc):.2f}-{max(acc):.2f}  {time.time()-tw:.0f}s", flush=True)
            json.dump(out, open(os.path.join(a.dir, f"{a.out}_t{t_all[i]:.2f}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
