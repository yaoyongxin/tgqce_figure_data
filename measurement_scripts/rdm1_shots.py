"""One-particle density matrix with the Quantinuum noise model and finite shots: three measurement schemes at
equal total shots, for the circuit chosen for Quantinuum in DIR/rdm1_t<t>.json (rdm1.py).

- orbital: the 2B+1 number-conserving orbital-rotation settings of rdm1.py (+ <= 30 ZZ each, noisy), Z
  measurement, post-selection on (N_up, N_dn), least-squares fit. Shots split evenly over the settings.
- qwc: the Jordan-Wigner Pauli strings of rho (X Z..Z X, X Z..Z Y, Y Z..Z X, Y Z..Z Y for i < j, Z_i; both
  blocks) grouped into qubit-wise commuting sets (greedy, largest-first); one setting per set (unused qubits Z),
  shots split evenly; each string averaged over all settings that measure it. No post-selection is possible
  (X/Y measurements do not conserve the particle numbers).
- shadow: classical shadows with random Pauli bases (uniform X/Y/Z per qubit), estimator 3^w prod(+-1) on
  matching bases; --reps shots per random basis (1 = textbook shadows; Quantinuum charges per circuit,
  so in practice many shots per basis).
qwc and shadow use readout correction (divide a weight-w string by prod (1 - 2 eps_q); eps is known in the model).
The error of the basis-change one-qubit gate of a qubit measured in X or Y (depolarizing dep1) is a bit flip of
probability dep1/2 and is added to its readout flip (eps_XY = eps + dep1/2 - eps dep1).

Sampling is exact for the noisy model: the noisy state rho_noisy (density-matrix simulation of the compiled
circuit with memory error) is diagonalized, each shot draws an eigenvector, rotates it to its basis and draws a
bit string; readout flips are drawn per bit. Errors: max_ab |rho_ab - rho_AVQDS_ab| over both blocks, per trial.

    python rdm1_shots.py DIR --times ... [--shots 1000 10000 100000] [--trials 20] [--reps 1 100] [--pmem 2e-4]
                         [--noise h2|npj --lam 1 --pmem 0] [--pin rdm1mit] [--out rdm1_shots]

writes DIR/<out>_t<t>.json; the circuit is that of DIR/rdm1_t<t>.json, or of DIR/<pin>_t<t>.json with --pin.
"""
import argparse, json, os, sys, time
import numpy as np
import h5py
from qiskit.quantum_info import DensityMatrix, SparsePauliOp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aqc_compress import to_qiskit_order
from hw_compile import (H5, PMEAS, load_circuits, quantinuum_compile, quantinuum_noise, with_memory_layers,
                        probs, readout, noise_params, with_init_reset)
from rdm1 import settings, rotation_circuit, design, fit_rho, occupations, exact_rho

H = np.array([[1, 1], [1, -1]])/np.sqrt(2)
BASIS = [H, H @ np.diag([1, -1j]), np.eye(2)]     # rotate X, Y, Z eigenbases to Z (codes 0, 1, 2)


def pauli_strings(blocks):
    """[(qubits, codes)] of all strings in rho, and the map rho_ab = sum_s c_s <s> (per block, a, b)."""
    strings, idx, terms = [], {}, []
    def add(qs, cs):
        key = (tuple(qs), tuple(cs))
        if key not in idx:
            idx[key] = len(strings); strings.append(key)
        return idx[key]
    for bi, b in enumerate(blocks):
        for x, i in enumerate(b):
            terms.append((bi, x, x, [(add([i], [2]), -0.5)], 0.5))          # n_i = (1 - Z_i)/2
            for y, j in enumerate(b[x + 1:], x + 1):
                qs = list(range(i, j + 1)); z = [2]*(j - i - 1)
                # a_i^dag a_j = (XX + i XY - i YX + YY)/4 with Z strings between (i < j)
                tt = [(add(qs, [0] + z + [0]), 0.25), (add(qs, [0] + z + [1]), 0.25j),
                      (add(qs, [1] + z + [0]), -0.25j), (add(qs, [1] + z + [1]), 0.25)]
                terms.append((bi, x, y, tt, 0.0))
    return strings, terms


def assemble(vals, terms, m, nblk):
    rho = np.zeros((nblk, m, m), complex)
    for bi, x, y, tt, c0 in terms:
        v = c0 + sum(c*vals[s] for s, c in tt)
        rho[bi, x, y] = v
        rho[bi, y, x] = np.conj(v)
    return rho


def qwc_groups(strings, n):
    """Greedy largest-first grouping into qubit-wise commuting sets; returns the basis codes per group."""
    order = sorted(range(len(strings)), key=lambda s: -len(strings[s][0]))
    groups = []
    for s in order:
        qs, cs = strings[s]
        for g in groups:
            if all(g[q] in (-1, c) for q, c in zip(qs, cs)):
                for q, c in zip(qs, cs):
                    g[q] = c
                break
        else:
            g = np.full(n, -1); g[list(qs)] = cs; groups.append(g)
    return [np.where(g < 0, 2, g) for g in groups]


class Sampler:
    """Exact shot sampler for a noisy n-qubit state rho (qiskit order) measured in per-shot Pauli bases; eps: readout
    flip probability per basis code (X, Y, Z)."""
    def __init__(self, rho, n, eps, rng):
        lam, V = np.linalg.eigh(rho)
        keep = lam > 1e-12
        self.lam = lam[keep]/lam[keep].sum(); self.V = V[:, keep]
        self.n, self.eps, self.rng = n, eps, rng

    def sample(self, bases):
        """bases: (N, n) codes; returns bits (N, n) in {0, 1} (column q = qubit q), with readout flips."""
        n, out = self.n, []
        for s in range(0, len(bases), 1000):
            b = bases[s:s + 1000]; B = len(b)
            k = self.rng.choice(len(self.lam), B, p=self.lam)
            X = self.V[:, k].T.reshape([B] + [2]*n)              # axis 1 + (n - 1 - q) = qubit q
            for q in range(n):
                U = np.stack(BASIS)[b[:, q]]                       # (B, 2, 2)
                ax = 1 + n - 1 - q
                X = np.moveaxis(X, ax, 1)
                sh = X.shape
                X = np.einsum("bij,bjr->bir", U, X.reshape(B, 2, -1)).reshape(sh)
                X = np.moveaxis(X, 1, ax)
            p = np.abs(X.reshape(B, -1))**2
            c = np.cumsum(p, axis=1)
            x = (c < self.rng.random((B, 1))*c[:, -1:]).sum(1)
            out.append((x[:, None] >> np.arange(n)) & 1)
        bits = np.concatenate(out)
        return bits ^ (self.rng.random(bits.shape) < np.asarray(self.eps)[bases])


def pauli_estimates(bits, bases, strings, eps, shadow):
    """Estimates of all strings: shadow (3^w prod over matching shots / N) or grouped (mean over matching shots),
    readout corrected."""
    sgn = 1 - 2*bits
    out = np.zeros(len(strings))
    for s, (qs, cs) in enumerate(strings):
        qs = list(qs)
        match = np.all(bases[:, qs] == cs, axis=1)
        val = np.prod(sgn[:, qs], axis=1)
        w = len(qs)
        est = 3**w*np.mean(match*val) if shadow else (val[match].mean() if match.any() else 0.0)
        out[s] = est/np.prod([1 - 2*eps[c] for c in cs])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--case", default="B5U1.5")
    ap.add_argument("--shots", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 100], help="shadow shots per random basis")
    ap.add_argument("--pmem", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--noise", default="h2", choices=["h2", "npj"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--pin", help="circuit and sectors from DIR/<PIN>_t<t>.json (rdm1.py output)")
    ap.add_argument("--out", default="rdm1_shots")
    a = ap.parse_args()
    from qiskit_aer import AerSimulator
    sim_q = AerSimulator(method="density_matrix", noise_model=quantinuum_noise(a.pmem, a.noise, a.lam))
    nq = noise_params(a.noise, a.lam)
    pm = nq["pmeas"]; pxy = pm + nq["dep1"]/2 - pm*nq["dep1"]
    eps3 = [pxy, pxy, pm]                                  # readout flip per basis code X, Y, Z
    rng = np.random.default_rng(a.seed)
    with h5py.File(H5.format(case=a.case), "r") as f:
        t_all = f["/0/t"][()]; st = f["/0/state"]
        n = st.shape[1].bit_length() - 1; nb = (n - 2)//2
        blocks = [list(range(nb + 1)), list(range(nb + 1, n))]; m = nb + 1
        strings, terms = pauli_strings(blocks)
        groups = qwc_groups(strings, n)
        ks = settings(m); A = design(ks)
        print(f"{len(strings)} Pauli strings, {len(groups)} QWC groups, {len(ks)} orbital settings", flush=True)
        for tt in a.times:
            i = int(np.argmin(abs(t_all - tt)))
            tw = time.time()
            psi_q = to_qiskit_order(st[i]/np.linalg.norm(st[i]), n)
            rho_ex = np.array([exact_rho(psi_q, b, n) for b in blocks])
            r1 = json.load(open(os.path.join(a.dir, f"{a.pin or 'rdm1'}_t{t_all[i]:.2f}.json")))
            sectors = tuple(r1["sectors"])
            meth, L = r1["platforms"]["quantinuum"]["method"], r1["platforms"]["quantinuum"]["layers"]
            qc = {(m_, L_): c for m_, L_, c in load_circuits(a.dir, t_all[i], psi_q, n, (meth,))}[(meth, L)]
            # check of the Pauli assembly on the exact state
            dm_ex = DensityMatrix(psi_q)
            ops = [SparsePauliOp.from_sparse_list([("".join("XYZ"[c] for c in cs), list(qs), 1)], num_qubits=n)
                   for qs, cs in strings]
            assert np.abs(assemble([dm_ex.expectation_value(o).real for o in ops], terms, m, 2) - rho_ex).max() < 1e-10
            # noisy state of the compiled circuit (memory error) and its exact Pauli expectations
            qk, info = quantinuum_compile(qc)
            qm, _ = with_memory_layers(qk)
            qm = with_init_reset(qm)
            qm.save_density_matrix()
            rho_n = np.asarray(sim_q.run(qm).result().data()["density_matrix"])
            dm = DensityMatrix(rho_n)
            pv = np.array([dm.expectation_value(o).real for o in ops])
            err = lambda r: float(np.abs(r - rho_ex).max())
            # readout flips scale a weight-w string by (1 - 2 eps)^w: raw = that, corrected = the noisy-state value
            ro = np.array([np.prod([1 - 2*eps3[c] for c in cs]) for _, cs in strings])
            res = dict(t=float(t_all[i]), circuit=f"{meth}{L}", n2q=info["n2q"], n_strings=len(strings),
                       noise=dict(model=a.noise, lam=a.lam, pmem=a.pmem, **nq),
                       n_qwc_groups=len(groups), n_orbital_settings=len(ks), sectors=sectors, shots={},
                       bias=dict(pauli_raw=err(assemble(pv*ro, terms, m, 2)),
                                 pauli_readout_corrected=err(assemble(pv, terms, m, 2))))
            # orbital settings: exact noisy distributions (rotation circuits included)
            p_set, n2_set = [], []
            for k in ks:
                qk2, info2 = quantinuum_compile(qc.compose(rotation_circuit(k, blocks, n)))
                qm2, _ = with_memory_layers(qk2)
                p_set.append(readout(probs(with_init_reset(qm2), list(range(n)), sim_q), [pm]*n).ravel())
                n2_set.append(info2["n2q"])
            occ = np.array([occupations(p.reshape([2]*n), blocks, sectors, nb, True)[0] for p in p_set])
            res["bias"]["orbital_ps"] = err(np.array([fit_rho(A, occ[:, b].ravel(), m) for b in blocks]))
            res["n2q_orbital_settings"] = n2_set
            sampler = Sampler(rho_n, n, eps3, rng)
            idx = np.indices([2]*n).reshape(n, -1)           # axis q of p (logical qubit q) -> bits
            keep = (idx[:nb + 1].sum(0) == sectors[0]) & (idx[nb + 1:].sum(0) == sectors[1])
            print(f"t = {t_all[i]:.2f}: {meth}{L} ({info['n2q']} ZZ), infinite-shot bias: orbital (ps) "
                  f"{res['bias']['orbital_ps']:.4f}, Pauli raw {res['bias']['pauli_raw']:.4f}, "
                  f"readout-corrected {res['bias']['pauli_readout_corrected']:.4f}  {time.time()-tw:.0f}s", flush=True)
            for N in a.shots:
                out = {}
                # orbital
                e = []
                for _ in range(a.trials):
                    occ = []
                    for p in p_set:
                        c = rng.multinomial(N//len(ks), p/p.sum())*keep
                        occ.append([(c*idx[q]).sum()/c.sum() for q in range(n)])
                    occ = np.array(occ)
                    e.append(err(np.array([fit_rho(A, occ[:, b].ravel(), m) for b in blocks])))
                out["orbital"] = e
                # qwc
                e = []
                for _ in range(a.trials):
                    bases = np.repeat(np.array(groups), N//len(groups), axis=0)
                    bits = sampler.sample(bases)
                    e.append(err(assemble(pauli_estimates(bits, bases, strings, eps3, False), terms, m, 2)))
                out["qwc"] = e
                # shadows
                for r in a.reps:
                    if r > N:
                        continue
                    e = []
                    for _ in range(a.trials):
                        bases = np.repeat(rng.integers(0, 3, (N//r, n)), r, axis=0)
                        bits = sampler.sample(bases)
                        e.append(err(assemble(pauli_estimates(bits, bases, strings, eps3, True), terms, m, 2)))
                    out[f"shadow_r{r}"] = e
                res["shots"][str(N)] = out
                print(f"  N = {N:6d}: median max-error  " + "  ".join(f"{k} {np.median(v):.4f}" for k, v in out.items())
                      + f"  {time.time()-tw:.0f}s", flush=True)
            json.dump(res, open(os.path.join(a.dir, f"{a.out}_t{t_all[i]:.2f}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
