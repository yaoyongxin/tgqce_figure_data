"""Number-conserving fermionic (matchgate) classical shadows of the one-particle density matrix (Low,
arXiv:2208.08964), with the Quantinuum noise model and finite shots; companion of rdm1_shots.py (same circuit,
same error measure).

Each shot applies an independent Haar-random orbital rotation u in U(m) to each spin block (m = B + 1 modes),
measures all qubits in Z and is post-selected on (N_up, N_dn). With b_a the measured occupation of rotated
orbital a (<b_a> = sum_cd conj(u_ac) u_ad rho_cd) the snapshot is
    rho_hat = (m + 1) u^T diag(b) conj(u) - N I       (N = particles in the block),
unbiased because E_q[|q><q| <q|X|q>] = (X + Tr X I)/(m (m + 1)) for Haar unit vectors q (rows of u).

Circuit of u: Givens decomposition u = D R_L^dag ... R_1^dag into m(m-1)/2 rotations of neighbouring modes
(neighbouring qubits, no Jordan-Wigner strings; the diagonal D does not change occupations and is dropped).
A two-mode rotation W acts on |n_c n_{c+1}> as 1 on |00>, W on the one-particle states, det W on |11>.

Noise (exact trajectory sampling of the Pauli channels): the state-preparation part is the noisy density matrix
of the compiled circuit (as rdm1_shots.py, eigenvector per shot); each Givens rotation compiles to two ZZPhase
gates (checked with pytket below), whose depolarizing channels commute with the rotation and combine into one,
lambda = 1 - (1 - 4/3 p2)^2; memory dephasing 2 p_mem (1 - p_mem) per qubit per Givens layer (ASAP layering,
both blocks in parallel); readout flips per bit. One-qubit gate errors of the rotation are neglected (3e-5 in the
h2 model; with the npj model, 4e-4 per gate, this makes the fermionic shadows slightly optimistic).

    python rdm1_fshadow.py DIR --times ... [--shots 1000 10000 100000] [--trials 20] [--bias-shots 1000000]
                           [--noise h2|npj --lam 1 --pmem 0] [--pin rdm1mit] [--out rdm1_fshadow]
    python rdm1_fshadow.py DIR --times 1.5 --check       (noiseless and Aer cross-checks only)

writes DIR/<out>_t<t>.json; circuit of DIR/rdm1_t<t>.json, or DIR/<pin>_t<t>.json with --pin.
"""
import argparse, json, os, sys, time
import numpy as np
import h5py
from scipy.stats import unitary_group
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aqc_compress import to_qiskit_order
from hw_compile import (H5, PMEAS, P2Q, load_circuits, quantinuum_compile, quantinuum_noise, with_memory_layers,
                        noise_params, with_init_reset,
                        probs, readout)
from rdm1 import exact_rho

PAULI = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
PAULI2 = np.array([np.kron(a, b) for a in PAULI for b in PAULI])[1:]      # 15 non-identity two-qubit Paulis


def givens(u):
    """Batched Givens decomposition. u: (S, m, m). Returns the mode pairs [(c, c+1)] (time order) and the
    two-mode rotations W (S, L, 2, 2) with u = D W_L ... W_1 (embedded), D diagonal."""
    M = u.copy(); S, m, _ = u.shape
    pairs, Ws = [], []
    for r in range(m - 1, 0, -1):
        for c in range(r):
            a, b = M[:, r, c], M[:, r, c + 1]
            rr = np.sqrt(abs(a)**2 + abs(b)**2)
            z = rr < 1e-14
            rr = np.where(z, 1, rr)
            R = np.empty((S, 2, 2), complex)                 # (a, b) R = (0, rr)
            R[:, 0, 0] = np.where(z, 1, b/rr); R[:, 0, 1] = np.where(z, 0, a.conj()/rr)
            R[:, 1, 0] = np.where(z, 0, -a/rr); R[:, 1, 1] = np.where(z, 1, b.conj()/rr)
            M[:, :, c:c + 2] = M[:, :, c:c + 2] @ R
            pairs.append((c, c + 1)); Ws.append(R.conj().transpose(0, 2, 1))
    assert np.allclose(np.abs(M), np.eye(m)[None], atol=1e-10)
    return pairs, np.stack(Ws, axis=1)


def mode_gate(W):
    """4x4 qubit gate (little endian: index n_c + 2 n_{c+1}) of the two-mode rotation W (..., 2, 2)."""
    G = np.zeros(W.shape[:-2] + (4, 4), complex)
    G[..., 0, 0] = 1
    G[..., 1:3, 1:3] = W
    G[..., 3, 3] = np.linalg.det(W)
    return G


_X = np.array([[0, 1], [1, 0]]); _Y = np.array([[0, -1j], [1j, 0]]); _I = np.eye(2)
XX, YY = np.kron(_X, _X), np.kron(_Y, _Y)
S0 = np.kron(_I, np.diag([1, 1j]))                       # S on qubit c (right factor, little endian)


def phases(c, d):
    return np.stack([np.ones_like(c), np.exp(1j*c), np.exp(1j*d), np.exp(1j*(c + d))], -1)[..., None, :]*np.eye(4)


def split_gate(W):
    """mode_gate(W) = D_out S0 XX(-g) YY(-g) S0^dag D_in, with XX(t) = exp(-i t XX/2) etc.: the form in which a
    Givens rotation is run on hardware (virtual Rz phases and two ZZPhase gates in rotated frames).
    Returns (A, B, C, Dg): mode_gate = A @ B @ C with A = D_out S0, B = XX(-g), C = YY(-g) S0^dag D_in."""
    g = np.arccos(np.clip(abs(W[:, 0, 0]), 0, 1))
    a1 = np.angle(W[:, 0, 0])
    small = np.sin(g) < 1e-9
    b2 = np.where(small, np.angle(W[:, 1, 1]), np.angle(-W[:, 0, 1]) - a1)
    a2 = np.where(small, 0.0, np.angle(W[:, 1, 0]))
    rot = lambda P, t: np.cos(t/2)[:, None, None]*np.eye(4) - 1j*np.sin(t/2)[:, None, None]*P
    A = phases(a1, a2) @ S0
    B = rot(XX, -g)
    C = rot(YY, -g) @ S0.conj().T @ phases(np.zeros_like(b2), b2)
    assert np.allclose(A @ B @ C, mode_gate(W), atol=1e-10)
    return A, B, C


def layers(pairs):
    """ASAP layer index of each gate of the sequence."""
    clock, out = {}, []
    for c, d in pairs:
        L = 1 + max(clock.get(c, 0), clock.get(d, 0))
        clock[c] = clock[d] = L
        out.append(L)
    return out


def apply2(X, G, qa, qb, n):
    """X (B, 2..2) qiskit order (axis 1 + n-1-q = qubit q); G (B, 4, 4) on (qa, qb), index n_qa + 2 n_qb."""
    B = X.shape[0]
    X = np.moveaxis(X, [1 + n - 1 - qb, 1 + n - 1 - qa], [1, 2])
    sh = X.shape
    X = np.einsum("bij,bjr->bir", G, X.reshape(B, 4, -1)).reshape(sh)
    return np.moveaxis(X, [1, 2], [1 + n - 1 - qb, 1 + n - 1 - qa])


class FermionicShadow:
    def __init__(self, rho, n, blocks, rng, p2=P2Q, pmem=2e-4, eps=PMEAS, noisy=True):
        lam, V = np.linalg.eigh(rho)
        keep = lam > 1e-12
        self.lam = lam[keep]/lam[keep].sum(); self.V = V[:, keep]
        self.n, self.blocks, self.rng, self.m = n, blocks, rng, len(blocks[0])
        lam1 = 4/3*p2
        self.pdep = (1 - (1 - lam1)**2)*15/16 if noisy else 0.0      # probability of a non-identity Pauli
        self.pmem = pmem if noisy else 0.0
        self.pz = 2*pmem*(1 - pmem) if noisy else 0.0
        self.eps = eps if noisy else 0.0

    def shots(self, S, u=None):
        """S shots: returns the per-block rotations u (S, 2, m, m; Haar random unless given) and bits (S, n)."""
        n, m, rng = self.n, self.m, self.rng
        if u is None:
            u = unitary_group.rvs(m, size=2*S, random_state=rng).reshape(S, 2, m, m)
        dec = [givens(u[:, k]) for k in range(2)]
        pairs = dec[0][0]
        lay = layers(pairs)
        k = rng.choice(len(self.lam), S, p=self.lam)
        X = self.V[:, k].T.reshape([S] + [2]*n)
        P1 = np.stack([_I, _X, _Y])                             # one-qubit I, X, Y
        for L in range(1, max(lay) + 1):
            busy = set()
            for g, (c, d) in enumerate(pairs):
                if lay[g] != L:
                    continue
                for bi, b in enumerate(self.blocks):
                    A, B, C = split_gate(dec[bi][1][:, g])
                    # memory dephasing during the two ZZ layers of the gate: in the logical frame a Y error
                    # after the YY half and an X error after the XX half (per qubit, prob p_mem each)
                    ym = P1[2*(rng.random((S, 2)) < self.pmem)]           # (S, 2, 2, 2): I or Y on (c, d)
                    xm = P1[1*(rng.random((S, 2)) < self.pmem)]
                    kron = lambda e: np.einsum("sij,skl->sikjl", e[:, 1], e[:, 0]).reshape(S, 4, 4)
                    G = A @ kron(xm) @ B @ kron(ym) @ C
                    hit = rng.random(S) < self.pdep
                    if hit.any():
                        G[hit] = PAULI2[rng.integers(0, 15, hit.sum())] @ G[hit]
                    X = apply2(X, G, b[c], b[d], n)
                    busy |= {b[c], b[d]}
            for q in set(range(n)) - busy:                       # idle qubits: Z dephasing over the 2 ZZ layers
                hit = rng.random(S) < self.pz
                if hit.any():
                    ax = 1 + n - 1 - q
                    Y = np.moveaxis(X, ax, 1)
                    Y[hit, 1] *= -1
                    X = np.moveaxis(Y, 1, ax)
        p = np.abs(X.reshape(S, -1))**2
        c = np.cumsum(p, axis=1)
        x = (c < rng.random((S, 1))*c[:, -1:]).sum(1)
        bits = (x[:, None] >> np.arange(n)) & 1
        return u, bits ^ (rng.random(bits.shape) < self.eps)

    def estimate(self, N, sectors, batch=1000):
        """rho estimate (2, m, m) from N shots (post-selected), and the acceptance."""
        m = self.m
        acc_sum = np.zeros((2, m, m), complex); nacc = 0
        for s in range(0, N, batch):
            u, bits = self.shots(min(batch, N - s))
            nblk = np.stack([bits[:, b].sum(1) for b in self.blocks], axis=1)
            ok = np.all(nblk == np.array(sectors), axis=1)
            for bi, b in enumerate(self.blocks):
                bb = bits[ok][:, b].astype(float)
                acc_sum[bi] += np.einsum("sac,sa,sad->cd", u[ok, bi], bb, u[ok, bi].conj())
            nacc += ok.sum()
        rho = (m + 1)*acc_sum/nacc - np.array(sectors)[:, None, None]*np.eye(m)[None]
        return rho, nacc/N


def circuit_of(u_blocks, blocks, n):
    """qiskit circuit of the orbital rotations (one u per block), from the Givens decomposition."""
    qc = QuantumCircuit(n)
    dec = [givens(u[None]) for u in u_blocks]
    for g, (c, d) in enumerate(dec[0][0]):
        for bi, b in enumerate(blocks):
            qc.append(UnitaryGate(mode_gate(dec[bi][1][0, g]), check_input=False), [b[c], b[d]])
    return qc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--case", default="B5U1.5")
    ap.add_argument("--shots", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--bias-shots", type=int, default=1000000)
    ap.add_argument("--pmem", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--noise", default="h2", choices=["h2", "npj"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--pin")
    ap.add_argument("--out", default="rdm1_fshadow")
    a = ap.parse_args()
    from qiskit_aer import AerSimulator
    sim_q = AerSimulator(method="density_matrix", noise_model=quantinuum_noise(a.pmem, a.noise, a.lam))
    nq = noise_params(a.noise, a.lam)
    kw = dict(p2=3/4*nq["dep2"], pmem=a.pmem, eps=nq["pmeas"])    # FermionicShadow takes the average infidelity
    rng = np.random.default_rng(a.seed)
    with h5py.File(H5.format(case=a.case), "r") as f:
        t_all = f["/0/t"][()]; st = f["/0/state"]
        n = st.shape[1].bit_length() - 1; nb = (n - 2)//2
        blocks = [list(range(nb + 1)), list(range(nb + 1, n))]; m = nb + 1
        for tt in a.times:
            i = int(np.argmin(abs(t_all - tt)))
            tw = time.time()
            psi_q = to_qiskit_order(st[i]/np.linalg.norm(st[i]), n)
            rho_ex = np.array([exact_rho(psi_q, b, n) for b in blocks])
            r1 = json.load(open(os.path.join(a.dir, f"{a.pin or 'rdm1'}_t{t_all[i]:.2f}.json")))
            sectors = tuple(r1["sectors"])
            meth, L = r1["platforms"]["quantinuum"]["method"], r1["platforms"]["quantinuum"]["layers"]
            qc = {(m_, L_): c for m_, L_, c in load_circuits(a.dir, t_all[i], psi_q, n, (meth,))}[(meth, L)]
            err = lambda r: float(np.abs(r - rho_ex).max())
            # check 1: circuit of a random rotation gives <n_a> = sum_cd conj(u_ac) u_ad rho_cd (exact state)
            ub = unitary_group.rvs(m, size=2, random_state=rng)
            rc = circuit_of(ub, blocks, n)
            sv = Statevector(psi_q).evolve(rc)
            pr = np.abs(sv.data)**2
            occ = np.array([pr[(np.arange(2**n) >> q) & 1 == 1].sum() for q in range(n)])
            pred = np.concatenate([np.einsum("ac,ad,cd->a", ub[bi].conj(), ub[bi], rho_ex[bi]).real for bi in range(2)])
            assert np.abs(occ - pred).max() < 1e-10, np.abs(occ - pred).max()
            # compiled cost of one random rotation (both blocks)
            _, info_r = quantinuum_compile(rc)
            _, info_t = quantinuum_compile(qc.compose(rc))
            res = dict(t=float(t_all[i]), circuit=f"{meth}{L}", noise=dict(model=a.noise, lam=a.lam, pmem=a.pmem, **nq),
                       n2q_rotation=info_r["n2q"], depth2q_rotation=info_r["depth2q"],
                       n2q_total=info_t["n2q"], sectors=sectors, shots={})
            print(f"t = {t_all[i]:.2f}: {meth}{L}; random rotation (both blocks) {info_r['n2q']} ZZ, 2q depth "
                  f"{info_r['depth2q']} ({len(givens(ub[:1])[0])} Givens per block, {max(layers(givens(ub[:1])[0]))} layers); "
                  f"total {info_t['n2q']} ZZ", flush=True)
            # noisy state after the state preparation
            qk, info = quantinuum_compile(qc)
            qm, _ = with_memory_layers(qk)
            qm = with_init_reset(qm)
            qm.save_density_matrix()
            rho_n = np.asarray(sim_q.run(qm).result().data()["density_matrix"])
            if a.check:
                # check 2: noiseless shadow from the exact state converges to rho
                fs0 = FermionicShadow(np.outer(psi_q, psi_q.conj()), n, blocks, rng, noisy=False)
                r0, _ = fs0.estimate(200000, sectors)
                print(f"  noiseless, 2e5 shots: max error {err(r0):.4f}", flush=True)
                # check 3: trajectory sampler vs Aer density matrix for one fixed rotation (post-selected <n_q>)
                fs = FermionicShadow(rho_n, n, blocks, rng, **kw)
                bits = np.concatenate([fs.shots(1000, np.broadcast_to(ub, (1000, 2, m, m)).copy())[1] for _ in range(100)])
                nblk = np.stack([bits[:, b].sum(1) for b in blocks], axis=1); ok = np.all(nblk == np.array(sectors), axis=1)
                occ_t = bits[ok].mean(0)
                qk2, _ = quantinuum_compile(qc.compose(rc))
                qm2, _ = with_memory_layers(qk2)
                p = readout(probs(with_init_reset(qm2), list(range(n)), sim_q), [nq["pmeas"]]*n).reshape(-1)
                idx = np.indices([2]*n).reshape(n, -1)
                keep = (idx[:nb + 1].sum(0) == sectors[0]) & (idx[nb + 1:].sum(0) == sectors[1])
                occ_a = np.array([(p*keep*idx[q]).sum() for q in range(n)])/(p*keep).sum()
                print(f"  fixed rotation, post-selected <n_q>: trajectories vs Aer max diff {np.abs(occ_t - occ_a).max():.4f} "
                      f"(shot noise ~{0.5/np.sqrt(ok.sum()):.4f}); acceptance {ok.mean():.3f} vs {(p*keep).sum():.3f}; "
                      f"vs noiseless max diff {np.abs(occ_a - pred).max():.4f}", flush=True)
                continue
            fs = FermionicShadow(rho_n, n, blocks, rng, **kw)
            rb, acc = fs.estimate(a.bias_shots, sectors)
            res["bias_estimate"] = dict(shots=a.bias_shots, max_error=err(rb), acceptance=float(acc))
            print(f"  {a.bias_shots} shots: max error {err(rb):.4f} (bias + shot noise), acceptance {acc:.3f}  "
                  f"{time.time()-tw:.0f}s", flush=True)
            for N in a.shots:
                e = [err(fs.estimate(N, sectors)[0]) for _ in range(a.trials)]
                res["shots"][str(N)] = dict(fshadow=e)
                print(f"  N = {N:6d}: median max-error fshadow {np.median(e):.4f}  {time.time()-tw:.0f}s", flush=True)
            json.dump(res, open(os.path.join(a.dir, f"{a.out}_t{t_all[i]:.2f}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
