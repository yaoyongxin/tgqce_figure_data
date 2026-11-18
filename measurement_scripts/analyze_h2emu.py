"""Estimates of rho and d from the H2-Emulator counts (counts_t<t>.npz), with the estimator of budget.py --sym:
post-selection on (N_up, N_dn) ("noisy") and, in addition, the mirror-circuit rescaling ("mitigated").

The point estimate uses the measured counts; the 5% and 95% quantiles come from a parametric bootstrap that
redraws each circuit's shots from its measured distribution (--trials, default 500). The noise-model results at the
same time (budget_sym_rdm1npj1.json, kappa = 1) are printed for comparison.

    python analyze_h2emu.py [DIR] [--ref REF] [--trials 500]

DIR (default: this script's folder) holds circuits_t<t>.json and counts_t<t>.npz; REF (default DIR/../
budget_sym_rdm1npj1.json) is the kappa = 1 noise-model budget for the printed comparison. Writes DIR/h2emu_data.dat
(one row per available time; columns in the header).
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]     # rdm1.py, budget.py: same folder or qhw/
from rdm1 import settings, design
from budget import estimate, symmetrize

GROUPS = ("noisy", "mitigated")


def errors(C, meta, A, m, mitigate, rho_ex, d_ex):
    blocks, sectors = meta["blocks"], tuple(meta["sectors"])
    nb = (meta["n_qubits"] - 2)//2
    rho, d = estimate(C, np.array(meta["conf"]), np.array(meta["f_exponent"]), blocks, sectors, nb, A, m, mitigate)
    rho = symmetrize(rho)
    return d, float(np.abs(rho[0] - rho_ex[0]).max()), d - d_ex


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default=HERE)
    ap.add_argument("--ref")
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    D = a.dir
    fr = a.ref or os.path.join(D, "..", "budget_sym_rdm1npj1.json")
    ref = {round(r["t"], 2): r for r in json.load(open(fr))["rows"]} if os.path.exists(fr) else {}
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "counts_t?.??.npz"))):     # not counts_t1.00_pilot.npz
        t = f.split("_t")[-1][:-4]
        meta = json.load(open(os.path.join(D, f"circuits_t{t}.json")))["meta"]
        C = np.load(f)["counts"]
        m = (meta["n_qubits"] - 2)//2 + 1
        A = design(settings(m))
        rho_ex = [np.array([[x[0] + 1j*x[1] for x in row] for row in r]) for r in meta["rho_avqds"]]
        d_ex = meta["d_avqds"]
        sh = C.reshape(len(C), -1).sum(1)
        P = C.reshape(len(C), -1)/sh[:, None]
        nb = (meta["n_qubits"] - 2)//2
        idx = np.indices(C[0].shape)
        keep = (idx[:nb + 1].sum(0) == meta["sectors"][0]) & (idx[nb + 1:].sum(0) == meta["sectors"][1])
        acc = [float(C[k][keep].sum()/sh[k]) for k in range(len(C))]
        row = [float(t), d_ex]
        print(f"t = {float(t):.2f}: shots {int(sh.sum())}, acceptance {min(acc[:-1]):.3f}-{max(acc[:-1]):.3f} "
              f"(mirror {acc[-1]:.3f}), d_AVQDS = {d_ex:.4f}")
        for g in GROUPS:
            mit = g == "mitigated"
            d0, r0, e0 = errors(C, meta, A, m, mit, rho_ex, d_ex)
            bs = np.array([errors(np.array([rng.multinomial(int(s), p) for s, p in zip(sh, P)], float).reshape(C.shape),
                                  meta, A, m, mit, rho_ex, d_ex) for _ in range(a.trials)])
            for v, col in ((d0, bs[:, 0]), (r0, bs[:, 1]), (abs(e0), np.abs(bs[:, 2]))):
                row += [v, np.percentile(col, 5), np.percentile(col, 95)]
            nm = ref.get(round(float(t), 2), {}).get("shots", {}).get("100000", {}).get(g)
            cmp = (f"   noise model kappa = 1 median: rho {np.median(nm['rho']):.4f}, |dd| {np.median(np.abs(nm['d'])):.4f}"
                   if nm else "")
            print(f"  {g:>9}: d = {d0:.4f} ({d0 - d_ex:+.4f}), max rho error {r0:.4f} "
                  f"[{np.percentile(bs[:, 1], 5):.4f}, {np.percentile(bs[:, 1], 95):.4f}]{cmp}")
        rows.append(row)
    hdr = "t d_AVQDS " + " ".join(f"{g}_{k}_{p}" for g in GROUPS for k in ("d", "rho_err", "d_err")
                                    for p in ("val", "p5", "p95"))
    np.savetxt(os.path.join(D, "h2emu_data.dat"), np.array(rows), fmt="%.6g",
               header=f"H2-Emulator (Nexus), 1e5 shots per time, symmetrized rho; val = measured counts, p5/p95 = "
                      f"bootstrap over {a.trials} redraws\n" + hdr)


if __name__ == "__main__":
    main()
