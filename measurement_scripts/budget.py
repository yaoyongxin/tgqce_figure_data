"""Error budget of d and the 1PDM at finite shots for the orbital-setting scheme (rdm1.py), three groups:

- shot:      noiseless compressed circuits, finite shots (compression + shot noise)
- noisy:     Quantinuum noise model, finite shots, post-selection on (N_up, N_dn)
- mitigated: as noisy, plus the mirror circuit (same shots as one setting) for the depolarizing rescaling of rdm1.py

N total shots are split over the 2B+1 settings and the mirror circuit (N/12 each for B = 5; --w-id gives the identity
setting w_id shares, for d), the same split in
all groups. d = 1/2 - P(q_B = 1, q_B+1 = 1) from the post-selected identity setting. Shots are drawn from the exact
outcome distributions saved by rdm1.py (DIR/<inp>_probs_quantinuum_t<t>.npz). Errors against the AVQDS state:
max_ab |rho_ab - rho_AVQDS_ab| over both blocks and d - d_AVQDS, per trial; "inf" = infinite shots.

--sym: use the exact symmetries of the paramagnetic half-filled embedding problem (B = 5 mode order of a block: bath
0..4, impurity last; the dn block has the impurity first): spin, rho_dn (impurity moved last) = rho_up, and particle-
hole, rho = S (1 - Q rho^T Q) S with Q swapping the bath pairs (0,1), (2,3) and S = -1 on the impurity. The estimate
is the average of the four equivalent estimates, compared with the AVQDS rho_up (which obeys both to <= 2e-3).

    python budget.py DIR [--inp rdm1mit] [--shots 10000 100000] [--trials 500] [--sym] [--png OUT.png]

writes DIR/budget[_sym][_<inp>].json (no suffix for the default --inp rdm1mit).
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdm1 import settings, design, fit_rho, occupations

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#0b0b0b"
GROUPS = [("shot", "shot noise only", AQUA), ("noisy", "noise model", ORANGE), ("mitigated", "noise model, mitigated", BLUE)]


SPIN = (1, 2, 3, 4, 5, 0)                 # dn-block modes in up-block order
PH = (1, 0, 3, 2, 4, 5)
S_PH = np.diag([1, 1, 1, 1, 1, -1])


def symmetrize(rho):
    a = (rho[0] + rho[1][np.ix_(SPIN, SPIN)])/2
    b = S_PH @ (np.eye(6) - a.T[np.ix_(PH, PH)]) @ S_PH
    return [(a + b)/2]


def estimate(P, conf, fexp, blocks, sectors, nb, A, m, mitigate):
    """rho (both blocks) and d from the setting distributions P[:-1] and the mirror distribution P[-1] (counts or
    probabilities)."""
    occ = np.array([occupations(p, blocks, sectors, nb, True)[0] for p in P[:-1]])
    idx = np.indices(P[0].shape)
    keep = (idx[:nb + 1].sum(0) == sectors[0]) & (idx[nb + 1:].sum(0) == sectors[1])
    p0 = P[0]
    p11 = p0[keep & (idx[nb] == 1) & (idx[nb + 1] == 1)].sum()/p0[keep].sum()
    if mitigate:
        om = occupations(P[-1], blocks, sectors, nb, True)[0]
        f_u = np.sqrt([np.clip(1 - 2*np.mean(np.abs(om[b] - conf[b])), 1e-3, 1) for b in blocks])
        f_s = f_u[None, :]**fexp[:, None]                                  # settings x blocks
        for bi, b in enumerate(blocks):
            occ[:, b] = (occ[:, b] - (1 - f_s[:, [bi]])/2)/f_s[:, [bi]]
        F = f_s[0, 0]*f_s[0, 1]
        p11 = (p11 - (1 - F)/4)/F
    return [fit_rho(A, occ[:, b].ravel(), m) for b in blocks], 0.5 - p11


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--inp", default="rdm1mit")
    ap.add_argument("--shots", type=int, nargs="+", default=[10000, 100000])
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--w-id", type=float, default=1.0, help="shot weight of the identity setting (d) relative to the others")
    ap.add_argument("--png")
    ap.add_argument("--plot-only", action="store_true", help="replot from DIR/budget[_sym].json")
    ap.add_argument("--plot-shots", type=int, help="N of the plot (default: the largest)")
    a = ap.parse_args()
    tag = "" if a.inp == "rdm1mit" else "_" + a.inp
    fj = os.path.join(a.dir, ("budget_sym" if a.sym else "budget") + tag + ".json")
    if a.plot_only:
        js = json.load(open(fj))
        plot(js["rows"], a.plot_shots or max(js["shots"]), a.png)
        return
    rng = np.random.default_rng(a.seed)
    out = []
    for f in sorted(glob.glob(os.path.join(a.dir, f"{a.inp}_probs_quantinuum_t*.npz"))):
        t = float(f.split("_t")[-1][:-4])
        js = json.load(open(os.path.join(a.dir, f"{a.inp}_t{t:.2f}.json")))
        z = np.load(f)
        blocks, sectors = js["blocks"], tuple(js["sectors"])
        n = sum(len(b) for b in blocks); nb = (n - 2)//2; m = nb + 1
        A = design(settings(m))
        rho_ex = [np.array([[x[0] + 1j*x[1] for x in row] for row in r]) for r in js["rho_avqds"]]
        q = js["platforms"]["quantinuum"]
        d_ex = q["d_avqds"]
        conf, fexp = z["conf"], z["f_exponent"]
        dist = {"shot": z["noiseless"], "noisy": z["noisy"], "mitigated": z["noisy"]}
        row = dict(t=js["t"], circuit=f"{q['method']}{q['layers']}", n2q_U=q["n2q_U"], n2q_mirror=q["mirror"]["n2q"],
                   n_circuits=len(z["noisy"]), d_avqds=d_ex, inf={}, shots={})

        def errs(rho, d):
            if a.sym:
                rho = symmetrize(rho)
            return float(max(np.abs(r - rho_ex[bi]).max() for bi, r in enumerate(rho))), float(d - d_ex)
        for g, _, _ in GROUPS:
            row["inf"][g] = errs(*estimate(dist[g], conf, fexp, blocks, sectors, nb, A, m, g == "mitigated"))
        for N in a.shots:
            w = np.ones(len(z["noisy"])); w[0] = a.w_id
            ns = (N*w/w.sum()).astype(int)
            row["shots"][str(N)] = {}
            for g, _, _ in GROUPS:
                P = dist[g].reshape(len(dist[g]), -1)
                P = P/P.sum(1, keepdims=True)
                e = []
                for _ in range(a.trials):
                    C = np.array([rng.multinomial(k, p) for k, p in zip(ns, P)], float).reshape(dist[g].shape)
                    e.append(errs(*estimate(C, conf, fexp, blocks, sectors, nb, A, m, g == "mitigated")))
                row["shots"][str(N)][g] = dict(rho=[x[0] for x in e], d=[x[1] for x in e])
            s = row["shots"][str(N)]
            print(f"t = {row['t']:.2f} {row['circuit']} N = {N}: rho median/95% " +
                  "  ".join(f"{g} {np.median(s[g]['rho']):.4f}/{np.percentile(s[g]['rho'], 95):.4f}" for g, _, _ in GROUPS) +
                  "   |dd| 95%: " + "  ".join(f"{g} {np.percentile(np.abs(s[g]['d']), 95):.4f}" for g, _, _ in GROUPS),
                  flush=True)
        print(f"t = {row['t']:.2f} infinite shots: " +
              "  ".join(f"{g} rho {row['inf'][g][0]:.4f} dd {row['inf'][g][1]:+.4f}" for g, _, _ in GROUPS), flush=True)
        out.append(row)
    out.sort(key=lambda r: r["t"])
    json.dump(dict(shots=a.shots, trials=a.trials, sym=a.sym, w_id=a.w_id, rows=out), open(fj, "w"))
    if a.png:
        plot(out, a.plot_shots or max(a.shots), a.png)


def plot(rows, N, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    t = np.array([r["t"] for r in rows])
    for gi, (g, lab, c) in enumerate(GROUPS):
        dx = (gi - 1)*0.09
        for ax, key in ((axs[0], "rho"), (axs[1], "d")):
            v = [np.abs(r["shots"][str(N)][g][key]) for r in rows]
            med = [np.median(x) for x in v]
            lo = [np.percentile(x, 5) for x in v]; hi = [np.percentile(x, 95) for x in v]
            ax.errorbar(t + dx, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="o", color=c, ms=4,
                        lw=1.2, capsize=2, label=lab)
            inf = [abs(r["inf"][g][0 if key == "rho" else 1]) for r in rows]
            ax.plot(t + dx, inf, "_", color=c, ms=9, mew=1.5)
    for ax, yl in zip(axs, ("max |ρ − ρ_AVQDS|", "|d − d_AVQDS|")):
        ax.axhline(0.01, color=INK, ls="--", lw=0.8)
        ax.set_xlabel("t"); ax.set_ylabel(yl)
        ax.set_ylim(0, None)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"N = {N:.0e} shots per time ({rows[0]['n_circuits']} circuits); median and 5-95% of trials, "
                 f"bars: infinite shots", fontsize=9)
    fig.tight_layout(); fig.savefig(png, dpi=140)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
