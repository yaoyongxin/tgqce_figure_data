"""Summary of rdm1_shots.py and rdm1_fshadow.py: max |rho - rho_AVQDS| (median over trials) vs total shots for the
measurement schemes, the infinite-shot bias, and the Quantinuum cost.

HQC per circuit (job) = 5 + C (N_1q + 10 N_2q + 5 N_meas)/5000 for C shots; a scheme with K distinct circuits
pays the 5 HQC K times (shadows with r shots per random basis: K = N/r). Basis changes of qwc/shadow are counted
as 12 extra one-qubit gates, a random orbital rotation of fshadow as 60 ZZ + 132 one-qubit gates (pytket).

    python summarize_shots.py DIR [--png OUT.png] [--times-plot 1.5 6]
"""
import argparse, glob, json, os
import numpy as np

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#0b0b0b"
METH = [("orbital", "orbital rotations (11)", BLUE), ("qwc", "QWC groups (36)", ORANGE),
        ("shadow_r1", "Pauli shadows", AQUA), ("shadow_r100", "Pauli shadows, 100/basis", YELLOW),
        ("fshadow", "fermionic shadows (N-cons.)", MAGENTA)]


def hqc(meth, N, r1, hw_row, n=12):
    q = hw_row["quantinuum"]
    if meth == "orbital":
        st = r1["platforms"]["quantinuum"]["settings"]
        return sum(5 + N/len(st)*(s["n1q"] + 10*s["n2q"] + 5*n)/5000 for s in st)
    per_shot = (q["n1q"] + 12 + 10*q["n2q"] + 5*n)/5000
    if meth == "qwc":
        return 36*5 + N*per_shot
    if meth.startswith("shadow_r"):
        return N/int(meth[8:])*5 + N*per_shot
    if meth == "fshadow":
        return N*5 + N*(q["n1q"] + 132 + 10*(q["n2q"] + 60) + 5*n)/5000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--png")
    ap.add_argument("--times-plot", type=float, nargs="+", default=[1.5, 6.0])
    a = ap.parse_args()
    runs = []
    for f in sorted(glob.glob(os.path.join(a.dir, "rdm1_shots_t*.json"))):
        r = json.load(open(f))
        tag = f"{r['t']:.2f}"
        ff = os.path.join(a.dir, f"rdm1_fshadow_t{tag}.json")
        if os.path.exists(ff):
            fs = json.load(open(ff))
            for N, v in fs["shots"].items():
                r["shots"].setdefault(N, {}).update(v)
            r["bias"]["fshadow_1e6"] = fs["bias_estimate"]["max_error"]
        r1 = json.load(open(os.path.join(a.dir, f"rdm1_t{tag}.json")))
        hw = json.load(open(os.path.join(a.dir, f"hw_t{tag}.json")))
        meth, L = r["circuit"][:3], int(r["circuit"][3:])
        r["hw_row"] = [x for x in hw["rows"] if x["method"] == meth and x["layers"] == L][0]
        r["r1"] = r1
        runs.append(r)
    runs.sort(key=lambda r: r["t"])
    Ns = sorted({int(N) for r in runs for N in r["shots"]})
    print("median over trials of max_ab |rho - rho_AVQDS| (both blocks); bias = infinite shots "
          "(orbital: post-selected; Pauli: readout-corrected; fshadow: one run of 1e6 shots, incl. its shot noise)")
    for N in Ns:
        print(f"\nN = {N} shots in total" + ("   [HQC at this N]" if N == max(Ns) else ""))
        print("   t   " + "".join(f"{k:>13s}" for k, _, _ in METH))
        for r in runs:
            s = r["shots"].get(str(N), {})
            row = f"  {r['t']:4.1f} " + "".join(f"{np.median(s[k]):13.4f}" if k in s else f"{'-':>13s}" for k, _, _ in METH)
            if N == max(Ns):
                row += "   | HQC " + " ".join(f"{hqc(k, N, r['r1'], r['hw_row']):.0f}" for k, _, _ in METH)
            print(row)
    print("\nbias:   t   orbital(ps)  Pauli(raw)  Pauli(readout-corr.)  fshadow(1e6 shots)")
    for r in runs:
        b = r["bias"]
        print(f"      {r['t']:4.1f}    {b['orbital_ps']:.4f}      {b['pauli_raw']:.4f}      {b['pauli_readout_corrected']:.4f}"
              f"              {b.get('fshadow_1e6', np.nan):.4f}")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sel = [r for r in runs if any(abs(r["t"] - t) < 0.05 for t in a.times_plot)]
        fig, axs = plt.subplots(1, len(sel), figsize=(5.2*len(sel), 4.0), sharey=True)
        for ax, r in zip(np.atleast_1d(axs), sel):
            labels = []
            for k, lab, c in METH:
                N = [n for n in Ns if k in r["shots"].get(str(n), {})]
                if not N:
                    continue
                y = [np.median(r["shots"][str(n)][k]) for n in N]
                lo = [np.percentile(r["shots"][str(n)][k], 25) for n in N]
                hi = [np.percentile(r["shots"][str(n)][k], 75) for n in N]
                ax.fill_between(N, lo, hi, color=c, alpha=0.15, lw=0)
                ax.plot(N, y, "o-", color=c, lw=1.5, ms=4)
                labels.append([y[-1], N[-1], lab, c])
            labels.sort()
            for i in range(1, len(labels)):           # keep direct labels >= 18% apart (log axis)
                labels[i][0] = max(labels[i][0], labels[i-1][0]*1.18)
            for y, n, lab, c in labels:
                ax.text(n*1.15, y, lab, color=c, va="center", fontsize=8)
            for key, c in (("orbital_ps", BLUE), ("pauli_readout_corrected", ORANGE), ("fshadow_1e6", MAGENTA)):
                if key in r["bias"]:
                    ax.axhline(r["bias"][key], color=c, ls=":", lw=1, xmax=0.68)   # stop before the labels
            ax.axhline(0.01, color=INK, ls="--", lw=0.8)
            ax.text(Ns[0], 0.0105, "0.01", color=INK, fontsize=8, va="bottom")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(Ns[0]/1.5, Ns[-1]*12)
            ax.set_xlabel("total shots N")
            ax.set_title(f"t = {r['t']:.1f} ({r['circuit']}, {r['n2q']} ZZ); dotted: infinite-shot bias", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
        np.atleast_1d(axs)[0].set_ylabel("max |ρ − ρ_AVQDS| (median of trials)")
        fig.tight_layout(); fig.savefig(a.png, dpi=140)
        print(f"wrote {a.png}")


if __name__ == "__main__":
    main()
