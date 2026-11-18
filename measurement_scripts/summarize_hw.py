"""Summary of hw_compile.py. Error bound of a circuit: |d_circuit - d_AVQDS| + |d_noisy,ps - d_circuit|
(compression error plus noise bias after post-selection; absolute values, so no credit for cancellations).
Per time and platform the cheapest circuit (fewest two-qubit gates) with bound < --tol is chosen, otherwise
the one with the smallest bound (marked *). Also the shots (and Quantinuum HQC) for a statistical error
sigma: N = P11 (1 - P11) / (sigma^2 acceptance).

    python summarize_hw.py DIR [--tol 0.01] [--sigma 5e-3] [--pmem 2e-4] [--png OUT.png]

--pmem selects the Quantinuum memory-error level (an entry of hw_compile.py --pmem).
"""
import argparse, glob, json, os
import numpy as np

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#0b0b0b"
PLAT = [("quantinuum", "Quantinuum (H2-class model)", "n2q", "ZZ"), ("ibm_fez", "IBM Heron (FakeFez)", "ncz", "CZ")]


def bound(r, p):
    return abs(r["dd"]) + abs(r[p]["d_noisy_ps"] - r["d"])


def choose(rows, p, key, tol, drho=None):
    """Cheapest circuit with bound < tol, else the smallest bound; with drho, only circuits whose compression
    error in the one-particle density matrix (rows[*]["drho"], rho_compression.py) is <= drho, if any."""
    if drho is not None and any(x.get("drho", np.inf) <= drho for x in rows):
        rows = [x for x in rows if x.get("drho", np.inf) <= drho]
    ok = [x for x in rows if bound(x, p) < tol]
    if ok:
        return min(ok, key=lambda x: (x[p][key], bound(x, p))), True
    return min(rows, key=lambda x: bound(x, p)), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--sigma", type=float, default=5e-3)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--pmem", default="2e-4")
    ap.add_argument("--png")
    a = ap.parse_args()
    runs = sorted((json.load(open(f)) for f in glob.glob(os.path.join(a.dir, "hw_t*.json"))), key=lambda r: r["t"])
    for r in runs:                      # AVQDS circuit size from the compression results
        f = os.path.join(a.dir, f"mps_t{r['t']:.2f}.json")
        r["ncx_avqds"] = json.load(open(f))["ncx_avqds"] if os.path.exists(f) else np.nan
        for x in r["rows"]:             # Quantinuum: use the selected memory-error level
            if "memory" in x.get("quantinuum", {}):
                x["quantinuum"].update(x["quantinuum"]["memory"][f"{float(a.pmem):g}"])
    best = {p: [] for p, *_ in PLAT}
    for p, name, key, g in PLAT:
        print(f"\n{name}" + (f", memory error {a.pmem}" if p == "quantinuum" else "") +
              f": cheapest circuit with bound < {a.tol:g} (* none: smallest bound), shots for sigma = {a.sigma:g}")
        print("    t   d_AVQDS | circuit  1-F      dd   | 2q  2q-depth | raw err  ps err  acc  | bound  | shots" +
              ("   HQC" if p == "quantinuum" else ""))
        for r in runs:
            x, ok = choose(r["rows"], p, key, a.tol)
            q = x[p]
            shots = q["p11_ps"]*(1 - q["p11_ps"])/(a.sigma**2*q["acceptance"])
            hqc = 5 + shots/5000*(q["n1q"] + 10*q["n2q"] + 5*r["n_qubits"]) if p == "quantinuum" else None
            best[p].append(dict(t=r["t"], d=r["d_avqds"], x=x, shots=shots, hqc=hqc))
            print(f"  {r['t']:4.2f}  {r['d_avqds']:.4f} | {x['method']} {x['layers']:2d}  {1-x['fidelity']:.1e} {x['dd']:+.1e} | "
                  f"{q[key]:3d}  {q['depth2q']:3d}      | {q['d_noisy']-r['d_avqds']:+.4f} {q['d_noisy_ps']-r['d_avqds']:+.4f} "
                  f"{q['acceptance']:.2f} | {bound(x, p):.4f}{' ' if ok else '*'}| {shots:7.0f}" + (f" {hqc:6.0f}" if hqc else ""))
        if p == "quantinuum":
            print(f"  total HQC (one Z-basis setting per time): {sum(b['hqc'] for b in best[p]):.0f}")
    if a.png and runs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.array([r["t"] for r in runs]); d = np.array([r["d_avqds"] for r in runs])
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
        ax.plot(t, d, "-", color=INK, lw=1.4)
        ax.text(t[-1], d[-1], "  AVQDS", color=INK, va="center")
        ax.fill_between(t, d - a.tol, d + a.tol, color=INK, alpha=0.08, lw=0)
        ax.text(t[-1], d[-1] - a.tol, f"  ±{a.tol:g}", color=INK, fontsize=8, va="center")
        ax.set_title(f"Quantinuum memory error {a.pmem} per qubit per 2q layer", fontsize=9, color=INK)
        for (p, name, key, g), c, off, lab in zip(PLAT, [BLUE, ORANGE], [0.03, -0.03], [1.12, 0.89]):
            raw = [b["x"][p]["d_noisy"] for b in best[p]]; ps = [b["x"][p]["d_noisy_ps"] for b in best[p]]
            ax.plot(t + off, raw, "o", mfc="none", color=c, ms=5)
            ax.plot(t + off, ps, "o", color=c, ms=5)
            ax.text(t[-1] + off, ps[-1], f"  {name.split(' (')[0]}", color=c, va="center")
            ax2.plot(t, [b["x"][p][key] for b in best[p]], "s-", color=c, ms=4, lw=1.2)
            ax2.text(t[-1], best[p][-1]["x"][p][key]*lab, f"  {name.split(' (')[0]} {g}", color=c, va="center")
        ax.text(0.02, 0.04, "open: raw, filled: post-selected (N_up, N_dn)", transform=ax.transAxes, color=INK, fontsize=8)
        ax.set_xlabel("t"); ax.set_ylabel("d (noise-model simulation)")
        ax.set_xlim(t[0] - 0.3, t[-1] + 1.8)
        ax2.plot(t, [r["ncx_avqds"] for r in runs], "o-", color=INK, lw=1.2)
        ax2.text(t[-1], runs[-1]["ncx_avqds"], "  AVQDS CNOT", color=INK, va="center")
        ax2.set_yscale("log")
        ax2.set_xlabel("t"); ax2.set_ylabel("two-qubit gates (chosen circuit)")
        ax2.set_xlim(t[0] - 0.3, t[-1] + 2.2)
        for x in (ax, ax2):
            x.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(a.png, dpi=140)
        print(f"wrote {a.png}")


if __name__ == "__main__":
    main()
