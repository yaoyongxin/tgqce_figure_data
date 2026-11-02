"""Summary of aqc_compress.py results: CNOTs needed for |d - d_AVQDS| below given tolerances, per time.

    python summarize_aqc.py DIR [--prefix aqc|mps] [--tols 5e-3 1e-3 1e-4] [--fmax 1e-2] [--png OUT.png]

--prefix mps reads mps_compress.py results and uses the AQC-refined circuits (fidelity_ref, dd_ref,
ncx_ref) when present.

A depth counts for a tolerance only if |dd| < tol AND 1-F <= fmax: d alone can match by accident
(e.g. an almost orthogonal state with the right d).
"""
import argparse, glob, json, os
import numpy as np

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#0b0b0b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tols", type=float, nargs="+", default=[5e-3, 1e-3, 1e-4])
    ap.add_argument("--fmax", type=float, default=1e-2)
    ap.add_argument("--prefix", default="aqc")
    ap.add_argument("--png")
    a = ap.parse_args()
    runs = sorted((json.load(open(f)) for f in glob.glob(os.path.join(a.dir, f"{a.prefix}_t*.json"))), key=lambda r: r["t"])
    for r in runs:                       # use the refined circuits of mps_compress.py when present
        for x in r["rows"]:
            if "fidelity_ref" in x:
                x["fidelity"], x["dd"], x["ncx"] = x["fidelity_ref"], x["dd_ref"], x["ncx_ref"]
    hdr = "   t     d_AVQDS  N_cx(AVQDS) | " + " | ".join(f"|dd|<{tol:g}: L  CX  1-F " for tol in a.tols)
    print(hdr)
    table = []
    for r in runs:
        cells, row = [], dict(t=r["t"], ncx_avqds=r["ncx_avqds"])
        for tol in a.tols:
            ok = [x for x in r["rows"] if x["dd"] < tol and 1 - x["fidelity"] <= a.fmax]
            if ok:
                x = ok[0]
                cells.append(f"{x['layers']:2d} {x['ncx']:4d} {1-x['fidelity']:.1e}")
                row[tol] = x["ncx"]
            else:
                last = r["rows"][-1]
                cells.append(f">{last['layers']:2d}  ---  (|dd| {last['dd']:.0e})")
                row[tol] = np.nan
        print(f"{r['t']:5.2f}  {r['d_avqds']:.5f}  {r['ncx_avqds']:6d}     | " + " | ".join(cells))
        table.append(row)
    if a.png and table:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.array([r["t"] for r in table])
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        ax.plot(t, [r["ncx_avqds"] for r in table], "o-", color=INK, lw=1.4)
        ax.text(t[-1], table[-1]["ncx_avqds"], "  AVQDS circuit", color=INK, va="center")
        for tol, c in zip(a.tols, [BLUE, ORANGE, AQUA, MAGENTA]):
            y = np.array([r[tol] for r in table], float)
            ax.plot(t, y, "s-", color=c, lw=1.2, ms=4)
            k = np.where(np.isfinite(y))[0]
            if len(k):
                ax.text(t[k[-1]], y[k[-1]], f"  {a.prefix.upper()}, |Δd| < {tol:g}", color=c, va="center")
        ax.set_xlabel("t"); ax.set_ylabel("CNOT count")
        ax.set_yscale("log")
        ax.set_xlim(t[0] - 0.2, t[-1] + 2.2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(a.png, dpi=140)
        print(f"wrote {a.png}")


if __name__ == "__main__":
    main()
