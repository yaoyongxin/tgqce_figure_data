"""Rebuild hw_data_lambda1.dat and hw_data_lambda1over3.dat (the data plotted in Fig. 4) from the raw results in raw/.

    python build_hw_data.py [--shots 100000] [--outdir .]

Inputs, for each noise strength (inp = rdm1npj1 for lambda = 1, rdm1npj033 for lambda = 1/3):
- raw/budget_sym_<inp>.json: finite-shot statistics (measurement_scripts/budget.py --sym), per time and shot number
  the estimates of d and the errors of rho and d for the 500 repetitions of the three cases;
- raw/<inp>_t<t>.json: noise-model results per time (measurement_scripts/rdm1.py), used here for the circuit choice;
- raw/swp/swp_t<t>.json: the compressed brickwork circuits (measurement_scripts/sweep_compress.py), for N_cx;
- ../fig3_B357_avqds/avqds_data/B5U1.5.dat: N_cx of the AVQDS circuit.
"""
import argparse, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
GROUPS = ("shot", "noisy", "mitigated")
SETS = (("rdm1npj1", "hw_data_lambda1.dat"), ("rdm1npj033", "hw_data_lambda1over3.dat"))


def build(inp, out, n_shots):
    bud = json.load(open(RAW / f"budget_sym_{inp}.json"))
    fd = np.loadtxt(HERE.parent / "fig3_B357_avqds" / "avqds_data" / "B5U1.5.dat")
    rows = []
    for r in bud["rows"]:
        q = json.load(open(RAW / f"{inp}_t{r['t']:.2f}.json"))["platforms"]["quantinuum"]
        sw = json.load(open(RAW / "swp" / f"swp_t{r['t']:.2f}.json"))
        ncx = [x for x in sw["rows"] if x["layers"] == q["layers"]][0]["ncx"]
        i = np.argmin(abs(fd[:, 0] - r["t"]))
        row = [r["t"], fd[i, 2], ncx, q["n2q_U"], r["d_avqds"]]
        for g in GROUPS:
            s = r["shots"][str(n_shots)][g]
            d = r["d_avqds"] + np.array(s["d"])
            for x in (d, np.array(s["rho"]), np.abs(s["d"])):
                row += [np.median(x), np.percentile(x, 5), np.percentile(x, 95)]
        rows.append(row)
    hdr = ("t Ncx_AVQDS Ncx_compressed NZZ_compressed d_AVQDS " +
           " ".join(f"{g}_{k}_{p}" for g in GROUPS for k in ("d", "rho_err", "d_err") for p in ("med", "p5", "p95")))
    np.savetxt(out, np.array(rows), fmt="%.6g",
               header=f"N = {n_shots} shots per time (11 settings + mirror, identity setting x3), {bud['trials']} trials, "
                      f"symmetrized rho\n" + hdr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=100000, help="shots per time: 100000, 300000 or 1000000")
    ap.add_argument("--outdir", type=Path, default=HERE)
    a = ap.parse_args()
    for inp, name in SETS:
        build(inp, a.outdir / name, a.shots)
