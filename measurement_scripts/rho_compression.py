"""Compression error of the one-particle density matrix for every candidate circuit (exact statevectors):
max_ab |rho_ab(circuit) - rho_ab(AVQDS)| over both spin blocks, added to DIR/hw_t<t>.json as rows[*]["drho"].

    python rho_compression.py DIR --times ...
"""
import argparse, json, os, sys
import numpy as np
import h5py
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aqc_compress import to_qiskit_order
from hw_compile import H5, load_circuits
from rdm1 import exact_rho


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--case", default="B5U1.5")
    ap.add_argument("--all", action="store_true", help="recompute rows that already have drho")
    a = ap.parse_args()
    with h5py.File(H5.format(case=a.case), "r") as f:
        t_all = f["/0/t"][()]; st = f["/0/state"]
        n = st.shape[1].bit_length() - 1; nb = (n - 2)//2
        blocks = [list(range(nb + 1)), list(range(nb + 1, n))]
        for tt in a.times:
            i = int(np.argmin(abs(t_all - tt)))
            psi_q = to_qiskit_order(st[i]/np.linalg.norm(st[i]), n)
            ex = [exact_rho(psi_q, b, n) for b in blocks]
            fh = os.path.join(a.dir, f"hw_t{t_all[i]:.2f}.json")
            hw = json.load(open(fh))
            circ = {(m, L): c for m, L, c in load_circuits(a.dir, t_all[i], psi_q, n)}
            for r in hw["rows"]:
                if "drho" in r and not a.all:
                    continue
                sv = Statevector(circ[(r["method"], r["layers"])]).data
                r["drho"] = float(max(np.abs(exact_rho(sv, b, n) - ex[k]).max() for k, b in enumerate(blocks)))
            json.dump(hw, open(fh, "w"), indent=1)
            print(f"t = {t_all[i]:.2f}: " + "  ".join(f"{r['method']}{r['layers']} 1-F {1-r['fidelity']:.0e} drho {r['drho']:.4f}"
                                                   for r in hw["rows"]), flush=True)


if __name__ == "__main__":
    main()
