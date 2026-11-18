"""Build the measurement circuits of the paper protocol (Sec. VI, Fig. 4) for the Quantinuum H2-Emulator.

For each time: the compressed swp circuit pinned in rdm1npj1_t<t>.json, followed by each of the 11 orbital-rotation
settings of rdm1.py, plus the mirror circuit, all qubits measured (bit i = qubit i).

Compilation: qiskit transpile to RZZ + U at optimization level 3 (exact two-qubit unitary synthesis), then pytket
maps RZZ -> ZZPhase and U -> PhasedX/Rz. The pytket route of the paper's noise-model runs
(hw_compile.quantinuum_compile) is not always exact: at t = 2 FullPeepholeOptimise and SquashRzPhasedX lose 1-F up to
1e-2 on some circuits (2.6e-3 in the fitted rho; exact to < 1e-9 at the other times), and KAKDecomposition alone loses
2e-2 on all of them, in pytket 2.18.5. The qiskit route is exact (1-F ~ 1e-10) and uses 0-14 fewer two-qubit gates. All Rz gates are then commuted
to the end and dropped (push_rz), as the emulator otherwise schedules them as separate operations.

Checks: noiseless outcome distributions equal those of the uncompiled circuits saved by rdm1.py (< 1e-4 in probabilities and in rho, against a shot noise of ~5e-3; actual 1e-8 to 1e-5); two-qubit
gate counts within 10% of those of rdm1npj1_t<t>.json. The mirror-rescaling exponents n2q(setting)/n2q(U) are
recomputed from the new counts.

    ~/pyvenvs/qhw/bin/python build_h2emu.py --times 1.0 [--shots 100000] [--w-id 3]

writes circuits_t<t>.json (pytket circuit dicts, shots per circuit, metadata) next to this script.
"""
import argparse, json, os, sys
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(HERE)                                  # qhw/B5U1.5
sys.path.insert(0, os.path.dirname(DIR))                     # qhw
from qiskit.quantum_info import Statevector
from aqc_compress import to_qiskit_order
from hw_compile import H5, load_circuits, quantinuum_tk as compile_tk
from rdm1 import settings, rotation_circuit, design, fit_rho, occupations


def probs_tk(tk, n):
    """Outcome probabilities of a measurement-free pytket circuit, array [2]*n with axis q = qubit q."""
    from pytket.extensions.qiskit import tk_to_qiskit
    p = np.abs(Statevector(tk_to_qiskit(tk)).data)**2
    return p.reshape([2]*n).transpose(list(range(n - 1, -1, -1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--case", default="B5U1.5")
    ap.add_argument("--pin", default="rdm1npj1")
    ap.add_argument("--shots", type=int, default=100000, help="total shots per time")
    ap.add_argument("--w-id", type=float, default=3.0, help="shot weight of the identity setting")
    a = ap.parse_args()
    from pytket import OpType
    with h5py.File(H5.format(case=a.case), "r") as f:
        t_all = f["/0/t"][()]; st = f["/0/state"]
        n = st.shape[1].bit_length() - 1; nb = (n - 2)//2; m = nb + 1
        blocks = [list(range(nb + 1)), list(range(nb + 1, n))]
        ks = settings(m)
        for tt in a.times:
            i = int(np.argmin(abs(t_all - tt))); t = float(t_all[i])
            psi_q = to_qiskit_order(st[i]/np.linalg.norm(st[i]), n)
            js = json.load(open(os.path.join(DIR, f"{a.pin}_t{t:.2f}.json")))
            q = js["platforms"]["quantinuum"]
            z = np.load(os.path.join(DIR, f"{a.pin}_probs_quantinuum_t{t:.2f}.npz"))
            qc = {(m_, L_): c for m_, L_, c in load_circuits(DIR, t, psi_q, n, (q["method"],))}[(q["method"], q["layers"])]
            U = qc.copy_empty_like()
            for ins in qc.data:
                if ins.operation.name != "x":
                    U.append(ins)
            mirror = qc.copy(); mirror.barrier(); mirror = mirror.compose(U.inverse())
            circs = [qc.compose(rotation_circuit(k, blocks, n)) for k in ks] + [mirror]
            n2_ref = [s["n2q"] for s in q["settings"]] + [q["mirror"]["n2q"]]
            w = np.ones(len(circs)); w[0] = a.w_id
            shots = [int(x) for x in (a.shots*w/w.sum()).astype(int)]
            out, maxdev, pc = [], 0.0, []
            for c, (circ, n2r) in enumerate(zip(circs, n2_ref)):
                tk = compile_tk(circ)
                n2 = tk.n_gates_of_type(OpType.ZZPhase)
                assert abs(n2 - n2r) <= 0.1*n2r, (t, c, n2, n2r)
                pk = probs_tk(tk, n); pc.append(pk)
                maxdev = max(maxdev, float(np.abs(pk - z["noiseless"][c]).max()))
                tk.measure_all()
                out.append(dict(index=c, kind="mirror" if c == len(circs) - 1 else f"setting{c}", n2q=int(n2),
                                n1q=int(tk.n_gates_of_type(OpType.PhasedX)), depth2q=int(tk.depth_2q()),
                                shots=shots[c], circuit=tk.to_dict()))
            # effect of the compilation on rho (noiseless, post-selected fit of both blocks)
            A = design(ks); sec = tuple(js["sectors"])
            occ = np.array([occupations(p_, blocks, sec, nb, True)[0] for p_ in pc[:-1]])
            occ0 = np.array([occupations(p_, blocks, sec, nb, True)[0] for p_ in z["noiseless"][:-1]])
            drho = max(np.abs(fit_rho(A, occ[:, b].ravel(), m) - fit_rho(A, occ0[:, b].ravel(), m)).max() for b in blocks)
            assert maxdev < 1e-4 and drho < 1e-4, (maxdev, drho)
            n2U = compile_tk(U).n_gates_of_type(OpType.ZZPhase)
            fexp = [c_["n2q"]/n2U for c_ in out[:-1]]
            meta = dict(t=t, step=i, n_qubits=n, blocks=blocks, sectors=js["sectors"], method=q["method"],
                        layers=q["layers"], pin=a.pin, total_shots=int(sum(shots)), w_id=a.w_id,
                        conf=z["conf"].tolist(), f_exponent=fexp, n2q_U=int(n2U), d_avqds=q["d_avqds"],
                        rho_avqds=js["rho_avqds"], bit_order="bit i = qubit i",
                        compile_dev_probs=maxdev, compile_dev_rho=float(drho))
            fn = os.path.join(HERE, f"circuits_t{t:.2f}.json")
            json.dump(dict(meta=meta, circuits=out), open(fn, "w"))
            hqc = sum(5 + c["shots"]*(c["n1q"] + 10*c["n2q"] + 5*n)/5000 for c in out)
            n2s = [c_["n2q"] for c_ in out]
            print(f"t = {t:.2f}: {len(out)} circuits, n2q {min(n2s)}-{max(n2s)} (paper {min(n2_ref)}-{max(n2_ref)}), U {n2U} "
                  f"(paper {q['n2q_U']}), shots {sum(shots)}, "
                  f"compilation: probs {maxdev:.1e}, rho {drho:.1e}, HQC estimate {hqc:,.0f} -> {fn}", flush=True)


if __name__ == "__main__":
    main()
