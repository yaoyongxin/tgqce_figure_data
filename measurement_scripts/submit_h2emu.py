"""Run the circuits of build_h2emu.py on the Quantinuum H2-Emulator through Nexus (env ~/anaconda3/envs/qnexus).

    python submit_h2emu.py cost  --t 1.00   upload, compile on Nexus (optimisation level 0; two-qubit counts and outcome
                                            distributions checked against the uploaded circuits),
                                            write nexus_t<t>.json (ids); with --hqc also the HQC cost estimate (needs access
                                            to the H2-1SC syntax checker, which this account lacks; the Nexus-hosted
                                            H2-Emulator reported no HQC cost, only CPU time, in a 100-shot probe)
    python submit_h2emu.py run   --t 1.00   start the execute job (circuits above 10,000 shots split into chunks)
    python submit_h2emu.py fetch --t 1.00   wait for the job and save the counts to counts_t<t>.npz
                                            (array [n_circuits] + [2]*n, axis 1 + q = qubit q)

Project on Nexus: tgqce-h2emu. Device: H2-Emulator (Nexus hosted, its default H2 noise model).
"""
import argparse, json, os
import numpy as np
import qnexus as qnx
from pytket import Circuit, OpType

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "H2-Emulator"
PROJECT = "tgqce-h2emu"
MAX_SHOTS = 10000                       # per circuit on H2-Emulator (MaxShotsExceeded above this)


def chunks(shots):
    """Split each circuit's shots into near-equal chunks <= MAX_SHOTS: list of (circuit index, shots)."""
    out = []
    for k, s in enumerate(shots):
        n = -(-s//MAX_SHOTS)
        out += [(k, s//n + (1 if j < s % n else 0)) for j in range(n)]
    return out


def probs(tk):
    """Outcome probabilities of a pytket circuit in Rz/PhasedX/ZZPhase/ZZMax (measurements ignored), [2]*n with axis q =
    qubit q, by direct numpy simulation (pytket's own simulator refuses 12 qubits)."""
    n = tk.n_qubits
    psi = np.zeros([2]*n, complex); psi[(0,)*n] = 1
    X = np.array([[0, 1], [1, 0]])
    rz = lambda a: np.diag([np.exp(-1j*np.pi*a/2), np.exp(1j*np.pi*a/2)])
    rx = lambda a: np.cos(np.pi*a/2)*np.eye(2) - 1j*np.sin(np.pi*a/2)*X
    for c in tk:
        ty = c.op.type
        if ty in (OpType.Measure, OpType.Barrier, OpType.Phase):
            continue
        qs = [tk.qubits.index(x) for x in c.qubits]; ps = [float(x) for x in c.op.params]
        if ty in (OpType.ZZPhase, OpType.ZZMax):
            a = ps[0] if ty == OpType.ZZPhase else 0.5
            d = np.exp(-1j*np.pi*a/2*np.array([[1, -1], [-1, 1]]))
            psi = np.moveaxis(np.moveaxis(psi, qs, [0, 1])*d[:, :, None].reshape([2, 2] + [1]*(n - 2)), [0, 1], qs)
            continue
        U = {OpType.Rz: lambda: rz(ps[0]), OpType.PhasedX: lambda: rz(ps[1]) @ rx(ps[0]) @ rz(-ps[1])}[ty]()
        psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [qs[0]])), 0, qs[0])
    return np.abs(psi)**2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["cost", "run", "fetch"])
    ap.add_argument("--t", required=True, help="time as in the file names, e.g. 1.00")
    ap.add_argument("--hqc", action="store_true", help="cost stage: also run the HQC estimate (needs H2-1SC access)")
    a = ap.parse_args()
    cj = json.load(open(os.path.join(HERE, f"circuits_t{a.t}.json")))
    fn = os.path.join(HERE, f"nexus_t{a.t}.json")
    project = qnx.projects.get_or_create(name=PROJECT)
    qnx.context.set_active_project(project)
    config = qnx.QuantinuumConfig(device_name=DEVICE)
    shots = [c["shots"] for c in cj["circuits"]]

    if a.stage == "cost":
        refs = [qnx.circuits.upload(circuit=Circuit.from_dict(c["circuit"]), name=f"t{a.t}_{c['kind']}", project=project)
                for c in cj["circuits"]]
        comp = qnx.compile(programs=refs, backend_config=config, name=f"compile_t{a.t}", optimisation_level=0,
                           project=project)
        for c, r in zip(cj["circuits"], comp):
            tk = r.download_circuit()
            n2 = tk.n_gates_of_type(OpType.ZZPhase) + tk.n_gates_of_type(OpType.ZZMax) + tk.n_gates_of_type(OpType.TK2)
            dp = float(np.abs(probs(tk) - probs(Circuit.from_dict(c["circuit"]))).max())
            print(f"{c['kind']:>9}: n2q local {c['n2q']}, after Nexus compile {n2}, max |dp| {dp:.1e}, shots {c['shots']}",
                  flush=True)
            assert n2 == c["n2q"] and dp < 1e-8, (c["kind"], n2, c["n2q"], dp)
        json.dump(dict(t=a.t, device=DEVICE, project=PROJECT, compiled_ids=[str(r.id) for r in comp], shots=shots),
                  open(fn, "w"), indent=1)
        if a.hqc:
            per = [qnx.circuits.cost(circuit_ref=r, n_shots=s, backend_config=config, syntax_checker="H2-1SC",
                                     project=project) for r, s in zip(comp, shots)]
            print("cost per circuit (HQC):", [round(x, 1) for x in per], "total", round(sum(per), 1), flush=True)
            json.dump(dict(t=a.t, device=DEVICE, project=PROJECT, compiled_ids=[str(r.id) for r in comp],
                           shots=shots, cost=per, total_cost=sum(per)), open(fn, "w"), indent=1)
    elif a.stage == "run":
        nj = json.load(open(fn))
        comp = [qnx.circuits.get(id=i) for i in nj["compiled_ids"]]
        ch = chunks(nj["shots"])
        job = qnx.start_execute_job(programs=[comp[k] for k, _ in ch], n_shots=[s for _, s in ch],
                                    backend_config=config, name=f"tgqce_B5U1.5_t{a.t}", project=project)
        nj["job_id"] = str(job.id); nj["chunks"] = ch
        json.dump(nj, open(fn, "w"), indent=1)
        print("started job", job.id, flush=True)
    else:
        nj = json.load(open(fn))
        job = qnx.jobs.get(id=nj["job_id"])
        qnx.jobs.wait_for(job, timeout=None)
        n = cj["meta"]["n_qubits"]
        res = qnx.jobs.results(job)
        ch = nj["chunks"]
        assert len(res) == len(ch)
        C = np.zeros([len(nj["shots"])] + [2]*n)
        for (k, _), r in zip(ch, res):
            br = r.download_result()
            for bits, cnt in br.get_counts().items():         # bits in the order of the circuit's bits c[0..n-1]
                C[(k,) + tuple(bits)] += cnt
        np.savez_compressed(os.path.join(HERE, f"counts_t{a.t}.npz"), counts=C)
        print("saved counts, shots per circuit:", C.reshape(len(C), -1).sum(1).astype(int).tolist(), flush=True)


if __name__ == "__main__":
    main()
