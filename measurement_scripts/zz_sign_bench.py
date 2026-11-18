"""Micro-benchmark of the H2-Emulator ZZ error versus the sign of the angle (env ~/anaconda3/envs/qnexus).

Two qubits in |+>|+>, N = 40 ZZPhase(a) gates (a in half-turns, N a = 0 mod 2, so the sequence is the identity up to a
global phase), then back to the Z basis; the probability of not returning to 00 measures the accumulated ZZ error.
Angles: +-0.25, 1.75 (= -0.25 mod 2), +-0.5, 1.5 (= -0.5 mod 2), plus N = 0 as the SPAM reference. If -a behaves like
2 - a, the emulator maps negative angles to large ones.

    python zz_sign_bench.py [--shots 4000]          writes zz_sign_bench.json
"""
import argparse, json, os
import numpy as np
import qnexus as qnx
from pytket import Circuit, OpType

HERE = os.path.dirname(os.path.abspath(__file__))
ANGLES = [0.25, -0.25, 1.75, 0.5, -0.5, 1.5]
N = 40


def bench(a, n):
    c = Circuit(2, 2)
    for q in (0, 1):
        c.add_gate(OpType.PhasedX, [0.5, 0.5], [q])          # Ry(pi/2): |0> -> |+>
    for _ in range(n):
        c.add_gate(OpType.ZZPhase, [a], [0, 1])
    for q in (0, 1):
        c.add_gate(OpType.PhasedX, [-0.5, 0.5], [q])
    c.measure_all()
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=4000)
    a = ap.parse_args()
    project = qnx.projects.get_or_create(name="tgqce-h2emu")
    qnx.context.set_active_project(project)
    config = qnx.QuantinuumConfig(device_name="H2-Emulator")
    cases = [("ref", 0.0, 0)] + [(f"a{x:+g}", x, N) for x in ANGLES]
    refs = [qnx.circuits.upload(circuit=bench(x, n), name=f"zzsign_{k}", project=project) for k, x, n in cases]
    comp = qnx.compile(programs=refs, backend_config=config, name="zzsign_compile", optimisation_level=0, project=project)
    for (k, x, n), r in zip(cases, comp):
        tk = r.download_circuit()
        zz = [float(g.op.params[0]) for g in tk if g.op.type == OpType.ZZPhase]
        print(f"{k}: compiled ZZ count {len(zz)}, angles {sorted(set(round(v, 6) for v in zz))}", flush=True)
    job = qnx.start_execute_job(programs=list(comp), n_shots=[a.shots]*len(comp), backend_config=config,
                                name="tgqce_zz_sign_bench", project=project)
    print("job", job.id, flush=True)
    qnx.jobs.wait_for(job, timeout=None)
    out = []
    for (k, x, n), r in zip(cases, qnx.jobs.results(job)):
        cnt = r.download_result().get_counts()
        tot = sum(cnt.values()); fail = 1 - cnt.get((0, 0), 0)/tot
        err = np.sqrt(fail*(1 - fail)/tot)
        out.append(dict(case=k, angle=x, n_zz=n, shots=tot, p_fail=fail, p_fail_err=err))
        print(f"{k:>7}: angle {x:+.2f}, N = {n}, P(not 00) = {fail:.4f} +- {err:.4f}", flush=True)
    json.dump(dict(job=str(job.id), rows=out), open(os.path.join(HERE, "zz_sign_bench.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
