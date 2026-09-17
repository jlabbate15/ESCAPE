"""Checks for the unified solver class in src/solver_api.py.

1. The three modules assemble into one class: clean MRO, no method-name
   collisions, and every method of the base and both mixins resolves on it.
2. The pre-refactor names (saarelma_connor_nondim, saarelma_connor_sc,
   saarelma_connor_base) still import, in either import order, and resolve
   to the assembled class.
3. solve() validates its kwargs: an unknown model name, a kwarg belonging to
   the other model, a kwarg valid only for the other backend, and a typo
   all raise before any solver runs.
4. Both models return the same result-dict keys, and the 1D model's
   reconstructed neutrals are physical (n_FC = nFC_x0 at the separatrix and
   decaying inward, n_CX >= 0) and agree with the solver's own converged
   exponential kernel.

Checks 1-3 need no equilibrium and run in seconds; check 4 builds a model
and solves both models once (a few minutes).  Pass --fast to skip it.

Usage: python tests/solver_api/_solver_api_check.py [--fast]
"""
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MHD_FP = Path("/mnt/homes_global/jal2351/software/sc_inputs/CAKEgeqdsks/g158091.01935")
KPROF_FP = Path("/mnt/homes_global/jal2351/software/sc_inputs/CAKEpfiles/p158091.01935")

FREE_PARAMS = dict(alpha_crit=3.0, C_KBM=1.0, De_chie_etg=1.0, nFC_x0=1e16)


def check_assembly():
    from src.solver import SaarelmaConnorBase
    from src.solver_nondim import NondimSolverMixin
    from src.solver_sc import SCSolverMixin
    from src.solver_api import saarelma_connor

    parts = (SaarelmaConnorBase, NondimSolverMixin, SCSolverMixin)
    methods = [{n for n, v in vars(c).items() if callable(v) and not n.startswith("__")}
               for c in parts]
    base, nondim, sc = methods
    collisions = (base & nondim) | (base & sc) | (nondim & sc)
    assert not collisions, f"method names defined in more than one part: {collisions}"

    for cls, names in zip(parts, methods):
        for name in names:
            owner = next(c for c in saarelma_connor.__mro__ if name in vars(c))
            assert owner is cls, f"{name} resolves to {owner.__name__}, expected {cls.__name__}"

    print(f"  {sum(map(len, methods))} methods across 3 parts, no collisions")


def check_aliases():
    # Each import order runs in a fresh interpreter, since the whole point is
    # which module happens to be imported first.
    orders = {
        "solver_api first": "import src.solver_api as a; from src.solver_nondim import saarelma_connor_nondim as n; from src.solver_sc import saarelma_connor_sc as s",
        "nondim alias first": "from src.solver_nondim import saarelma_connor_nondim as n; from src.solver_sc import saarelma_connor_sc as s; import src.solver_api as a",
        "sc alias first": "from src.solver_sc import saarelma_connor_sc as s; from src.solver_nondim import saarelma_connor_nondim as n; import src.solver_api as a",
    }
    for label, imports in orders.items():
        code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); {imports}; "
                "assert n is a.saarelma_connor and s is a.saarelma_connor; "
                "from src.solver import saarelma_connor_base, SaarelmaConnorBase; "
                "assert saarelma_connor_base is SaarelmaConnorBase; "
                "import src.solver_sc as m\n"
                "try:\n    m.not_a_real_name\nexcept AttributeError:\n    pass\n"
                "else:\n    raise AssertionError('module __getattr__ swallowed an unknown name')\n"
                "print('ALIASES_OK')")
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert "ALIASES_OK" in proc.stdout, f"{label} failed:\n{proc.stderr[-2000:]}"
        print(f"  import order OK: {label}")


def check_kwarg_validation():
    from src.solver_api import saarelma_connor

    # Dispatch and validation happen before any solver touches the
    # equilibrium, so an uninitialised instance is enough here.
    m = saarelma_connor.__new__(saarelma_connor)

    for good, expected in [("1D", "1D"), ("3d", "3D"), ("SC", "1D"), ("coupled", "3D")]:
        assert m._normalise_model(good) == expected

    try:
        m.solve(model="2D")
    except ValueError:
        pass
    else:
        raise AssertionError("model='2D' did not raise")

    cases = [
        ("1D", "firedrake", {"nCX_ic": "solve"}, "belongs to model='3D'"),
        ("3D", "firedrake", {"eq6_form": "complete"}, "belongs to model='1D'"),
        ("1D", "firedrake", {"bvp_tol": 1e-6}, "not with solver_structure='firedrake'"),
        ("1D", "scipy", {"fe_degree": 2}, "not with solver_structure='scipy'"),
        ("3D", "firedrake", {"x_ress": 20}, "not an argument of any solver"),
    ]
    for model, backend, kwargs, fragment in cases:
        try:
            m.solve(model=model, solver_structure=backend, **kwargs)
        except TypeError as err:
            assert fragment in str(err), f"message for {kwargs} lacks {fragment!r}:\n{err}"
        else:
            raise AssertionError(f"solve(model={model!r}, {kwargs}) did not raise")

    try:
        m.solve(model="1D", solver_structure="scipy", implementation="firedrake")
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting solver_structure / implementation did not raise")

    print(f"  {len(cases) + 2} rejection cases raise with the expected message")


def check_solves():
    from src.solver_api import saarelma_connor

    m = saarelma_connor(P_tot_e=5e6, ncx_x0_ratio=1.0, mhd_fp=MHD_FP,
                        kprof_fp=KPROF_FP, verbose=False, **FREE_PARAMS)

    res_1d = m.solve(model="1D", solver_structure="scipy", x_res=200,
                     initial_guess="pfile", bc_origin="p-file", picard_relax=0.5)
    res_3d = m.solve(model="3D", solver_structure="firedrake", x_res=20,
                     initial_guess="tanh", bc_origin="p-file",
                     nCX_ic="scale nFC", nFC_ic="solve", picard_relax=0.5)

    assert set(res_1d) == set(res_3d), f"key sets differ: {set(res_1d) ^ set(res_3d)}"
    assert (res_1d["model"], res_3d["model"]) == ("1D", "3D")
    for res in (res_1d, res_3d):
        n = len(res["x"])
        for key in ("ne", "dne_dx", "nFC", "nCX"):
            assert len(res[key]) == n, f"{res['model']} {key} length {len(res[key])} != {n}"
            assert np.isfinite(res[key]).all(), f"{res['model']} {key} not finite"

    # A 1D solve after a 3D one on the same instance must not pick up the
    # 3D run's nFC_sol / nCX_sol.
    res_1d_again = m.solve(model="1D", solver_structure="scipy", x_res=200,
                           initial_guess="pfile", bc_origin="p-file", picard_relax=0.5)
    assert len(res_1d_again["nFC"]) == len(res_1d_again["x"])

    order = np.argsort(res_1d["x"])
    nFC, nCX = res_1d["nFC"][order], res_1d["nCX"][order]
    E = np.asarray(res_1d["diagnostics"]["E_sol"])[order]
    nFC_x0 = FREE_PARAMS["nFC_x0"]
    assert np.isclose(nFC[-1], nFC_x0, rtol=1e-10), "n_FC at the separatrix != nFC_x0"
    assert np.all(np.diff(nFC) >= 0), "n_FC does not decay monotonically inward"
    assert np.all(nCX >= 0), "n_CX has negative entries"
    kernel_err = np.max(np.abs(nFC - nFC_x0 * E)) / nFC_x0
    assert kernel_err < 1e-3, f"n_FC disagrees with the converged kernel by {kernel_err:.2e}"

    print(f"  1D/3D share keys {sorted(res_1d)}")
    print(f"  1D neutrals physical; |n_FC - nFC_x0 E| / nFC_x0 = {kernel_err:.1e}")


def main():
    checks = [check_assembly, check_aliases, check_kwarg_validation]
    if "--fast" not in sys.argv:
        checks.append(check_solves)

    failed = []
    for check in checks:
        print(f"[{check.__name__}]")
        try:
            check()
        except Exception:
            traceback.print_exc()
            failed.append(check.__name__)

    print("\nFAILED: " + ", ".join(failed) if failed else "\nall checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
