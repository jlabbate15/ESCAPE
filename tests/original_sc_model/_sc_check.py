"""End-to-end check of the ORIGINAL Saarelma-Connor model solvers in
src/solver_sc.py: the scipy (solve_bvp) and Firedrake (SNES) paths are
run on the same case and cross-compared.

Run from the repo root:
    python tests/original_sc_model/_sc_check.py

Checks
------
1. Both implementations converge and reproduce the boundary conditions
   (Dirichlet n_e(0) = ne_x0, Neumann dn_e/dx(x_inner)).
2. Each satisfies its own conservative form of Eq. (7),

       [f n_e'](x) - [f n_e'](x_inner) = int_{x_inner}^{x} RHS dx',

   which only involves first derivatives of the solution.
3. The frozen Picard kernel E(x) is self-consistent with the converged
   n_e (i.e. the fixed point really is a fixed point).
4. The two implementations agree with each other, and each is grid
   converged.
5. The KBM gate responds to alpha_crit in the expected direction (more
   KBM transport -> lower pedestal-top density).

This is written by AI.
"""
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.solver_sc import saarelma_connor_sc

MHD_FP = Path("/mnt/homes_global/jal2351/software/sc_inputs/CAKEgeqdsks/g158091.01935")
KPROF_FP = Path("/mnt/homes_global/jal2351/software/sc_inputs/CAKEpfiles/p158091.01935")

FREE_PARAMS = dict(alpha_crit=3.0, C_KBM=1.0, De_chie_etg=1.0, nFC_x0=1e16)
SOLVE_KW = dict(initial_guess="pfile", bc_origin="p-file", picard_relax=0.5)


def build(**overrides):
    fp = dict(FREE_PARAMS, **overrides)
    return saarelma_connor_sc(
        P_tot_e=5e6, ncx_x0_ratio=1.0,
        mhd_fp=MHD_FP, kprof_fp=KPROF_FP, verbose=False, **fp,
    )


def conservative_residual(m, interp="pchip"):
    """Max |R| / max |f n_e'| for the converged solution (see module doc).

    ``interp`` selects how the equilibrium coefficients are reconstructed
    between p-file points; use the one the solver itself uses, otherwise
    the metric measures the interpolation difference instead of the
    solution error ("pchip" for the scipy path, "linear" for Firedrake).
    """
    x, ne, dne, E = m.x_sol, m.ne_sol, m.dne_dx_sol, m.E_sol
    if interp == "linear":
        ip = lambda a: np.interp(x, m.x_init, a)
    else:
        ip = lambda a: PchipInterpolator(m.x_init, a)(x)

    Si, Scx = ip(m.S_i_pres), ip(m.S_cx_pres)
    Vcx, fFC, fCX = ip(np.abs(m.V_cx_pres)), ip(m.fFC), ip(m.fCX)
    f = (ip(m.gradr2_fsa * (m.D_NEO + m._D_KBM))
         + ip(m.gradr2_fsa * m.C_ETG) / ne)
    C_cx = 1.0 - (abs(m.V_FC) * fFC / (Vcx * fCX)) * (
        (Si + 0.5 * Scx) / (Si + Scx))
    rhs = -ne * Si * (
        -(f / (Vcx * fCX)) * (dne - m.dne_dx_inner) + C_cx * m.nFC_x0 * E)
    J = f * dne
    R = (J - J[0]) - cumulative_trapezoid(rhs, x, initial=0.0)
    return float(np.max(np.abs(R)) / np.max(np.abs(J)))


def kernel_self_consistency(m):
    """Relative difference between E(x) frozen in the last solve and
    E(x) recomputed from the converged n_e."""
    E_re = m._exp_kernel_sc(m.ne_sol, m.x_sol)
    return float(np.max(np.abs(E_re - m.E_sol)) / np.max(np.abs(E_re)))


def run(label, solver_structure, model=None, **kw):
    m = model if model is not None else build()
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            x, ne, dne = m.solve_sc(solver_structure=solver_structure,
                                    **SOLVE_KW, **kw)
        except Exception:
            traceback.print_exc()
            print(f"[{label}] FAILED")
            return None
    dt = time.time() - t0

    interp = "pchip" if solver_structure == "scipy" else "linear"
    res = conservative_residual(m, interp)
    dE = kernel_self_consistency(m)
    bc_out = abs(ne[-1] / m.ne_x0 - 1.0)
    bc_in = abs(dne[0] / m.dne_dx_inner - 1.0)

    print(f"[{label}] {dt:5.2f}s  picard it = {m.picard_info['iterations']:3d}"
          f"  first_step = {m.first_step_used!r}")
    print(f"[{label}]   n_e(x_inner) = {ne[0]:.5e}  n_e(0) = {ne[-1]:.5e} m^-3"
          f"  (min n_e = {ne.min():.4e}, all finite = "
          f"{bool(np.all(np.isfinite(ne)))})")
    print(f"[{label}]   BC error: Dirichlet {bc_out:.2e}, Neumann {bc_in:.2e}")
    print(f"[{label}]   Eq.(7) conservative residual = {res:.3e}"
          f"   E self-consistency = {dE:.3e}")
    print(f"[{label}]   alpha_bar = {m.alpha_bar_ped:.4f}  KBM "
          f"{'ON' if m.kbm_gate_on else 'OFF'}  (alpha_crit = "
          f"{m.alpha_crit})")
    for w in caught:
        if issubclass(w.category, RuntimeWarning):
            print(f"[{label}]   warning: {str(w.message).splitlines()[0][:110]}")
    ok = (bc_out < 1e-8 and bc_in < 1e-2 and res < 1e-2 and dE < 1e-2
          and np.all(ne > 0))
    print(f"[{label}]   -> {'PASS' if ok else 'CHECK'}")
    return dict(model=m, x=x, ne=ne, ok=ok)


print("=" * 72)
print("1-2. Both implementations on the same case")
print("=" * 72)
results = {}
for impl, kw in (("scipy", dict(x_res=200)),
                 ("firedrake", dict(x_res=200, fe_degree=2))):
    r = run(f"{impl}", impl, **kw)
    if r:
        results[impl] = r

if len(results) == 2:
    xs, nes = results["scipy"]["x"], results["scipy"]["ne"]
    xf, nef = results["firedrake"]["x"], results["firedrake"]["ne"]
    d = np.max(np.abs(np.interp(xf, xs, nes) - nef)) / np.max(np.abs(nef))
    print(f"\n[compare] max |n_e(scipy) - n_e(firedrake)| / max n_e = {d:.3e}"
          f"  -> {'PASS' if d < 1e-2 else 'CHECK'}")

print("\n" + "=" * 72)
print("3. Grid convergence")
print("=" * 72)
prev = None
for impl, kw in (("scipy", dict(x_res=100)), ("scipy", dict(x_res=400)),
                 ("firedrake", dict(x_res=100, fe_degree=2)),
                 ("firedrake", dict(x_res=400, fe_degree=3))):
    m = build()
    try:
        x, ne, _ = m.solve_sc(solver_structure=impl, **SOLVE_KW, **kw)
    except Exception as exc:
        print(f"  {impl} {kw}: FAILED ({exc})")
        continue
    print(f"  {impl:10s} {str(kw):32s} n_e(x_inner) = {ne[0]:.6e}")

print("\n" + "=" * 72)
print("4. KBM gate response to alpha_crit (expect lower pedestal top when on)")
print("=" * 72)
for ac in (0.2, 3.0):
    m = build(alpha_crit=ac)
    try:
        x, ne, _ = m.solve_sc(solver_structure="firedrake", x_res=200,
                              **SOLVE_KW)
    except Exception as exc:
        print(f"  alpha_crit = {ac}: FAILED ({exc})")
        continue
    print(f"  alpha_crit = {ac:4g}: KBM "
          f"{'ON ' if m.kbm_gate_on else 'OFF'}  alpha_bar = "
          f"{m.alpha_bar_ped:.4f}  n_e(x_inner) = {ne[0]:.4e} m^-3")

print("\n" + "=" * 72)
print("5. Error paths")
print("=" * 72)
m = build()
try:
    m.solve_sc(solver_structure="scipy", x_res=200, first_step="eq6", **SOLVE_KW)
    print("  first_step='eq6': solved (Eq. 6 has a solution for this case)")
except RuntimeError as exc:
    print(f"  first_step='eq6' raises as expected:\n    "
          f"{str(exc)[:160]}...")
m = build()
try:
    m.solve_sc(solver_structure="scipy", x_res=200, eq6_form="bogus",
               **SOLVE_KW)
    print("  eq6_form='bogus': NO ERROR (unexpected)")
except ValueError as exc:
    print(f"  eq6_form='bogus' rejected: {str(exc)[:90]}...")
m = build(alpha_crit=None)
try:
    m.solve_sc(solver_structure="scipy", x_res=200, **SOLVE_KW)
    print("  missing free parameter: NO ERROR (unexpected)")
except ValueError as exc:
    print(f"  missing free parameter rejected: {str(exc)[:90]}...")
