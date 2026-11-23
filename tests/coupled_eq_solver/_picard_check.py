"""End-to-end check of the new kbm_treatment='picard' path in
src/solver_nondim.py against the existing 'inline' path.

Run from the repo root:
    python tests/coupled_eq_solver/_picard_check.py


This is written by AI
"""
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from escape.solver_nondim import saarelma_connor_nondim

MHD_FP = Path("/mnt/homes_global/jal2351/software/sc_inputs/CAKEgeqdsks/g158091.01935")
KPROF_FP = Path("/mnt/homes_global/jal2351/software/sc_inputs/CAKEpfiles/p158091.01935")

SOLVE_KW = dict(
    x_res=20,
    fe_degree=2,
    initial_guess="tanh",
    ne_inner_bc="dirichlet",
    linear_solver="lu",
    verbose=False,
)


def build_model():
    return saarelma_connor_nondim(
        P_tot_e=5e6,
        alpha_crit=1.0,
        C_KBM=1.0,
        De_chie_etg=1.0,
        nFC_x0=1e16,
        ncx_x0_ratio=1.0,
        mhd_fp=MHD_FP,
        kprof_fp=KPROF_FP,
        verbose=False,
    )


def run_case(label, **kw):
    print(f"\n===== {label} =====", flush=True)
    m = build_model()
    try:
        x, ne, nFC, nCX = m.solve_coupled_nondim(**SOLVE_KW, **kw)
    except Exception:
        traceback.print_exc()
        print(f"[{label}] FAILED")
        return None
    ok_finite = np.all(np.isfinite(ne)) and np.all(np.isfinite(nFC)) and np.all(np.isfinite(nCX))
    ok_pos = np.all(ne > 0)
    print(f"[{label}] finite={ok_finite} ne>0={ok_pos} "
          f"ne range [{ne.min():.3e}, {ne.max():.3e}] m^-3")
    print(f"[{label}] kbm_info: {m.kbm_info}")
    if hasattr(m, "picard_info"):
        pi = m.picard_info
        print(f"[{label}] picard: converged={pi['converged']} "
              f"iterations={pi['iterations']} gate_mode={pi['gate_mode']}")
        for h in pi["history"]:
            print(f"    it {h['iteration']:2d}: dne_rel={h['dne_rel']:.3e} "
                  f"alpha_bar={h['alpha_bar']:.4f} "
                  f"frac_above={h['alpha_frac_above']:.2f} "
                  f"gate={'ON' if h['kbm_gate_on'] else 'OFF'}")
    return dict(model=m, x=x, ne=ne)


results = {}
results["inline"] = run_case("inline (existing default)", kbm_treatment="inline")
results["picard_avg"] = run_case(
    "picard / average alpha", kbm_treatment="picard", picard_gate_mode="average")
results["picard_maj"] = run_case(
    "picard / majority vote", kbm_treatment="picard", picard_gate_mode="majority")
# Override the Dirichlet default in SOLVE_KW for a Neumann check.
print("\n===== picard / average + Neumann BC =====", flush=True)
m_neu = build_model()
try:
    kw_neu = dict(SOLVE_KW)
    kw_neu["ne_inner_bc"] = "neumann"
    kw_neu["kbm_treatment"] = "picard"
    kw_neu["picard_gate_mode"] = "average"
    x, ne, nFC, nCX = m_neu.solve_coupled_nondim(**kw_neu)
    ok_finite = np.all(np.isfinite(ne)) and np.all(np.isfinite(nFC)) and np.all(np.isfinite(nCX))
    ok_pos = np.all(ne > 0)
    print(f"[picard / average + Neumann BC] finite={ok_finite} ne>0={ok_pos} "
          f"ne range [{ne.min():.3e}, {ne.max():.3e}] m^-3")
    print(f"[picard / average + Neumann BC] kbm_info: {m_neu.kbm_info}")
    if hasattr(m_neu, "picard_info"):
        pi = m_neu.picard_info
        print(f"[picard / average + Neumann BC] picard: converged={pi['converged']} "
              f"iterations={pi['iterations']}")
    results["picard_avg_neumann"] = dict(model=m_neu, x=x, ne=ne) if ok_finite and ok_pos else None
except Exception:
    traceback.print_exc()
    print("[picard / average + Neumann BC] FAILED")
    results["picard_avg_neumann"] = None

# Majority-ON case: local alpha is strongly skewed (mean ~1.1 but only
# ~30-40% of DOFs above alpha_crit=1), so drop alpha_crit far enough
# that frac_above > 0.5 and the whole-grid A/B freeze path is exercised.
print("\n===== picard / majority ON (alpha_crit=-1.0) =====", flush=True)
m_maj_on = build_model()
m_maj_on.alpha_crit = -1.0
try:
    x, ne, nFC, nCX = m_maj_on.solve_coupled_nondim(
        **SOLVE_KW, kbm_treatment="picard", picard_gate_mode="majority")
    ok_finite = np.all(np.isfinite(ne)) and np.all(np.isfinite(nFC)) and np.all(np.isfinite(nCX))
    ok_pos = np.all(ne > 0)
    print(f"[picard / majority ON] finite={ok_finite} ne>0={ok_pos} "
          f"ne range [{ne.min():.3e}, {ne.max():.3e}] m^-3")
    print(f"[picard / majority ON] kbm_info: {m_maj_on.kbm_info}")
    if hasattr(m_maj_on, "picard_info"):
        pi = m_maj_on.picard_info
        print(f"[picard / majority ON] picard: converged={pi['converged']} "
              f"iterations={pi['iterations']}")
        for h in pi["history"]:
            print(f"    it {h['iteration']:2d}: dne_rel={h['dne_rel']:.3e} "
                  f"alpha_bar={h['alpha_bar']:.4f} "
                  f"frac_above={h['alpha_frac_above']:.2f} "
                  f"gate={'ON' if h['kbm_gate_on'] else 'OFF'}")
        if not any(h["kbm_gate_on"] for h in pi["history"]):
            print("[picard / majority ON] WARNING: gate never turned ON")
            results["picard_maj_on"] = None
        else:
            results["picard_maj_on"] = (
                dict(model=m_maj_on, x=x, ne=ne) if ok_finite and ok_pos else None
            )
    else:
        results["picard_maj_on"] = dict(model=m_maj_on, x=x, ne=ne) if ok_finite and ok_pos else None
except Exception:
    traceback.print_exc()
    print("[picard / majority ON] FAILED")
    results["picard_maj_on"] = None

# Bad-arg validation checks
print("\n===== validation checks =====")
m = build_model()
for bad_kw, exc in [
    (dict(kbm_treatment="bogus"), ValueError),
    (dict(kbm_treatment="picard", picard_gate_mode="local"), ValueError),
    (dict(kbm_treatment="picard", picard_relax=0.0), ValueError),
    (dict(kbm_treatment="picard", picard_relax=1.5), ValueError),
]:
    try:
        m.solve_coupled_nondim(**SOLVE_KW, **bad_kw)
        print(f"ERROR: no exception for {bad_kw}")
    except exc as e:
        print(f"OK: {bad_kw} -> {exc.__name__}: {e}")
    except Exception as e:
        print(f"UNEXPECTED for {bad_kw}: {type(e).__name__}: {e}")

# Cross-compare converged profiles (Dirichlet cases only)
print("\n===== profile comparison =====")
ref = results.get("inline")
for k, r in results.items():
    if r is None or ref is None or "neumann" in k:
        continue
    d = np.linalg.norm(r["ne"] - ref["ne"]) / np.linalg.norm(ref["ne"])
    print(f"|ne_{k} - ne_inline| / |ne_inline| = {d:.3e}")

n_fail = sum(1 for r in results.values() if r is None)
print(f"\n{'ALL SOLVES OK' if n_fail == 0 else f'{n_fail} SOLVE(S) FAILED'}")
