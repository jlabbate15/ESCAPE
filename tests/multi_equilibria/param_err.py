import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
from pathlib import Path
from scipy.interpolate import interp1d
ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(ROOT))
from escape.solver_nondim import saarelma_connor_nondim

# Global parameters
equil_num = 100

# Scan parameters
N = 3 # size of each free parameter array
alpha_crits = np.logspace(-1, 1, N)
C_KBMs = np.logspace(-1, 1, N)
De_chie_etgs = np.logspace(-1, 1, N)
nFC_x0s = np.logspace(14, 17, N)
ncx_x0_ratios = np.logspace(0.1,1.25,N)
psi_val = 0.85 # this is for the scan, not the analysis
x_res = 20

# Static parameters
P_tot_e = 5e6 # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox

# Output data and files
GEQDSK_DIR = Path("/mnt/homes_global/jal2351/software/sc_inputs/CAKEgeqdsks")
PFILE_DIR = Path("/mnt/homes_global/jal2351/software/sc_inputs/CAKEpfiles")
scan_success_dir = Path('scan_results_eqdbOak_equil100_N3_kbmavg/')
verbose = False

# Solver parameters
SOLVE_KW = dict(
    x_res=x_res,
    fe_degree=2,
    initial_guess="tanh",
    ne_inner_bc="neumann",   # Saarelma A7 default; see dirichlet comparison below
    linear_solver="lu",      # or "gamg" for GMRES + algebraic multigrid on J
    nCX_ic="solve",
    kbm_treatment="picard",
    kbm_gate_eps=0.1, # 1e-3 minimum
    picard_gate_mode="average",
    picard_max_it=50,
    picard_rtol=1e-8,
    picard_relax=1.0,
    verbose=verbose,
)

sys.path.insert(0, str(ROOT))

# Add some more files
success_fp = Path('success_PTHmode.txt')
failure_fp = Path('failure_PTHmode.txt')
error_messages_fp = Path('error_messages_PTHmode.txt')

from collections import defaultdict


def initialize_inputs(equil_num, geqdsk_dir=GEQDSK_DIR, pfile_dir=PFILE_DIR):
    """Select equil_num g/p file pairs from the CAKE input directories.

    Files are named g{shot}.{time} and p{shot}.{time}. Selection prioritizes
    one equilibrium per shot number before adding additional times from shots
    that already have a selected equilibrium.
    """
    geqdsk_dir = Path(geqdsk_dir)
    pfile_dir = Path(pfile_dir)

    g_by_suffix = {
        f.name[1:]: f for f in geqdsk_dir.glob("g*") if f.is_file()
    }
    p_by_suffix = {
        f.name[1:]: f for f in pfile_dir.glob("p*") if f.is_file()
    }
    shared_suffixes = sorted(set(g_by_suffix) & set(p_by_suffix))

    by_shot = defaultdict(list)
    for suffix in shared_suffixes:
        shot, time = suffix.split(".", 1)
        by_shot[shot].append((time, suffix))
    for shot in by_shot:
        by_shot[shot].sort()

    shots = sorted(by_shot)
    selected = []
    time_idx = 0
    while len(selected) < equil_num:
        added_this_round = False
        for shot in shots:
            if len(selected) >= equil_num:
                break
            entries = by_shot[shot]
            if time_idx < len(entries):
                suffix = entries[time_idx][1]
                selected.append(
                    (str(g_by_suffix[suffix]), str(p_by_suffix[suffix]))
                )
                added_this_round = True
        if not added_this_round:
            break
        time_idx += 1

    if len(selected) < equil_num:
        raise ValueError(
            f"Requested {equil_num} equilibria but only found {len(selected)} "
            f"matching g/p pairs in {geqdsk_dir} and {pfile_dir}"
        )

    # print(f"Selected {len(selected)} g/p file pairs:")
    # for mhd_fp, kprof_fp in selected:
        # print(f"  g: {mhd_fp}\n  p: {kprof_fp}")
    return selected


equilibria = initialize_inputs(equil_num)

# Run scan: outer loop over equilibria (g/p pairs). Within each equilibrium we
# build ONE base_model and reuse it across all parameter combinations via
# update_free_params (cheap), instead of constructing a new model per combo.
# For each parameter combo we recompute the valid range of psi_N_inner_boundary
# from the slope-zero inner limit to the nFC/nCX-threshold outer limit, then
# sample n_psi_inner_pts boundaries within that range.
n_iter = 0
for eq_idx, (mhd_fp, kprof_fp) in enumerate(equilibria):
    print(f"Equilibrium {eq_idx + 1}/{len(equilibria)}: {mhd_fp}, {kprof_fp}")

    eq_tag = Path(mhd_fp).name[1:]  # e.g. "150840.03000"
    eq_dir = scan_success_dir / eq_tag
    eq_dir.mkdir(parents=True, exist_ok=True)

    eq_success_fp = eq_dir / success_fp
    eq_failure_fp = eq_dir / failure_fp
    eq_error_fp = eq_dir / error_messages_fp

    # Clear outputs from any previous scan for this equilibrium.
    for old in eq_dir.glob('ne_*.npy'):
        old.unlink()
    for fp in (eq_success_fp, eq_failure_fp, eq_error_fp):
        fp.unlink(missing_ok=True)
    with open(eq_success_fp, "w") as f:
        f.write("alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, psi_N_inner, L2_error\n")
    with open(eq_failure_fp, "w") as f:
        f.write("alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, psi_N_inner\n")
    with open(eq_error_fp, "w") as f:
        f.write("alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, psi_N_inner, message\n")

    # # Clear any prior runs for this equilibrium so re-running this cell yields
    # # a clean dataset.
    # for old in eq_dir.glob('ne_*.npy'):
    #     os.remove(old)

    # print(f"\n=== Equilibrium {eq_idx + 1}/{len(equilibria)}: {eq_tag} ===")

    # Build the base model ONCE per equilibrium — the expensive flux-surface
    # averaging and equilibrium loading runs here only, not inside the loop.

    base_model = saarelma_connor_nondim(
        P_tot_e      = P_tot_e,
        alpha_crit   = round(float(alpha_crits[0]), 3),
        C_KBM        = round(float(C_KBMs[0]), 3),
        De_chie_etg  = round(float(De_chie_etgs[0]), 3),
        nFC_x0       = round(float(nFC_x0s[0]), 3),
        ncx_x0_ratio = round(float(ncx_x0_ratios[0]), 3),
        mhd_fp       = mhd_fp,
        kprof_fp     = kprof_fp,
        verbose      = verbose,
    )
    # print(f"  Base model built for {eq_tag} — starting parameter scan.")
    i=0
    for alpha_crit in alpha_crits:
        j=0
        for C_KBM in C_KBMs:
            k=0
            for De_chie_etg in De_chie_etgs:
                l=0
                for nFC_x0 in nFC_x0s:
                    for ncx_x0_ratio in ncx_x0_ratios:
                        ac = round(float(alpha_crit), 3)
                        ck = round(float(C_KBM), 3)
                        de = round(float(De_chie_etg), 3)
                        nf = round(float(nFC_x0), 3)
                        nc = round(float(ncx_x0_ratio), 3)
                        try:
                                base_model.update_free_params(
                                        alpha_crit            = ac,
                                        C_KBM                 = ck,
                                        De_chie_etg           = de,
                                        nFC_x0                = nf,
                                        ncx_x0_ratio          = nc,
                                        psi_N_inner_boundary  = psi_val,
                                )
                                x_sol, ne_sol, nFC_sol, nCX_sol = base_model.solve_coupled_nondim(**SOLVE_KW)
                                sol = {'x': x_sol, 'y': ne_sol, 'nFC': nFC_sol, 'nCX': nCX_sol}
                        except Exception as e:  # run fails
                            with open(eq_failure_fp, 'a') as f:
                                f.write(f"{eq_tag}, {ac}, {ck}, {de}, {nf}, {nc}, {psi_val:.4f}\n")
                            with open(eq_error_fp, 'a') as f:
                                f.write(f"{eq_tag}, {ac}, {ck}, {de}, {nf}, {nc}, {psi_val:.4f}, {e}\n")
                        else:  # run works
                            # Calculate L2 error between predicted and pfile
                            ne_pfile = interp1d(base_model.x_init, base_model.n_e_pres, kind='linear', bounds_error=False, fill_value='extrapolate')(x_sol)
                            L2_error = np.linalg.norm(ne_sol - ne_pfile) / np.linalg.norm(ne_pfile) # normalized L2 error
                            with open(eq_success_fp, 'a') as f:
                                f.write(f"{eq_tag}, {ac}, {ck}, {de}, {nf}, {nc}, {psi_val:.4f}, {L2_error}\n")
                            np.save(eq_dir / f"ne_a{ac}_C{ck}_D{de}_n{nf}_nc{nc}_b{psi_val:.4f}", sol, allow_pickle=True)
        i+=1
        # print(f"Completed {i} of {len(alpha_crits)} alpha_crits for equilibrium {eq_tag}")
    del base_model # Drop the per-equilibrium base model before moving to the next one.