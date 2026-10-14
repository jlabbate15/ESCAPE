import sys
from pathlib import Path
import json

import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np

ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(ROOT))
from escape.profiles_loop_solve_scloop import profiles_loop_solve_scloop

from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk

verbose = True

# Output data and files
ne_success_fp = 'success_profloop_1'
success_fp = 'success_PTHmode.txt'
failure_fp = 'failure_PTHmode.txt'
error_messages_fp = 'error_messages_PTHmode.txt'

# Scan parameters
N = 4
x_res = 40
eped_tol_max = 1e-5
alpha_crits_minmax = [-1,1]
C_KBMs_minmax = [-1,1]
De_chie_etgs_minmax = [-1,1]
nFC_x0s_minmax = [14,18]
ncx_x0_ratios_minmax = [0.1,1.3]
eped_iter_max = 50

# Equilibria parameters
equil_num = 5
geqdsk_dir = Path('/mnt/homes_global/jal2351/software/sc_inputs/CAKEgeqdsks')
pfile_dir = Path('/mnt/homes_global/jal2351/software/sc_inputs/CAKEpfiles')


# Load in equilibria
from collections import defaultdict

def initialize_inputs(equil_num, geqdsk_dir=geqdsk_dir, pfile_dir=pfile_dir):
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

    print(f"Selected {len(selected)} g/p file pairs:")
    for mhd_fp, kprof_fp in selected:
        print(f"  g: {mhd_fp}\n  p: {kprof_fp}")
    return selected

equilibria = initialize_inputs(equil_num, geqdsk_dir, pfile_dir)

def pres_pred(ped_wid, gfile_pres, gfile_pres_grid):
    """
    Calculate pedestal pressure from pedestal width and MHD equilibrium file.
    """
    psiN_top = 1 - ped_wid
    return np.interp(psiN_top, gfile_pres_grid, gfile_pres)


# For each equilbrium, call profiles_loop_solve() from profiles_loop_solve.py
output_root = Path(ne_success_fp)
output_root.mkdir(parents=True, exist_ok=True)
eped_loop_success_fp = output_root / 'eped_loop_success.txt'
eped_loop_failure_fp = output_root / 'eped_loop_failed.txt'

failed=[]
success=[]
for mhd_fp, kprof_fp in equilibria:
    print(mhd_fp)
    print(kprof_fp)
    eq_tag = Path(mhd_fp).name[1:]  # e.g. "150840.03000"
    eq_dir = output_root / eq_tag
    eq_dir.mkdir(parents=True, exist_ok=True)

    try:
        ped_wid, ped_h_out, gfile_pres, gfile_pres_grid = profiles_loop_solve_scloop(
            MHD_FP = mhd_fp,
            KPROF_FP = kprof_fp,
            ne_success_fp = eq_dir,
            initial_guess = "tanh",
            ne_inner_bc = "neumann",
            success_fp = eq_dir / success_fp,
            failure_fp = eq_dir / failure_fp,
            error_messages_fp = eq_dir / error_messages_fp,
            x_res = x_res,
            N = N,
            alpha_crits_minmax = alpha_crits_minmax,
            C_KBMs_minmax = C_KBMs_minmax,
            De_chie_etgs_minmax = De_chie_etgs_minmax,
            nFC_x0s_minmax = nFC_x0s_minmax,
            ncx_x0_ratios_minmax = ncx_x0_ratios_minmax,
            eped_tol_max = eped_tol_max,
            eped_iter_max = eped_iter_max,
            verbose = verbose,
        )
        p_gfile = pres_pred(ped_wid, gfile_pres, gfile_pres_grid)
        out_dict = {
            'eq_tag': eq_tag,
            'mhd_fp': mhd_fp,
            'kprof_fp': kprof_fp,
            'ped_h': ped_h_out,
            'ped_wid': ped_wid,
            'p_gfile': p_gfile,
        }
        success.append(out_dict)
        with open(eped_loop_success_fp, 'a') as f:
            f.write(json.dumps(out_dict, default=str) + '\n')
    except Exception:
        failed.append(eq_tag)
        with open(eped_loop_failure_fp, 'a') as f:
            f.write(f'{eq_tag}\n')