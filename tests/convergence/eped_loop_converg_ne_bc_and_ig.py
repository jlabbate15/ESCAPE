# Purpose of this script is to see the convergence behavior of the EPED-SC loop
# specifically using the n_e from the previous loop as the boundary condition and initial guess

import sys
from pathlib import Path
import numpy as np

ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(ROOT))
from src.profiles_loop_solve import profiles_loop_solve
from ESCAPE.helpers.load_equil import initialize_inputs

from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk

verbose_EPEDNNloop = False
verbose_sc = False

# Output data and files
out_dir = 'eped_loop_converg_ne_bc_and_ig'
Path(out_dir).mkdir(parents=True, exist_ok=True)

# Scan parameters
x_res = 20
eped_tol_max = 1e-5
eped_iter_max = 1000
free_params = {
    'alpha_crit': 0.1,
    'C_KBM': 1,
    'De_chie_etg': 0.1,
    'nFC_x0': 3.16228e15,
    'ncx_x0_ratio': 17.783
}
ig = 'manual'

# Equilibria parameters
equil_num = 100
kprof_loc = 'pfile'
geqdsk_dir = Path('/mnt/homes_global/jal2351/software/sc_inputs/CAKEgeqdsks')
pfile_dir = Path('/mnt/homes_global/jal2351/software/sc_inputs/CAKEpfiles')
# geqdsk_dir = Path('/mnt/homes_global/jal2351/software/sc_inputs/gHighPerfHMode')
# pfile_dir = Path('/mnt/homes_global/jal2351/software/sc_inputs/OMFITnc_HighPerfHMode')
kbm_treatment = "picard"
kbm_gate_eps = 0.1
picard_gate_mode = "average"
picard_max_it = 50
picard_rtol = 1e-8
picard_relax = 1.0



def pres_pred(ped_wid, gfile_pres, gfile_pres_grid):
    """
    Calculate pedestal pressure from pedestal width and MHD equilibrium file.
    """
    psiN_top = 1 - ped_wid
    return np.interp(psiN_top, gfile_pres_grid, gfile_pres)

# Load in equilibria
equilibria = initialize_inputs(equil_num, geqdsk_dir, pfile_dir, p_filetype=kprof_loc)
equilibria = equilibria[1:] # remove first equilibrium because it is problematic
del equilibria[2] # remove another problematic equilibrium

# For each equilbrium, call profiles_loop_solve() from profiles_loop_solve.py
for mhd_fp, kprof_fp in equilibria:
    equil_tag = Path(mhd_fp).name[1:]
    out_path = Path(out_dir) / equil_tag
    out_path.mkdir(parents=True, exist_ok=True)
    print(f'Processing {mhd_fp}')
    try:
        ped_wid, ped_h_out, gfile_pres, gfile_pres_grid = profiles_loop_solve(
            MHD_FP = mhd_fp,
            kprof_loc = kprof_loc,
            KPROF_FP = kprof_fp,
            out_dir = out_dir,
            x_res = x_res,
            eped_tol_max = eped_tol_max,
            eped_iter_max = eped_iter_max,
            free_params = free_params,
            kbm_treatment = kbm_treatment,
            kbm_gate_eps = kbm_gate_eps,
            picard_gate_mode = picard_gate_mode,
            picard_max_it = picard_max_it,
            picard_rtol = picard_rtol,
            picard_relax = picard_relax,
            ig = ig,
            verbose = verbose_EPEDNNloop,
            verbose_sc = verbose_sc,
        )
        p_gfile = pres_pred(ped_wid, gfile_pres, gfile_pres_grid)
        out_dict = {
            'mhd_fp': mhd_fp,
            'kprof_fp': kprof_fp,
            'ped_h': ped_h_out,
            'ped_wid': ped_wid,
            'p_gfile': p_gfile,
        }
        np.save(out_path / 'success_outdict.npy', out_dict)
    except Exception as e:
        out_dict = {
            'error': e,
            'fail_string': 'lmao this failed'
        }
        np.save(out_path / 'failed_outdict.npy', out_dict)