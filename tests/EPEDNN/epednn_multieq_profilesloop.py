'''
100 equilibria, use one free parameter combination (or maybe 5) 
and then run the EPEDNN loop and make a plot of pressure predicted 
vs. pfile pressure at psiN=1-delta and psiN=0.85
'''

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(ROOT))
from src.profiles_loop_solve import profiles_loop_solve
from ESCAPE.helpers.load_equil import initialize_inputs


# ------------------------------------------------------------
# Inputs
verbose_EPEDNNloop = False
verbose_sc = False

# Output directory
output_dir = 'multiequil_PTHmode_Snyder'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Scan parameters
x_res = 20
eped_tol_max = 1e-5
eped_iter_max = 5
# From param_err.py scan with kbm_treatment= picard average: alpha_crit=0.1, C_KBM=0.1, De_chie_etg=0.1, nFC_x0=3.16228e+15, ncx_x0_ratio=1.259
free_params = {
    'alpha_crit': 0.1,
    'C_KBM': 0.1,
    'De_chie_etg': 0.1,
    'nFC_x0': 3.16228e15,
    'ncx_x0_ratio': 1.259
}
ig = 'solve' # bc
EPEDNN_core = 'previous T, stiched ne'
kbm_treatment = "picard"
kbm_gate_eps = 0.1
picard_gate_mode = "average"
picard_max_it = 50
picard_rtol = 1e-8
picard_relax = 1.0

# Equilibria parameters
geqdsk_dir = Path('/mnt/homes_global/jal2351/software/sc_inputs/gHighPerfHMode')
pfile_dir = Path('/mnt/homes_global/jal2351/software/sc_inputs/OMFITnc_HighPerfHMode')
kprof_loc = 'OMFITnc'
equil_num = len(list(pfile_dir.glob('*.cdf')))
# ------------------------------------------------------------

# Load in equilibria
equilibria = initialize_inputs(equil_num, geqdsk_dir, pfile_dir)

def pres_pred(ped_wid, gfile_pres, gfile_pres_grid):
    """
    Calculate pedestal pressure from pedestal width and MHD equilibrium file.
    """
    psiN_top = 1 - ped_wid
    return np.interp(psiN_top, gfile_pres_grid, gfile_pres)

# For each equilbrium, call profiles_loop_solve() from profiles_loop_solve.py
failed=[]
success=[]
for mhd_fp, kprof_fp in equilibria:
    try:
        print(mhd_fp)
        ped_wid, ped_h_out, gfile_pres, gfile_pres_grid = profiles_loop_solve(
            MHD_FP = mhd_fp,
            KPROF_FP = kprof_fp,
            ne_success_fp = output_dir,
            x_res = x_res,
            free_params = free_params,
            eped_tol_max = eped_tol_max,
            eped_iter_max = eped_iter_max,
            kbm_gate_eps = kbm_gate_eps,
            EPEDNN_core = EPEDNN_core,
            kbm_treatment = kbm_treatment,
            picard_gate_mode = picard_gate_mode,
            picard_max_it = picard_max_it,
            picard_rtol = picard_rtol,
            picard_relax = picard_relax,
            verbose = verbose,
        )
        p_gfile = pres_pred(ped_wid, gfile_pres, gfile_pres_grid)
    except Exception as e:
        fail_dict = {'mhd_fp': mhd_fp, 'error': e}
        failed.append(fail_dict)
    else:
        out_dict = {
            'mhd_fp': mhd_fp,
            'kprof_fp': kprof_fp,
            'ped_h': ped_h_out,
            'ped_wid': ped_wid,
            'p_gfile': p_gfile,
        }
        success.append(out_dict)
np.save(output_dir+'/success.npy', success)
np.save(output_dir+'/failed.npy', failed)