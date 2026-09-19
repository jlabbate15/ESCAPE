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

from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk


# ------------------------------------------------------------
# Inputs
verbose_EPEDNNloop = False
verbose_sc = False

# Output directory
output_dir = 'multiequil_PTHmode_Snyder_bc_newfp'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Scan parameters
x_res = 20
eped_tol_max = 1e-5
eped_iter_max = 5
# From param_err.py scan with kbm_treatment= picard average, for the non-Snyder CAKE database: alpha_crit=0.1, C_KBM=0.1, De_chie_etg=0.1, nFC_x0=3.16228e+15, ncx_x0_ratio=1.259
# From a N=3 scan for the Snyder database: alpha_crit=1, C_KBM=1, De_chie_etg=0.1, nFC_x0=3.16228e+15, ncx_x0_ratio=4.732
free_params = {
    'alpha_crit': 1,
    'C_KBM': 1,
    'De_chie_etg': 0.1,
    'nFC_x0': 3.16228e15,
    'ncx_x0_ratio': 4.732
}
ig = 'solve' # solve=bc, manual=bc+ig
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
print(f'Number of equilibria: {equil_num}')
# ------------------------------------------------------------

# Load in equilibria
equilibria = initialize_inputs(equil_num, geqdsk_dir, pfile_dir, p_filetype='OMFITnc')

# For each equilbrium, call profiles_loop_solve() from profiles_loop_solve.py
failed=[]
success=[]
for mhd_fp, kprof_fp in equilibria:
    equil_tag = Path(mhd_fp).name[1:]
    out_path = Path(output_dir) / equil_tag
    out_path.mkdir(parents=True, exist_ok=True)
    print(f'Processing {mhd_fp}')
    try:
        print(mhd_fp)
        ped_wid, ped_h_out = profiles_loop_solve(
            MHD_FP = mhd_fp,
            kprof_loc = kprof_loc,
            KPROF_FP = kprof_fp,
            out_dir = output_dir,
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
        out_dict = {
            'mhd_fp': mhd_fp,
            'kprof_fp': kprof_fp,
            'ped_h': ped_h_out,
            'ped_wid': ped_wid,
        }
        np.save(out_path / 'success_outdict.npy', out_dict)
    except Exception as e:
        out_dict = {
            'mhd_fp': mhd_fp,
            'kprof_fp': kprof_fp,
            'error': e,
        }
        np.save(out_path / 'failed_outdict.npy', out_dict)
