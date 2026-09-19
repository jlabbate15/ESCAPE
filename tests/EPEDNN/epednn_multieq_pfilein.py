'''
100 equilibria, set density to pfile
run the EPEDNN without loop and make a plot of pressure predicted
vs. pfile pressure at psiN=1-delta and psiN=0.85
'''

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(ROOT))
from src.solver_nondim import saarelma_connor_nondim
from ESCAPE.helpers.load_equil import initialize_inputs


# ------------------------------------------------------------
# Inputs
verbose = False
psi_N_inner = 0.85
kprof_loc = 'p'

# Output directory
output_dir = 'multiequil_epednn_pfile'

# Equilibria parameters
equil_num = 100
geqdsk_dir = Path('/mnt/homes_global/jal2351/software/sc_inputs/CAKEgeqdsks')
pfile_dir = Path('/mnt/homes_global/jal2351/software/sc_inputs/CAKEpfiles')
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
i=1
for mhd_fp, kprof_fp in equilibria:
    try:
        print(f'Equilibrium {i}: {mhd_fp}')
        sc = saarelma_connor_nondim(
            psi_N_inner_boundary = psi_N_inner,
            mhd_fp       = mhd_fp,
            kprof_loc    = kprof_loc,
            kprof_fp     = kprof_fp,
            verbose      = verbose,
            P_tot_e      = 1, # dummy value, not used
            alpha_crit   = 1, # dummy value, not used
            C_KBM        = 1, # dummy value, not used
            De_chie_etg  = 1, # dummy value, not used
            nFC_x0       = 1, # dummy value, not used
            ncx_x0_ratio = 1, # dummy value, not used
        )

        gfile_pres = sc.pres_gfile
        gfile_pres_grid = sc.psi_N_pres
        
        sc.setup_solver_grids() # needed for self.x_init
        ne_ped = sc.n_e_pres
        x_ne = sc.x_init

        # Run EPEDNN
        sc.setup_epednn()
        ped_h_out, ped_w_out = sc.feed_epednn(ne_ped = ne_ped, x_ne = x_ne, EPEDNN_core = 'pfile')
        p_gfile = pres_pred(ped_w_out, gfile_pres, gfile_pres_grid)
    except Exception as e:
        print(f'{mhd_fp} failed')
        fail_dict = {'mhd_fp': mhd_fp, 'error': e}
        failed.append(fail_dict)
    else:
        print(f'{mhd_fp} successful')
        out_dict = {
            'mhd_fp': mhd_fp,
            'kprof_fp': kprof_fp,
            'ped_h': ped_h_out,
            'ped_wid': ped_w_out,
            'p_gfile': p_gfile,
        }
        success.append(out_dict)
    i+=1
np.save(output_dir+'/success.npy', success)
np.save(output_dir+'/failed.npy', failed)