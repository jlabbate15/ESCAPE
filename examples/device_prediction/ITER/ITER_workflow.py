import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import itertools
from pathlib import Path
import urllib.request # needed for geqdsk import
ROOT = Path.cwd().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.profiles_loop_solve import profiles_loop_solve
from examples.device_prediction.helper_functions import calc_pressure_profile


mhd_fp = '/mnt/homes_global/jal2351/software/saarelma-conner-ped/examples/device_prediction/ITER/ITER_Hmode_tMakerEx.geqdsk'

profiles = np.load('ITER_Hmode_tMakerEx_profiles.npy',allow_pickle=True).item()

psi_n = profiles['psi_n']
ne_ref = profiles['ne']
Te_ref = profiles['Te'] / 1e3 # keV

manual_profs = {
    'ne': ne_ref,
    'Te': Te_ref,
    'psi_N_ne': psi_n,
    'psi_N_Te': psi_n
}

# Parameters for ESCAPE #

# Output directory
out_dir_ref = 'ITER_output_logs'
Path(out_dir_ref).mkdir(parents=True, exist_ok=True)

verbose = True
verbose_sc = True

# Scan parameters
N = 3
alpha_crits = np.logspace(-1, 1, 1)
C_KBMs = np.array([0.0])
De_chie_etgs = np.logspace(-1, 0, N)
nFC_x0s = np.logspace(14.5, 16.5, N)
ncx_x0_ratios = np.array([0.1,1,10,20])
x_res = 50
epednn_model = 'EPED1' # 'EPED1' or 'EPED_SPARC'
eped_tol_max = 1e-5
eped_iter_max = 5
EPEDNN_core = 'stiff T_e and n_e'
kbm_treatment = "picard"
kbm_gate_eps = 0.1
picard_gate_mode = "average"
picard_max_it = 50
picard_rtol = 1e-8
picard_relax = 1.0
verbose_EPEDNNloop = False
verbose_sc = False

scan_total = len(alpha_crits) * len(C_KBMs) * len(De_chie_etgs) * len(nFC_x0s) * len(ncx_x0_ratios)
print(f'Total number of scans: {scan_total}')

# Model run
i=0
for combo in itertools.product(alpha_crits, C_KBMs, De_chie_etgs, nFC_x0s, ncx_x0_ratios):
    free_params = {
        'alpha_crit': combo[0],
        'C_KBM': combo[1],
        'De_chie_etg': combo[2],
        'nFC_x0': combo[3],
        'ncx_x0_ratio': combo[4]
    }
    out_dir = out_dir_ref / Path(f'fp{i}')

    try:
        ped_wid, ped_h_out, sol = profiles_loop_solve(
            MHD_FP = mhd_fp,
            kprof_loc = 'manual profs',
            manual_profs = manual_profs,
            P_tot_e = 40e6, # from https://iopscience-iop-org.ezproxy.cul.columbia.edu/article/10.1088/0029-5515/49/6/065012/pdf
            psi_N_inner = 0.85,
            out_dir = out_dir,
            species = 'D-T',
            # Z_i = Zeff,
            x_res = x_res,
            free_params = free_params,
            eped_tol_max = eped_tol_max,
            eped_iter_max = eped_iter_max,
            kbm_gate_eps = kbm_gate_eps,
            kbm_treatment = kbm_treatment,
            picard_gate_mode = picard_gate_mode,
            picard_max_it = picard_max_it,
            picard_rtol = picard_rtol,
            picard_relax = picard_relax,
            ig = 'manual',
            epednn_model = epednn_model,
            EPEDNN_core = EPEDNN_core, 
            verbose = verbose_EPEDNNloop,
            verbose_sc = verbose_sc,
        )

        # Save model output
        out_dict = {
            'mhd_fp': mhd_fp,
            'profiles': manual_profs, # experimental profiles
            'ESCAPE_ped_h': ped_h_out,
            'ESCAPE_ped_wid': ped_wid,
            'free_params': free_params,
            'profiles_solved': sol, # solved profiles
            'i': i,
        }
        np.save(out_dir / 'out_dict.npy', out_dict)
    except Exception as e:
        out_dict = {'failed': True, 'free_params': free_params, 'error': str(e)}
        np.save(out_dir / 'failed.npy', out_dict)

    if i%50 == 0:
        print(f'Scan {i} of {scan_total} completed')
    i+=1