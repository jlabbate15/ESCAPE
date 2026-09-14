import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import urllib.request # needed for geqdsk import
ROOT = Path.cwd().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.profiles_loop_solve import profiles_loop_solve

from examples.device_prediction.helper_functions import calc_pressure_profile

import itertools


# Output directory
output_dir = 'ARC_workflow_noKBM_ETG05_large'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Free parameters
alpha_crits = np.logspace(-1, 1, 1)
C_KBMs = np.array([0.0])
De_chie_etgs = np.array([0.5])
nFC_x0s = np.logspace(14.5, 17.0, 5)
ncx_x0_ratios = np.array([0.1,1,10,20,30])

# geqdsk
mhd_fp = '../geqdsk-ARCv3a'


scan_total = len(alpha_crits) * len(C_KBMs) * len(De_chie_etgs) * len(nFC_x0s) * len(ncx_x0_ratios)
print(f'Total number of scans: {scan_total}')


def load_digitized_profile(filepath):
    """Load psi_N and profile data from a Web Plot Digitizer CSV export.

    Parameters
    ----------
    filepath : str or Path
        Path to an ARC_ne.csv / ARC_Te.csv file.

    Returns
    -------
    psi_N : ndarray
        Normalized poloidal flux, sorted ascending and clipped to [0, 1].
    profile : ndarray
        ne in 10^20 m^-3 or Te in keV, depending on the file.
    """
    data = np.loadtxt(filepath, delimiter=',')
    psi_N, profile = data[:, 0], data[:, 1]

    # Digitization noise can push the outermost point just past the separatrix
    psi_N = np.clip(psi_N, 0.0, 1.0)

    order = np.argsort(psi_N, kind='stable')
    psi_N, profile = psi_N[order], profile[order]

    # Clipping can collide two points at psi_N = 1; keep the outermost sample
    unique = np.append(np.diff(psi_N) > 0, True)
    return psi_N[unique], profile[unique]


ne_fp = '../ARC_ne.csv'
Te_fp = '../ARC_Te.csv'

psi_N_ne, ne_ref = load_digitized_profile(ne_fp)   # 10^20 m^-3
psi_N_Te, Te_ref = load_digitized_profile(Te_fp)   # keV

print(f'Loaded ne: {len(psi_N_ne)} points, psi_N in [{psi_N_ne[0]:.4f}, {psi_N_ne[-1]:.4f}], '
      f'{ne_ref[0]:.2f} -> {ne_ref[-1]:.2f} x10^20 m^-3')
print(f'Loaded Te: {len(psi_N_Te)} points, psi_N in [{psi_N_Te[0]:.4f}, {psi_N_Te[-1]:.4f}], '
      f'{Te_ref[0]:.2f} -> {Te_ref[-1]:.2f} keV')

# kprof_loc = 'manual profs' reads the psi_N grids directly (no rho -> psi_N
# mapping) and expects n_e in m^-3 rather than 10^20 m^-3.
manual_profs = {
    'ne': ne_ref * 1e20,   # 10^20 m^-3 -> m^-3
    'Te': Te_ref,          # keV
    'psi_N_ne': psi_N_ne,
    'psi_N_Te': psi_N_Te,
}


# Parameters for ESCAPE #
x_res = 50
solver_structure = "firedrake"
epednn_model = 'EPED1' # 'EPED1' or 'EPED_SPARC'
eped_tol_max = 1e-5
eped_iter_max = 15
EPEDNN_core = 'stiff T_e and n_e'
kbm_treatment = "picard"
kbm_gate_eps = 0.1
picard_gate_mode = "average"
picard_max_it = 50
picard_rtol = 1e-8
picard_relax = 1.0
ne_grad_bc_loc = "inner"
verbose_EPEDNNloop = False
verbose_sc = False

out_dir_ref = Path(output_dir)

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
            P_tot_e = ( 21.5 + 0.8 + 227 ) * 0.5 * 1e6, # table 4 of Hillesheim et al. 2026
            psi_N_inner = 0.75,
            out_dir = out_dir,
            species = 'D-T',
            # Z_i = Zeff,
            x_res = x_res,
            solver_structure = solver_structure,
            ne_grad_bc_loc = ne_grad_bc_loc,
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