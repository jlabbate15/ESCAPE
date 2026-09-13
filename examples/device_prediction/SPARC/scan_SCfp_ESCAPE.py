import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import tempfile
import urllib.request # needed for geqdsk import
ROOT = Path.cwd().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.profiles_loop_solve import profiles_loop_solve

tokamaker_python_path = os.getenv('OFT_ROOTPATH')
if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path,'python'))
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import create_isoflux, read_eqdsk

from examples.device_prediction.helper_functions import read_sparcpublic_profiles, psi_n_and_rho_psi, build_manual_profs, calc_pressure_profile, _print_profile_summary, read_popcon, _to_watts


# Import the SPARC baseline scenario geqdsk
mhd_fp = 'gSPARC_DN_PRD.geqdsk'
eqdsk = read_eqdsk(mhd_fp)


# Import SPARC kinetic profiles from local SPARCPublic exports.
# TRANSP: 5 - transp_20221013.txt
# CGYRO:  6 - cgyro_20221013.txt
TRANSP_PATH = Path('5 - transp_20221013.txt')
CGYRO_PATH = Path('6 - cgyro_20221013.txt')
PROFILE_SOURCE = 'cgyro'   # 'transp' or 'cgyro' — feeds manual_profs for the solve
OVERPLOT_PROFILES = False   # if True, plot TRANSP and CGYRO together for comparison

read_transp_profiles = read_sparcpublic_profiles
read_cgyro_profiles = read_sparcpublic_profiles

profiles_transp = read_transp_profiles(TRANSP_PATH)
profiles_cgyro = read_cgyro_profiles(CGYRO_PATH)

source = PROFILE_SOURCE.strip().lower()
if source == 'transp':
    profiles = profiles_transp
    prof_path = TRANSP_PATH
elif source == 'cgyro':
    profiles = profiles_cgyro
    prof_path = CGYRO_PATH
else:
    raise ValueError(f"PROFILE_SOURCE must be 'transp' or 'cgyro', got {PROFILE_SOURCE!r}")

manual_profs = build_manual_profs(profiles)

print(f'Using {prof_path.name} ({source}) for manual_profs')
_print_profile_summary(source, prof_path, profiles)


# Import heating power and Zeff from SPARCPublic POPCON table
POPCON_PATH = Path('1 - PRD_POPCON_20221013.csv')
popcon = read_popcon(POPCON_PATH, keys=('Ohmic power', 'RF power', 'Zeff'))
P_ohmic = _to_watts(popcon['Ohmic power']['value'], popcon['Ohmic power']['unit'])  # W
P_RF = _to_watts(popcon['RF power']['value'], popcon['RF power']['unit'])            # W
Zeff = float(popcon['Zeff']['value'])  # dimensionless

# P_tot_e: heating power to electrons [W].
# Saarelma et al. take ~half of the total (Ohmic+RF) heating as electron channel.
P_tot_e = (P_ohmic + P_RF) / 2


# Parameters for ESCAPE #

# Output directory
output_dir = f'SPARC_workflow_noKBM_lessETG'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Scan parameters
x_res = 50
epednn_model = 'EPED1' # 'EPED1' or 'EPED_SPARC'
eped_tol_max = 1e-5
eped_iter_max = 5
kbm_treatment = "picard"
kbm_gate_eps = 0.1
picard_gate_mode = "average"
picard_max_it = 50
picard_rtol = 1e-8
picard_relax = 1.0
EPEDNN_core = 'stiff T_e and n_e'
verbose_EPEDNNloop = False
verbose_sc = False


import itertools

N = 3
alpha_crits = np.logspace(-1, 1, 1)
C_KBMs = np.array([0.0])
De_chie_etgs = np.logspace(-3, -2, N)
nFC_x0s = np.logspace(14.5, 16.5, N)
ncx_x0_ratios = np.array([0.1,1,10,20])

# ne_x0s = [None, 1e20, 2e20] # m^-3, manually specify outer bc for electron density
ne_x0s = [None] # m^-3, manually specify outer bc for electron density

scan_total = len(alpha_crits) * len(C_KBMs) * len(De_chie_etgs) * len(nFC_x0s) * len(ncx_x0_ratios) * len(ne_x0s)
print(f'Total number of scans: {scan_total}')

out_dir_ref = Path(output_dir)
j=0
for ne_x0 in ne_x0s:
    i=0
    for combo in itertools.product(alpha_crits, C_KBMs, De_chie_etgs, nFC_x0s, ncx_x0_ratios):
        free_params = {
            'alpha_crit': combo[0],
            'C_KBM': combo[1],
            'De_chie_etg': combo[2],
            'nFC_x0': combo[3],
            'ncx_x0_ratio': combo[4]
        }

        out_dir = out_dir_ref / Path(f'neouter{j}_fp{i}')


        try:
            ped_wid, ped_h_out, sol = profiles_loop_solve(
                MHD_FP = mhd_fp,
                kprof_loc = 'manual psi_N grid',
                manual_profs = manual_profs,
                P_tot_e = P_tot_e,
                species = 'D-T',
                Z_i = Zeff,
                out_dir = out_dir,
                x_res = x_res,
                ne_x0 = ne_x0,
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

            # Save results
            psi_N_p, p_Pa, p_mode = calc_pressure_profile(profiles)
            psi_ped_top = 1.0 - float(ped_wid)
            p_at_ped_top = float(np.interp(psi_ped_top, psi_N_p, p_Pa))

            # Save model output
            out_dict = {
                'mhd_fp': mhd_fp,
                'profiles': manual_profs, # need to edit this line
                'ESCAPE_ped_h': ped_h_out,
                'ESCAPE_ped_wid': ped_wid,
                'experimental_ped_h': p_at_ped_top,
                'free_params': free_params,
                'profiles_solved': sol, # profiles
                'i': i,
            }
            np.save(out_dir / 'out_dict.npy', out_dict)
        except Exception as e:
            out_dict = {'failed': True, 'free_params': free_params, 'error': str(e)}
            np.save(out_dir / 'failed.npy', out_dict)

        if i%50 == 0:
            print(f'Scan {i} of {scan_total} completed')
        i+=1
    j+=1
