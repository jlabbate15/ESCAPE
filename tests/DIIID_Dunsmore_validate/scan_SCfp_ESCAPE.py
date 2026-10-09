import os
import numpy as np
import sys
from pathlib import Path
import pickle # to check which state attributes np.save can store
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent # ESCAPE root (tests/DIIID_Dunsmore_validate/..)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from src.ESCAPE.ESCAPE_solve import ESCAPE_solve
from examples.device_prediction.helper_functions import calc_pressure_profile
from helpers.ped_width_proxy import ped_center, ped_width
from parse_dunsmore_db import (DB_PATH, EQDSK_DIR, load_db, load_equilibrium, build_kprof,
                               kprof_to_profiles, experimental_data, heating_power)
import time


# ---------------------- COMMON USER INPUTS ----------------------
equil_num = None # None for all
# equil_list = ['189627_2850','184307_2450']
equil_list = None # database keys '{shot}_{time_ms}'; None to use equil_num

alpha_crits = np.array([2])
nFC_x0s = np.array([1e15])
C_KBMs = np.array([1])
De_chie_etgs = np.array([0.05])
ncx_x0_ratios = np.array([15])

# Overridable from the environment (see submit_scan.sh)
sc_model = os.environ.get('SC_MODEL', '1D')
sc_implementation = os.environ.get('SC_IMPLEMENTATION', 'scipy')
ne_grad_bc_loc = os.environ.get('NE_GRAD_BC_LOC', 'outer')
picard_relax = float(os.environ.get('PICARD_RELAX', 0.5))
picard_max_it = int(os.environ.get('PICARD_MAX_IT', 50))
run_suffix = os.environ.get('RUN_SUFFIX', '_V4_fullP')

output_dir = f'outputs/DIIIDDunsmore_ESCAPE_{sc_model}{sc_implementation}{ne_grad_bc_loc}{run_suffix}'
# ----------------------------------------------------------------


Path(output_dir).mkdir(parents=True, exist_ok=True)

# Equilibria: the Dunsmore/Saarelma database carries kinetic profiles and
# scalars but no 2D equilibrium, so ESCAPE's g-files come from EQDSK_DIR
# (sc_inputs/Jamie_inputs_twindow: {shot}.{time_ms}.npy pickled OMFITgeqdsk
# objects at t_min and t_min + 100 ms of the entry's window, time-averaged
# into one equilibrium; a lone g-file is used as is). load_equilibrium checks it
# against the database (shot, time, Ip, Bt, R, a, kappa, delta, q95; see
# CHECK_TOL in parse_dunsmore_db.py); entries without a consistent g-file
# are skipped and listed.

db = load_db()
equil_num_total = len(db)
if equil_list is not None:
    equil_tags = list(equil_list)
else:
    equil_tags = list(db)[:equil_num if equil_num is not None else equil_num_total]
equilibria, skipped = {}, {}
for tag in equil_tags:
    try:
        equilibria[tag] = load_equilibrium(db[tag], EQDSK_DIR) # (g-file paths, time-averaged read_eqdsk-style dict)
    except (FileNotFoundError, ValueError) as e:
        skipped[tag] = str(e)
equil_tags = [tag for tag in equil_tags if tag in equilibria]
equil_num = len(equil_tags)
print(f'Number of equilibria found: {equil_num_total}')
for tag, reason in skipped.items():
    print(f'Skipping {tag}: {reason}')
print(f'Number of equilibria to process: {equil_num}')
if equil_num == 0:
    sys.exit('No database entries have a consistent g-file; check EQDSK_DIR in parse_dunsmore_db.py')


# ── Heating power ────────────────────────────────────────────────────────────
# Unlike the HighPerfHMode set, this database has per-shot powers (MW):
# ptot = pinj + pech + pohm, pnet = ptot - prad.
P_HEAT_KEY = 'pnet'     # net (NBI + ECH + Ohmic - radiated) heating, assumes dE/dt of plasma is small (since we already assume steady-state)
ELECTRON_FRAC = 1     # Saarelma et al. take ~half the heating as the electron channel

# ── Experimental n_e -> SC inputs ────────────────────────────────────────────
# ne_outer (ESCAPE ne_x0) is the experimental n_e at psi_N = 1, and the tanh
# initial-guess centre is psi_N = 1 - width/2 of the experimental pedestal
# (helpers/ped_width_proxy.ped_center, fitted on psi_N >= psi_N_inner_boundary);
# ESCAPE_solve maps that psi_N onto the SI x grid for sc_params['tanh_center'].
PED_CENTER_METHOD = 'tanh+grad' # ped_width_proxy method(s) averaged for the pedestal width
USE_EXP_TANH_CENTER = True # False: tanh_center=None (midpoint default) for every equilibrium
# First-ESCAPE-iteration n_e guess: 'tanh' (above) or 'tanh_linear' (a line from
# ne_inner with the experimental slope at x_inner out to psi_N_pwg = 1 - pedestal width,
# joined with matching value and slope to a tanh step down to ne_outer at the separatrix;
# ESCAPE_solve maps psi_N_pwg to sc_params['x_pwg'])
INITIAL_GUESS_0 = 'tanh' # 'tanh' or 'tanh_linear'


def state_to_dict(state):
    """Picklable snapshot of an ESCAPE state's attributes, for np.save.

    The state object itself can't be pickled: it holds the juliacall EPEDNN
    model and (after a Firedrake solve) Firedrake meshes/Functions. Those
    attributes are left out and their names recorded under '_not_saved'.
    juliacall values also pickle, but only unpickle inside a live Julia
    session ("error deserializing this value"), which makes the whole
    out_dict.npy unreadable elsewhere, so those are left out too.
    """
    saved, not_saved = {}, []
    for name, val in vars(state).items():
        try:
            blob = pickle.dumps(val)
        except Exception:
            not_saved.append(name)
            continue
        if b'juliacall' in blob:
            not_saved.append(name)
        else:
            saved[name] = val
    saved['_not_saved'] = not_saved
    return saved


# ESCAPE parameters
ESCAPE_tol_max = 1e-5
ESCAPE_iter_max = 1
out_dir = output_dir
quasineutral_flag = True
T_rat_flag = True # True if using a temperature ratio between ions and electrons, False if doing something else
T_rat = 1
pol_norm = False # True for when the poloidal flux is not normalized by 2pi. COCOS 7 convention is pol_norm=False, so poloidal flux is normalized by 2pi
species = 'D' # species of ions, currently supporting: D, D-T
regime_flag = 'PT H-mode' # regime of the plasma, currently supporting: 'PT H-mode', 'NT'
verbose_ESCAPE = False

# Saarelma-Connor initialization parameters
sc_params = {
    'psi_N_inner_boundary': 0.85, # normalized poloidal flux at the inner boundary (boundary condition); overridden by find_inner_boundary if nFC_threshold or nCX_threshold is set
    'nFC_threshold': None, # fraction of nFC at the separatrix below which the inner boundary is placed (None to disable)
    'nCX_threshold': None, # fraction of nCX at the separatrix below which the inner boundary is placed (None to disable)
    'x_method': 'radas', # method to use for the cross-section rates, currently supporting: 'adas', 'radas'
    'verbose': False,
    'model': sc_model,
    'implementation': sc_implementation,
}

# Saarelma-Connor (SC) shared solver parameters
sc_params.update({
    'x_res':40,
    'ne_grad_bc_loc':ne_grad_bc_loc,
    # 'Saarelma2023' (Eq. 20 separatrix gradient) needs the Neumann BC at the
    # separatrix, so the 'inner' runs fall back to the state.n_e gradient
    'dne_method': 'Saarelma2023' if ne_grad_bc_loc == 'outer' else 'state',
    'tau_par': 1e-3 if ne_grad_bc_loc == 'outer' else None, # s, parallel loss time (Saarelma2023 only)
    'initial_guess':'tanh', # ESCAPE_solve overrides: 'tanh' on the first ESCAPE iteration, 'state' after
    'picard_gate_mode':'steep_grad',
    'picard_max_it':picard_max_it,
    'picard_rtol':1e-6,
    'picard_relax':picard_relax,
    'reuse_setup':False,
})

# SC scipy specific
sc_params.update({
    'bvp_tol':1e-6, # solve_bvp ('inner')
    'bvp_max_nodes':5000,
    'ivp_method':"Radau", # solve_ivp ('outer')
    'ivp_rtol':1e-8,
    'ivp_atol':1e-10,
})

# SC firedrake specific
sc_params.update({
    'fe_degree':2, # firedrake specific
    'grad_bc_tol':1e-8,
    'grad_bc_max_it':25,
    'grad_bc_seed':None,
    'linear_solver':"lu",
    'ksp_rtol':1e-8,
    'ksp_max_it':200,
})

# SC 3D specific
sc_params.update({
    'nCX_ic':"scale nFC", # 'scale nFC' or 'solve' or 'state'
    'nFC_ic':"solve",
    'neutrals_treatment':"fem", # only for firedrake (FEM)
    'n_neutral_sub':4001, # only for firedrake (FEM)
})

# SC 1D specific
sc_params.update({
    'eq6_form':"complete",
    'first_step':"auto",
})

# EPEDNN parameters
epednn_params = {
    'epednn_model': 'EPED1', # 'EPED1' or 'EPED_SPARC'
    'pres_gfile': False, # use pressure from kprofs
}


import itertools

# ne_x0s = [None, 1e20, 2e20] # m^-3, manually specify outer bc for electron density
ne_x0s = [None] # m^-3, manually specify outer bc for electron density; None uses the experimental separatrix n_e

scan_total = len(alpha_crits) * len(C_KBMs) * len(De_chie_etgs) * len(nFC_x0s) * len(ncx_x0_ratios) * len(ne_x0s)
print(f'Total number of scans: {scan_total}*{equil_num}')

for equil_tag in equil_tags:
    entry = db[equil_tag]
    out_path = Path(output_dir) / equil_tag
    out_path.mkdir(parents=True, exist_ok=True)

    Z_eff = float(entry['Zeff']) # dimensionless, as measured (a few entries are < 1)
    Z_i = 1

    # ESCAPE_state inputs, plus the same profiles in calc_pressure_profile form
    # so out_dict['profiles'] and the experimental pedestal pressure match them.
    mhd_fps, equil_params = equilibria[equil_tag] # g-file paths, time-averaged read_eqdsk-style dict
    kprof_params = build_kprof(entry)
    profiles = kprof_to_profiles(kprof_params)
    psi_N_p, p_Pa, p_mode = calc_pressure_profile(profiles)

    # Experimental n_e -> ne_outer and the tanh initial-guess centre
    psin_ne_exp = np.asarray(profiles['psin_ne'], dtype=float)
    ne_exp = np.asarray(profiles['n_e'], dtype=float)
    ne_x0_exp = float(np.interp(1.0, psin_ne_exp, ne_exp)) # m^-3, experimental separatrix density
    ped_mask = psin_ne_exp >= sc_params['psi_N_inner_boundary'] # fit on the SC domain only
    tanh_center_psiN = None
    if USE_EXP_TANH_CENTER:
        try:
            tanh_center_psiN = ped_center(psin_ne_exp[ped_mask], ne_exp[ped_mask], method=PED_CENTER_METHOD)
        except (RuntimeError, ValueError) as e:
            print(f'{equil_tag}: pedestal centre fit failed ({e}); using the midpoint tanh_center')
    psiN_pwg = None
    if INITIAL_GUESS_0 == 'tanh_linear':
        psiN_pwg = 1.0 - ped_width(psin_ne_exp[ped_mask], ne_exp[ped_mask], method=PED_CENTER_METHOD)
    sc_params.pop('tanh_center', None) # ESCAPE_solve sets it in place; don't carry it to the next equilibrium
    sc_params.pop('x_pwg', None)
    sc_params['initial_guess'] = INITIAL_GUESS_0 # ESCAPE_solve overwrites it with 'state' after iteration 0

    # P_tot_e: heating power to electrons [W]
    P_tot_e = heating_power(entry, P_HEAT_KEY) * ELECTRON_FRAC

    tc_str = 'midpoint' if tanh_center_psiN is None else f'{tanh_center_psiN:.4f}'
    print(f'{equil_tag}: Z_eff = {Z_eff:.3f}, P_tot_e = {P_tot_e/1e6:.2f} MW, pressure: {p_mode}, '
          f'ne_x0 = {ne_x0_exp:.3e} m^-3, tanh_center psi_N = {tc_str}')

    out_dir_ref = Path(out_path) # out_dir / equil_tag

    j=0
    for ne_x0_scan in ne_x0s:
        ne_x0 = ne_x0_exp if ne_x0_scan is None else ne_x0_scan
        i=0
        for combo in itertools.product(alpha_crits, C_KBMs, De_chie_etgs, nFC_x0s, ncx_x0_ratios):
            free_params = {
                'alpha_crit': combo[0],
                'C_KBM': combo[1],
                'De_chie_etg': combo[2],
                'nFC_x0': combo[3],
                'ncx_x0_ratio': combo[4]
            }
            sc_params['free_params'] = free_params

            out_dir = out_dir_ref / Path(f'neouter{j}_fp{i}') # out_dir / equil_tag / neouterj_fpi
            out_dir.mkdir(parents=True, exist_ok=True)


            try:
                t0 = time.perf_counter()
                state = ESCAPE_solve(
                    sc_params,
                    epednn_params,
                    Z_i = Z_i, # Z of ions
                    Z_eff = Z_eff,
                    P_tot_e = P_tot_e, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002)
                    ne_x0 = ne_x0, # m^-3, electron density at the separatrix (experimental unless ne_x0s overrides it)
                    tanh_center_psiN = tanh_center_psiN, # psi_N, experimental pedestal centre for the tanh n_e initial guess
                    psiN_pwg = psiN_pwg, # psi_N, experimental pedestal top for the 'tanh_linear' n_e initial guess
                    equil_params = equil_params, # dictionary of required equilibrium parameters
                    kprof_params = kprof_params, # dictionary of required kinetic profile (density, temperature) parameters
                    quasineutral_flag = quasineutral_flag,
                    T_rat_flag = T_rat_flag, # True if using a temperature ratio between ions and electrons, False if doing something else
                    T_rat = T_rat,
                    pol_norm = pol_norm, # True for when the poloidal flux is not normalized by 2pi. COCOS 7 convention is pol_norm=False, so poloidal flux is normalized by 2pi
                    species = species, # species of ions, currently supporting: D, D-T
                    regime_flag = regime_flag, # regime of the plasma, currently supporting: 'PT H-mode', 'NT'
                    ESCAPE_iter_max = ESCAPE_iter_max,
                    ESCAPE_tol_max = ESCAPE_tol_max,
                    out_dir = out_dir, # directory to output files
                    verbose = verbose_ESCAPE,
                )
                t1 = time.perf_counter()


                # ESCAPE pedestal from the final EPEDNN call on the returned state
                ped_wid = float(state.pedestal_width)      # normalized poloidal flux (float() drops any juliacall wrapper)
                # EPED1 returns MPa (as ESCAPE_solve assumes); convert to Pa to
                # match the experimental p_Pa below
                ped_h_out = float(state.pedestal_pressure) * 1e6 # Pa

                # Experimental pedestal pressure at psi_N = 1 - ped_wid, from the
                # input profiles (psi_N_p / p_Pa built once per equilibrium above).
                psi_ped_top = 1.0 - float(ped_wid)
                p_at_ped_top = float(np.interp(psi_ped_top, psi_N_p, p_Pa))

                # Save model output
                out_dict = {
                    'equil_tag': equil_tag,
                    'mhd_fp': ','.join(str(fp) for fp in mhd_fps),
                    'db_path': str(DB_PATH),
                    'profiles': profiles,
                    'ESCAPE_ped_h': ped_h_out, # Pa
                    'ESCAPE_ped_wid': ped_wid,
                    'experimental_ped_h': p_at_ped_top, # Pa
                    'experimental': experimental_data(entry), # database entry (LLAMA n0/ionization, raw profiles, ...)
                    'free_params': free_params,
                    'p_mode': p_mode,
                    'Zeff': Z_eff,
                    'P_tot_e': P_tot_e,
                    'i': i,
                    'ne_x0': ne_x0, # m^-3
                    'tanh_center_psiN': tanh_center_psiN,
                    'initial_guess_0': INITIAL_GUESS_0,
                    'psiN_pwg': psiN_pwg,
                    'x_pwg': getattr(state, 'x_pwg_used', None), # m, as passed to the SC solver
                    'tanh_center_x': getattr(state, 'tanh_center_used', None), # m, as passed to the SC solver
                    'state': state_to_dict(state), # picklable ESCAPE state attributes
                    'ESCAPE_runtime': t1-t0,
                }
                np.save(out_dir / 'out_dict.npy', out_dict)
            except Exception as e:
                out_dict = {'failed': True, 'free_params': free_params, 'error': str(e),
                            'ne_x0': ne_x0, 'tanh_center_psiN': tanh_center_psiN,
                            'initial_guess_0': INITIAL_GUESS_0, 'psiN_pwg': psiN_pwg}
                np.save(out_dir / 'failed.npy', out_dict)

            if i%50 == 0:
                print(f'Scan {i} of {scan_total} completed')
            i+=1
        j+=1
