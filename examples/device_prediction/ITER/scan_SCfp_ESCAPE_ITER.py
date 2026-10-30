import os
import numpy as np
import sys
from pathlib import Path
import pickle # to check which state attributes np.save can store
import netCDF4 as nc # OMFIT profile files are netCDF
ROOT = Path.cwd().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.ESCAPE.ESCAPE_solve import ESCAPE_solve
from helpers.load_equil import initialize_inputs
from examples.device_prediction.helper_functions import calc_pressure_profile
import time
from helpers.parse_equil import mhd_load
from helpers.parse_prof import kprof_load


# ---------------------- COMMON USER INPUTS ----------------------
alpha_crits = np.array([0.01])
nFC_x0s = [1e13,1e14,5e14,1e15,5e15,1e16,5e16,1e17,5e17,1e18,5e18,1e19,5e19,1e20,5e20,1e21,5e21,1e22,5e22,1e23]
C_KBMs = np.array([0.0])
De_chie_etgs = [0.001,0.01,0.1,1,10]
ncx_x0_ratios = np.array([15])

# Filepath to geqdsk equilibrium state and profiles
mhd_fp = 'ITER_Hmode_tMakerEx.geqdsk'
prof_fp = Path('ITER_Hmode_tMakerEx_profiles.npy') # psi_n, ne [m^-3], Te [eV]

# Overridable from the environment (see submit_scan_ITER.sh)
sc_model = os.environ.get('SC_MODEL', '3D')
sc_implementation = os.environ.get('SC_IMPLEMENTATION', 'firedrake')
ne_grad_bc_loc = os.environ.get('NE_GRAD_BC_LOC', 'inner')
picard_relax = os.environ.get('PICARD_RELAX', '1.0') # float, or a schedule 'it:relax;it:relax;...' (no commas: sbatch --export splits on them)
picard_relax = ([(int(s), float(r)) for s, r in (e.split(':') for e in picard_relax.split(';'))]
                if ':' in picard_relax else float(picard_relax))
picard_rel_tol = float(os.environ.get('PICARD_REL_TOL', 1e-6)) # SC Picard loop: converged when |dne|_rel < this
run_suffix = os.environ.get('RUN_SUFFIX', '')
eq6_form = os.environ.get('EQ6_FORM', 'complete') # 'complete' or 'paper'
eq7_form = os.environ.get('EQ7_FORM', 'complete') # 'complete' or 'paper' (1D scipy only)
impurity_dilution = os.environ.get('IMPURITY_DILUTION', 'none') # 'none' or 'carbon' (1D scipy only; KBM alpha)
impurity_dilution = None if impurity_dilution.lower() == 'none' else impurity_dilution
D_TG_treatment = os.environ.get('D_TG_TREATMENT', 'implicit') # 'implicit' or 'picard' (1D scipy only)
picard_gate_mode = os.environ.get('PICARD_GATE_MODE', 'steep_grad') # 'average' or 'steep_grad' (KBM alpha gate)
bvp_tol = float(os.environ.get('BVP_TOL', 1e-6)) # solve_bvp tolerance ('inner', scipy only)
bvp_max_nodes = int(os.environ.get('BVP_MAX_NODES', 5000)) # solve_bvp max mesh nodes ('inner', scipy only)
# First-iteration n_e pedestal top, where EPEDNN reads neped (helpers/ped_width_proxy.ped_top_width):
# 'tanh', 'grad', 'lin', 'tanh+grad', ..., 'all' (psi_N = 1 - width), or 'corner' (two-line fit corner)
ped_width_method = os.environ.get('PED_WIDTH_METHOD', 'tanh+grad')

output_dir = (f'outputs/ITER_ESCAPE_{sc_model}{sc_implementation}{ne_grad_bc_loc}'
              + (f'_{run_suffix}' if run_suffix else ''))
# ----------------------------------------------------------------


Path(output_dir).mkdir(parents=True, exist_ok=True)

# Equilibria parameters
equil_num = 1
equil_list = None
mhd_loc = 'eqdsk'
kprof_loc = 'manual profs'
print(f'Number of equilibria to process: {equil_num}')

# Heating power
P_tot_e = 40e6 # W, as in ITER_workflow.py (from https://iopscience.iop.org/article/10.1088/0029-5515/49/6/065012/pdf)

# Starting profiles (as in ITER_workflow.py)
profiles_in = np.load(prof_fp, allow_pickle=True).item()
psi_n = np.asarray(profiles_in['psi_n'], dtype=float)

kprof_params = {
    'n_e': np.asarray(profiles_in['ne'], dtype=float),         # m^-3
    'T_e': np.asarray(profiles_in['Te'], dtype=float) / 1e3,   # eV -> keV
    'psin_ne': psi_n,
    'psin_Te': psi_n,
}

print(f'Loaded {prof_fp.name}: {len(psi_n)} points, psi_N in [{psi_n[0]:.4f}, {psi_n[-1]:.4f}], '
      f'ne {kprof_params["n_e"][0]/1e20:.2f} -> {kprof_params["n_e"][-1]/1e20:.2f} x10^20 m^-3, '
      f'Te {kprof_params["T_e"][0]:.2f} -> {kprof_params["T_e"][-1]:.2f} keV')


def load_equilibrium(mhd_fp, mhd_loc):
    """Build the ESCAPE_state inputs for one g-file.

    Returns
    -------
    equil_params : dict
        read_eqdsk dictionary of the g-file (psirz, fpol, pres, ip, rzout, ...),
        which is what ESCAPE_state.set_equil_params consumes.
    """
    equil_params = mhd_load(None, mhd_loc, mhd_fp)
    return equil_params


def state_to_dict(state):
    """Picklable snapshot of an ESCAPE state's attributes, for np.save.

    The state object itself can't be pickled: it holds the juliacall EPEDNN
    model and (after a Firedrake solve) Firedrake meshes/Functions. Those
    attributes are left out and their names recorded under '_not_saved'.
    juliacall values (epednn_model) do pickle, but only unpickle in a process
    with the Julia model loaded, so they are left out too.
    """
    saved, not_saved = {}, []
    for name, val in vars(state).items():
        if type(val).__module__.startswith('juliacall'):
            not_saved.append(name)
            continue
        try:
            pickle.dumps(val)
        except Exception:
            not_saved.append(name)
        else:
            saved[name] = val
    saved['_not_saved'] = not_saved
    return saved


# ESCAPE parameters
ESCAPE_tol_max = 1e-5
ESCAPE_iter_max = 5
out_dir = output_dir
ne_x0 = None, # m^-3, electron density at the separatrix (boundary condition, default is to use from profiles)        
quasineutral_flag = True
T_rat_flag = True # True if using a temperature ratio between ions and electrons, False if doing something else
T_rat = 1
pol_norm = False # True for when the poloidal flux is not normalized by 2pi. COCOS 7 convention is pol_norm=False, so poloidal flux is normalized by 2pi
species = 'D-T' # species of ions, currently supporting: D, D-T
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
    'picard_gate_mode':picard_gate_mode,
    'picard_max_it':50,
    'picard_rtol':picard_rel_tol,
    'picard_relax':picard_relax,
    'reuse_setup':False,
})

# SC scipy specific
sc_params.update({
    'bvp_tol':bvp_tol, # solve_bvp ('inner')
    'bvp_max_nodes':bvp_max_nodes,
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
    'eq6_form':eq6_form,
    'eq7_form':eq7_form, # scipy only
    'impurity_dilution':impurity_dilution, # scipy only
    'first_step':"auto",
    'D_TG_treatment':D_TG_treatment, # scipy only
})

# EPEDNN parameters
epednn_params = {
    'epednn_model': 'EPED1', # 'EPED1' or 'EPED_SPARC'
    'pres_gfile': False, # use pressure from kprofs
}


import itertools

# ne_x0s = [None, 1e20, 2e20] # m^-3, manually specify outer bc for electron density
ne_x0s = [None] # m^-3, manually specify outer bc for electron density

scan_total = len(alpha_crits) * len(C_KBMs) * len(De_chie_etgs) * len(nFC_x0s) * len(ncx_x0_ratios) * len(ne_x0s)
print(f'Total number of scans: {scan_total}*{equil_num}')

Z_eff = 1.7 # https://lac913.epfl.ch/epsppd3/2018/pdf/P1.1077.pdf
Z_i = 1

# ESCAPE_state inputs, plus the same profiles in calc_pressure_profile form
# so out_dict['profiles'] and the experimental pedestal pressure match them.
equil_params = load_equilibrium(mhd_fp, mhd_loc)

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
        sc_params['free_params'] = free_params

        # ESCAPE_solve clears its out_dir on every call, so each scan point gets its own
        run_dir = f'{out_dir}/run_{i}'

        try:
            t0 = time.perf_counter()
            state = ESCAPE_solve(
                sc_params,
                epednn_params,
                Z_i = Z_i, # Z of ions
                Z_eff = Z_eff, 
                P_tot_e = P_tot_e, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
                ne_x0 = ne_x0, # m^-3, electron density at the separatrix (boundary condition, default is to use from profiles)        
                ped_width_method = ped_width_method, # first-iteration n_e pedestal top
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
                out_dir = run_dir, # directory to output files
                verbose = verbose_ESCAPE,
            )
            t1 = time.perf_counter()

            # Save model output
            out_dict = {
                'mhd_fp': mhd_fp,
                'prof_fp': str(prof_fp),
                'profiles': kprof_params, # experimental profiles
                'free_params': free_params,
                'ped_width_method': ped_width_method,
                'Zeff': Z_eff,
                'P_tot_e': P_tot_e,
                'i': i,
                'state': state_to_dict(state), # picklable ESCAPE state attributes
                'ESCAPE_runtime': t1-t0,
            }
            np.save(run_dir + '/out_dict.npy', out_dict)
        except Exception as e:
            out_dict = {'failed': True, 'free_params': free_params, 'error': str(e)}
            Path(run_dir).mkdir(parents=True, exist_ok=True) # in case ESCAPE_solve raised before creating it
            np.save(run_dir + '/failed.npy', out_dict)

        if i%50 == 0:
            print(f'Scan {i+1} of {scan_total} completed')
        i+=1
    j+=1