import os
import numpy as np
import sys
from pathlib import Path
import pickle # to check which state attributes np.save can store
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent # ESCAPE root (examples/device_prediction/ARC/../../..)
sys.path.insert(0, str(ROOT))
from src.ESCAPE.ESCAPE_api import ESCAPE # ESCAPE_state + epednn_class (+ saarelma_connor, unused here)
from helpers.parse_equil import mhd_load
from examples.device_prediction.helper_functions import calc_pressure_profile
import time


# ---------------------- COMMON USER INPUTS ----------------------
# Filepath to geqdsk equilibrium state and profiles
mhd_fp = HERE / 'ITER_Hmode_tMakerEx.geqdsk'
prof_fp = HERE / 'ITER_Hmode_tMakerEx_profiles.npy' # psi_n, ne [m^-3], Te [eV]

# Density pedestal top at which neped is read off the input n_e profile. Hardcoded: the ITER
# n_e profile has no clear density pedestal, so the width proxy (helpers/ped_width_proxy, used
# for ARC/SPARC) puts it at psi_N ~ 0.76
PSI_NE_PED = 0.85

run_suffix = os.environ.get('RUN_SUFFIX', '')

output_dir = HERE / ('outputs/ITER_EPEDNN' + (f'_{run_suffix}' if run_suffix else ''))
# ----------------------------------------------------------------


Path(output_dir).mkdir(parents=True, exist_ok=True)

# Equilibrium parameters
mhd_loc = 'eqdsk'

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
    equil_params = mhd_load(None, mhd_loc, str(mhd_fp))
    return equil_params


def kprof_to_profiles(kprof_params):
    """kprof_params -> the layout calc_pressure_profile expects, so the
    input pedestal pressure comes from the same profiles EPEDNN is fed.
    With no Ti/ni keys, calc_pressure_profile uses p = 2*ne*Te."""
    return {
        'psin_ne': kprof_params['psin_ne'],
        'n_e': kprof_params['n_e'],           # m^-3
        'psin_Te': kprof_params['psin_Te'],
        'T_e': kprof_params['T_e'],           # keV
        'units': {'ne': 'm^-3', 'Te': 'keV'},
    }


def state_to_dict(state):
    """Picklable snapshot of an ESCAPE state's attributes, for np.save.

    The state object itself can't be pickled: it holds the juliacall EPEDNN
    model. juliacall values also pickle, but only unpickle inside a live Julia
    session ("error deserializing this value"), which makes the whole
    .npy unreadable elsewhere, so those are left out too and their names
    recorded under '_not_saved'.
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


# ESCAPE state parameters
ne_x0 = None # m^-3, not needed for EPEDNN (defaults to the profile value)
quasineutral_flag = True
T_rat_flag = True # True if using a temperature ratio between ions and electrons, False if doing something else
T_rat = 1
pol_norm = False # True for when the poloidal flux is not normalized by 2pi. COCOS 7 convention is pol_norm=False, so poloidal flux is normalized by 2pi
species = 'D-T' # species of ions, currently supporting: D, D-T
regime_flag = 'PT H-mode' # regime of the plasma, currently supporting: 'PT H-mode', 'NT'
verbose_ESCAPE = False

# EPEDNN parameters
epednn_params = {
    'epednn_model': 'EPED1', # 'EPED1' or 'EPED_SPARC'
    'pres_gfile': False, # betan from the kinetic profiles, not the g-file pressure
}

Z_eff = 1.7 # https://lac913.epfl.ch/epsppd3/2018/pdf/P1.1077.pdf
Z_i = 1

# ESCAPE_state inputs, plus the same profiles in calc_pressure_profile form
# so out_dict['profiles'] and the input pedestal pressure match them.
equil_params = load_equilibrium(mhd_fp, mhd_loc)
profiles = kprof_to_profiles(kprof_params)
psi_N_p, p_Pa, p_mode = calc_pressure_profile(profiles)

# Initialize ESCAPE (ESCAPE_state.__init__ sets the geometry, a, R, kappa,
# delta, Ip, V_plasma and the kinetic profiles) and the EPEDNN model
state = ESCAPE(
    Z_eff = Z_eff,
    Z_i = Z_i, # Z of ions
    P_tot_e = P_tot_e, # W, not used by EPEDNN
    ne_x0 = ne_x0, # m^-3, not needed for EPEDNN (defaults to the profile value)
    equil_params = equil_params, # dictionary of required equilibrium parameters
    kprof_params = kprof_params, # dictionary of required kinetic profile (density, temperature) parameters
    quasineutral_flag = quasineutral_flag,
    T_rat_flag = T_rat_flag, # True if using a temperature ratio between ions and electrons, False if doing something else
    T_rat = T_rat,
    pol_norm = pol_norm, # True for when the poloidal flux is not normalized by 2pi. COCOS 7 convention is pol_norm=False, so poloidal flux is normalized by 2pi
    species = species, # species of ions, currently supporting: D, D-T
    regime_flag = regime_flag, # regime of the plasma, currently supporting: 'PT H-mode', 'NT'
    verbose = verbose_ESCAPE,
)
state.setup_epednn(model=epednn_params['epednn_model']) # also sets state.bt

# Input neped: n_e at the hardcoded density pedestal top psi_N = PSI_NE_PED
ped_wid_in = 1.0 - PSI_NE_PED # normalized poloidal flux
neped = float(np.interp(1.0 - ped_wid_in, state.psi_ne_eval, state.n_e)) # m^-3

print(f'ITER: Z_eff = {Z_eff:.3f}, pressure: {p_mode}, '
      f'neped at psi_N = {PSI_NE_PED}: neped = {neped:.3e} m^-3')

t0 = time.perf_counter()
ped_h, ped_wid, betan = state.feed_epednn(model=epednn_params['epednn_model'],
                                          ne_ped_h=neped,
                                          pres_gfile=epednn_params['pres_gfile'])
t1 = time.perf_counter()

# float() drops any juliacall wrapper. EPED1 returns MPa; convert to
# Pa to match the input p_Pa below
ped_wid = float(ped_wid) # normalized poloidal flux
ped_h_out = float(ped_h) * 1e6 # Pa
betan = float(np.ravel(betan)[0])

# Input-profile pedestal pressure at psi_N = 1 - ped_wid (EPEDNN width)
# and at the density pedestal top PSI_NE_PED that neped came from
p_at_ped_top = float(np.interp(1.0 - ped_wid, psi_N_p, p_Pa))
p_at_ped_top_in = float(np.interp(1.0 - ped_wid_in, psi_N_p, p_Pa))

# Save model output
out_dict = {
    'mhd_fp': str(mhd_fp),
    'prof_fp': str(prof_fp),
    'profiles': profiles,
    'neped': neped, # m^-3, EPEDNN input
    'psi_ne_ped': PSI_NE_PED, # hardcoded density pedestal top
    'input_ped_wid': ped_wid_in, # 1 - PSI_NE_PED
    'betan': betan, # EPEDNN input, from the kinetic profiles
    'epednn_inputs': {'a': float(state.a), 'betan': betan, 'bt': float(abs(np.ravel(state.bt)[0])),
                      'delta': float(state.delta), 'ip': float(abs(state.Ip)), 'kappa': float(state.kappa),
                      'm': float(state.M_eff), 'neped': neped / 1e19, 'r': float(state.Rmajor),
                      'zeffped': float(state.Z_eff)},
    'EPEDNN_ped_h': ped_h_out, # Pa
    'EPEDNN_ped_wid': ped_wid,
    'input_ped_h': p_at_ped_top, # Pa, input profile at psi_N = 1 - EPEDNN_ped_wid
    'input_ped_h_proxy': p_at_ped_top_in, # Pa, input profile at psi_N = PSI_NE_PED
    'epednn_model': epednn_params['epednn_model'],
    'p_mode': p_mode,
    'Zeff': Z_eff,
    'P_tot_e': P_tot_e,
    'EPEDNN_runtime': t1-t0,
    'state': state_to_dict(state), # picklable ESCAPE state attributes
}
np.save(Path(output_dir) / 'out_dict.npy', out_dict)

print(f'EPEDNN inputs: {out_dict["epednn_inputs"]}')
print(f'neped = {neped:.3e} m^-3, betan = {betan:.3f}: '
      f'p_ped = {ped_h_out/1e3:.2f} kPa, width = {ped_wid:.4f} '
      f'(input profile: {p_at_ped_top/1e3:.2f} kPa at 1 - EPEDNN width, '
      f'{p_at_ped_top_in/1e3:.2f} kPa at psi_N = {PSI_NE_PED})')
print(f'Saved to {Path(output_dir) / "out_dict.npy"}')
