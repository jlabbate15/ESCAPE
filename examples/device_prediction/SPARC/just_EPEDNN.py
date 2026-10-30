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
from helpers.ped_width_proxy import ped_width, ped_width_all
from examples.device_prediction.helper_functions import (calc_pressure_profile, read_sparcpublic_profiles,
                                                         psi_n_and_rho_psi, _print_profile_summary,
                                                         read_popcon, _to_watts)
import time


# ---------------------- COMMON USER INPUTS ----------------------
# Filepath to geqdsk equilibrium state and profiles
mhd_fp = HERE / 'gSPARC_DN_PRD.geqdsk'
TRANSP_PATH = HERE / '5 - transp_20221013.txt'
CGYRO_PATH = HERE / '6 - cgyro_20221013.txt'
PROFILE_SOURCE = os.environ.get('PROFILE_SOURCE', 'cgyro') # 'transp' or 'cgyro'
POPCON_PATH = HERE / '1 - PRD_POPCON_20221013.csv' # heating power and Zeff

# Pedestal top used to read neped off the input n_e profile, from helpers/ped_width_proxy.
# 'corner': the corner of the two-line fit. 'width': psi_N = 1 - width, with width from
# PED_WIDTH_METHOD (as on ESCAPE_solve's first iteration). SPARC's n_e flattens to a foot at
# psi_N ~ 0.99, inside the separatrix, so 1 - width (0.976) lands mid-gradient, not at the top
# (~0.965); fig_ESCAPEvsEPED.ipynb takes SPARC's experimental pedestal top at the corner too
PED_TOP_METHOD = os.environ.get('PED_TOP_METHOD', 'corner') # 'corner' or 'width'
PED_WIDTH_METHOD = os.environ.get('PED_WIDTH_METHOD', 'tanh+grad') # 'tanh', 'grad', 'lin', 'tanh+grad', ..., 'all'; for 'width'

run_suffix = os.environ.get('RUN_SUFFIX', '')

output_dir = HERE / (f'outputs/SPARC_EPEDNN_{PROFILE_SOURCE}' + (f'_{run_suffix}' if run_suffix else ''))
# ----------------------------------------------------------------


Path(output_dir).mkdir(parents=True, exist_ok=True)

# Equilibrium parameters
mhd_loc = 'eqdsk'

# Heating power and Zeff from the SPARCPublic POPCON table (as in scan_SCfp_ESCAPE.py)
popcon = read_popcon(POPCON_PATH, keys=('Ohmic power', 'RF power', 'Zeff'))
P_ohmic = _to_watts(popcon['Ohmic power']['value'], popcon['Ohmic power']['unit'])  # W
P_RF = _to_watts(popcon['RF power']['value'], popcon['RF power']['unit'])            # W
Z_eff = float(popcon['Zeff']['value'])  # dimensionless
# Saarelma et al. take ~half of the total (Ohmic+RF) heating as electron channel
P_tot_e = (P_ohmic + P_RF) / 2


# Starting profiles from the SPARCPublic TRANSP or CGYRO export (as in scan_SCfp_ESCAPE.py),
# on psi_N from the file's polflux
source = PROFILE_SOURCE.strip().lower()
if source == 'transp':
    prof_path = TRANSP_PATH
elif source == 'cgyro':
    prof_path = CGYRO_PATH
else:
    raise ValueError(f"PROFILE_SOURCE must be 'transp' or 'cgyro', got {PROFILE_SOURCE!r}")
profiles_in = read_sparcpublic_profiles(prof_path) # ne, ni [m^-3], Te, Ti [keV]
_print_profile_summary(source, prof_path, profiles_in)
psi_N, _ = psi_n_and_rho_psi(profiles_in)

# n_i / T_i are passed only when the file has them; otherwise ESCAPE_state takes
# n_i = n_e (quasineutral_flag) and T_i = T_rat * T_e (T_rat_flag)
kprof_params = {
    'n_e': profiles_in['ne'],   # m^-3
    'T_e': profiles_in['Te'],   # keV
    'psin_ne': psi_N,
    'psin_Te': psi_N,
}
if 'Ti' in profiles_in:
    kprof_params['T_i'] = profiles_in['Ti']   # keV
    kprof_params['psin_Ti'] = psi_N
if 'ni' in profiles_in:
    kprof_params['n_i'] = profiles_in['ni']   # m^-3
    kprof_params['psin_ni'] = psi_N


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
    With T_i it uses p = ne*Te + ni*Ti (ni = ne unless given), else p = 2*ne*Te."""
    profiles = {
        'psin_ne': kprof_params['psin_ne'],
        'n_e': kprof_params['n_e'],           # m^-3
        'psin_Te': kprof_params['psin_Te'],
        'T_e': kprof_params['T_e'],           # keV
        'units': {'ne': 'm^-3', 'Te': 'keV'},
    }
    if 'T_i' in kprof_params:
        profiles.update({'T_i': kprof_params['T_i'], 'psin_Ti': kprof_params['psin_Ti']})
    if 'n_i' in kprof_params:
        profiles.update({'n_i': kprof_params['n_i'], 'psin_ni': kprof_params['psin_ni']})
    return profiles


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
    'epednn_model': 'EPED_SPARC', # 'EPED1' or 'EPED_SPARC'
    'pres_gfile': False, # betan from the kinetic profiles, not the g-file pressure
}

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

# Input neped: n_e at the pedestal top psi_N = 1 - ped_wid_in (PED_TOP_METHOD)
if PED_TOP_METHOD == 'corner':
    res = ped_width_all(state.psi_ne_eval, state.n_e, method='lin')
    x0, L, sign, _ = res['norm']
    ped_wid_in = 1.0 - float(sign * (x0 + L * res['lin']['top'])) # 1 - two-line corner, normalized poloidal flux
elif PED_TOP_METHOD == 'width':
    ped_wid_in = float(ped_width(state.psi_ne_eval, state.n_e, method=PED_WIDTH_METHOD)) # normalized poloidal flux
else:
    raise ValueError(f"PED_TOP_METHOD must be 'corner' or 'width', got {PED_TOP_METHOD!r}")
neped = float(np.interp(1.0 - ped_wid_in, state.psi_ne_eval, state.n_e)) # m^-3

print(f'SPARC ({source}): Z_eff = {Z_eff:.3f}, pressure: {p_mode}, '
      f'input ped top = 1 - {ped_wid_in:.4f}, neped = {neped:.3e} m^-3')

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
# and at the pedestal top neped came from (psi_N = 1 - ped_wid_in)
p_at_ped_top = float(np.interp(1.0 - ped_wid, psi_N_p, p_Pa))
p_at_ped_top_in = float(np.interp(1.0 - ped_wid_in, psi_N_p, p_Pa))

# Save model output
out_dict = {
    'mhd_fp': str(mhd_fp),
    'prof_fp': str(prof_path),
    'profile_source': source,
    'profiles': profiles,
    'neped': neped, # m^-3, EPEDNN input
    'ped_top_method': PED_TOP_METHOD,
    'ped_width_method': PED_WIDTH_METHOD, # used only if ped_top_method is 'width'
    'input_ped_wid': ped_wid_in, # 1 - pedestal top that neped is read at
    'betan': betan, # EPEDNN input, from the kinetic profiles
    'epednn_inputs': {'a': float(state.a), 'betan': betan, 'bt': float(abs(np.ravel(state.bt)[0])),
                      'delta': float(state.delta), 'ip': float(abs(state.Ip)), 'kappa': float(state.kappa),
                      'm': float(state.M_eff), 'neped': neped / 1e19, 'r': float(state.Rmajor),
                      'zeffped': float(state.Z_eff)},
    'EPEDNN_ped_h': ped_h_out, # Pa
    'EPEDNN_ped_wid': ped_wid,
    'input_ped_h': p_at_ped_top, # Pa, input profile at psi_N = 1 - EPEDNN_ped_wid
    'input_ped_h_proxy': p_at_ped_top_in, # Pa, input profile at psi_N = 1 - input_ped_wid
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
      f'{p_at_ped_top_in/1e3:.2f} kPa at the input pedestal top)')
print(f'Saved to {Path(output_dir) / "out_dict.npy"}')
