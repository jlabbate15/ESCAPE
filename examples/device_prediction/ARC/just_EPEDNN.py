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
from helpers.ped_width_proxy import ped_width
from examples.device_prediction.helper_functions import calc_pressure_profile
import time


# ---------------------- COMMON USER INPUTS ----------------------
# Filepath to geqdsk equilibrium state and profiles
mhd_fp = HERE / 'geqdsk-ARCv3a'
ne_fp = HERE / 'ARC_ne.csv'
Te_fp = HERE / 'ARC_Te.csv'

# Pedestal top used to read neped off the input n_e profile: psi_N = 1 - width,
# with width from helpers/ped_width_proxy (as on ESCAPE_solve's first iteration)
PED_WIDTH_METHOD = os.environ.get('PED_WIDTH_METHOD', 'tanh+grad') # 'tanh', 'grad', 'lin', 'tanh+grad', ..., 'all'

run_suffix = os.environ.get('RUN_SUFFIX', '')

output_dir = HERE / ('outputs/ARC_EPEDNN' + (f'_{run_suffix}' if run_suffix else ''))
# ----------------------------------------------------------------


Path(output_dir).mkdir(parents=True, exist_ok=True)

# Equilibrium parameters
mhd_loc = 'eqdsk'

# Heating power
P_tot_e = ( 21.5 + 0.8 + 227 ) * 0.5 * 1e6 # table 4 of Hillesheim et al. 2026


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


psi_N_ne, ne_ref = load_digitized_profile(ne_fp)   # 10^20 m^-3
psi_N_Te, Te_ref = load_digitized_profile(Te_fp)   # keV

print(f'Loaded ne: {len(psi_N_ne)} points, psi_N in [{psi_N_ne[0]:.4f}, {psi_N_ne[-1]:.4f}], '
      f'{ne_ref[0]:.2f} -> {ne_ref[-1]:.2f} x10^20 m^-3')
print(f'Loaded Te: {len(psi_N_Te)} points, psi_N in [{psi_N_Te[0]:.4f}, {psi_N_Te[-1]:.4f}], '
      f'{Te_ref[0]:.2f} -> {Te_ref[-1]:.2f} keV')

kprof_params = {
    'n_e': ne_ref * 1e20,   # 10^20 m^-3 -> m^-3
    'T_e': Te_ref,          # keV
    'psin_ne': psi_N_ne,
    'psin_Te': psi_N_Te,
}


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
species = 'D' # species of ions, currently supporting: D, D-T
regime_flag = 'PT H-mode' # regime of the plasma, currently supporting: 'PT H-mode', 'NT'
verbose_ESCAPE = False

# EPEDNN parameters
epednn_params = {
    'epednn_model': 'EPED1', # 'EPED1' or 'EPED_SPARC'
    'pres_gfile': False, # betan from the kinetic profiles, not the g-file pressure
}

Z_eff = 1.52 # https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/overview-of-the-physics-basis-for-the-arc-fusion-power-plant/B472B3A64EF71DA1899B9EFB65D7C390
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

# Input neped: n_e at the pedestal top psi_N = 1 - width (width proxy)
ped_wid_in = float(ped_width(state.psi_ne_eval, state.n_e, method=PED_WIDTH_METHOD)) # normalized poloidal flux
neped = float(np.interp(1.0 - ped_wid_in, state.psi_ne_eval, state.n_e)) # m^-3

print(f'ARC: Z_eff = {Z_eff:.3f}, pressure: {p_mode}, '
      f'input ped width = {ped_wid_in:.4f}, neped = {neped:.3e} m^-3')

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
# and at the width-proxy pedestal top neped came from
p_at_ped_top = float(np.interp(1.0 - ped_wid, psi_N_p, p_Pa))
p_at_ped_top_in = float(np.interp(1.0 - ped_wid_in, psi_N_p, p_Pa))

# Save model output
out_dict = {
    'mhd_fp': str(mhd_fp),
    'ne_fp': str(ne_fp),
    'Te_fp': str(Te_fp),
    'profiles': profiles,
    'neped': neped, # m^-3, EPEDNN input
    'ped_width_method': PED_WIDTH_METHOD,
    'input_ped_wid': ped_wid_in, # width proxy
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
      f'{p_at_ped_top_in/1e3:.2f} kPa at 1 - proxy width)')
print(f'Saved to {Path(output_dir) / "out_dict.npy"}')
