import os
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import pickle # to check which state attributes np.save can store
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] # ESCAPE/
sys.path.insert(0, str(ROOT))
from src.ESCAPE.ESCAPE_api import ESCAPE # ESCAPE_state + epednn_class (+ saarelma_connor, unused here)
from helpers.parse_equil import mhd_load
from helpers.ped_width_proxy import ped_width_all
from examples.device_prediction.helper_functions import calc_pressure_profile
from elmfree_inputs import find_pairs, select_pairs, read_ida_cdf, cdf_to_kprof, ipb98_heating_power
import time


# ---------------------- COMMON USER INPUTS ----------------------
equil_num = None # None for all
# equil_list = ['158103.03796_610'] # g-file names (leading 'g' optional); overrides equil_num
equil_list = None

# Main-ion species of the RMP-database rows ('label' column, source RMP) are
# D, H or He. ESCAPE only models D (and D-T), so H and He plasmas are left out.
EXCLUDE_LABELS = ('H', 'He')

# Pedestal top used to read neped off the input n_e profile: psi_N = 1 - width,
# with width from helpers/ped_width_proxy run on the n_e or the T_e profile,
# whichever has the more defined pedestal (see pick_ped_width)
PED_WIDTH_METHOD = os.environ.get('PED_WIDTH_METHOD', 'tanh+grad') # 'tanh', 'grad', 'lin', 'tanh+grad', ..., 'all'

run_suffix = os.environ.get('RUN_SUFFIX', '')

output_dir = HERE / ('outputs/DIIIDRMP_EPEDNN' + (f'_{run_suffix}' if run_suffix else ''))
# ----------------------------------------------------------------


Path(output_dir).mkdir(parents=True, exist_ok=True)

# Equilibria: RMP windows of sc_inputs/metadb.csv, profiles from
# sc_inputs/elmfree_prof (IDA netCDF), g-files from sc_inputs/gELM_freeDB.
# See elmfree_inputs.py for how the files are matched.
pairs = find_pairs('RMP')
excluded = pairs.label.isin(EXCLUDE_LABELS)
print(f'Excluding {excluded.sum()} pairs labelled {EXCLUDE_LABELS}')
pairs = pairs[~excluded]
if equil_num is None:
    equil_num = len(pairs)
print(f'Number of equilibria found: {len(pairs)}')
equilibria = select_pairs(pairs, equil_num, select_equil=equil_list)
print(f'Number of equilibria to process: {len(equilibria)}')


# ── Heating power ────────────────────────────────────────────────────────────
# Not used by EPEDNN, but ESCAPE_state needs a value. Same per-shot IPB98(y,2)
# estimate as scan_SCfp_ESCAPE_RMP.py.
P_HEAT_TOT = None       # W, total (NBI + ECH + Ohmic) heating; None: per-shot IPB98(y,2) estimate
H98_VALID = (0.1, 5.0)  # metadb H98 outside this range is treated as missing
ELECTRON_FRAC = 0.5     # Saarelma et al. take ~half the heating as the electron channel

# Fallback Zeff for files that carry no Zeff variable (2 of the RMP files).
# 2.7 is the median of Zeff(psi_N = 0.95) over the 356 D-plasma RMP files that have it.
ZEFF_DEFAULT = 2.7
ZEFF_PSI_N = 0.95       # evaluate Zeff at the pedestal top, not on axis


def zeff_at(cdf, name, psi_eval=ZEFF_PSI_N, default=ZEFF_DEFAULT):
    """Zeff at psi_N = psi_eval from time-averaged read_ida_cdf output.
    Returns `default` if the file has no usable Zeff."""
    if 'Zeff' not in cdf:
        print(f'  Zeff missing in {name}, using {default}')
        return float(default)
    good = np.isfinite(cdf['Zeff'])
    if not good.any():
        print(f'  Zeff all-NaN in {name}, using {default}')
        return float(default)
    return float(np.interp(psi_eval, cdf['psi_n'][good], cdf['Zeff'][good]))


def load_equilibrium(mhd_fp, kprof_fp):
    """Build the ESCAPE_state inputs for one (g-file, IDA netCDF) pair.

    Returns
    -------
    equil_params : dict
        read_eqdsk dictionary of the g-file (psirz, fpol, pres, ip, rzout, ...),
        which is what ESCAPE_state.set_equil_params consumes.
    kprof_params : dict
        T_e [keV] / n_e [m^-3] on their psi_N grids. Main-ion channels are not
        measured (CER gives carbon), so n_i = n_e is set explicitly here
        (quasi-neutrality) and T_i is left out so that ESCAPE_state takes
        T_i = T_rat * T_e.
    cdf : dict
        Time-averaged read_ida_cdf output, for Zeff.
    """
    equil_params = mhd_load(None, 'eqdsk', mhd_fp)
    cdf = read_ida_cdf(kprof_fp)
    kprof_params = cdf_to_kprof(cdf)
    kprof_params['n_i'] = kprof_params['n_e'].copy()          # m^-3
    kprof_params['psin_ni'] = kprof_params['psin_ne'].copy()
    return equil_params, kprof_params, cdf


def kprof_to_profiles(kprof_params):
    """kprof_params -> the layout calc_pressure_profile expects, so the
    input pedestal pressure comes from the same profiles EPEDNN is fed.
    With no T_i/n_i keys, calc_pressure_profile uses p = 2*ne*Te."""
    return {
        'psin_ne': kprof_params['psin_ne'],
        'n_e': kprof_params['n_e'],           # m^-3
        'psin_Te': kprof_params['psin_Te'],
        'T_e': kprof_params['T_e'],           # keV
        'units': {'ne': 'm^-3', 'Te': 'keV'},
    }


def heating_power(equil_params, kprof_params, pair):
    """Total heating power [W] and how it was obtained (see P_HEAT_TOT above)."""
    if P_HEAT_TOT is not None:
        return float(P_HEAT_TOT), 'fixed'
    h98 = pair.h98
    if np.isfinite(h98) and H98_VALID[0] < h98 < H98_VALID[1]:
        return ipb98_heating_power(equil_params, kprof_params, pair.wmhd, h98), 'IPB98 at metadb H98'
    return ipb98_heating_power(equil_params, kprof_params, pair.wmhd, 1.0), 'IPB98 at H98=1'


def ped_contrast(psi_N, prof, width):
    """How well defined a pedestal of `width` is on `prof`: mean |d prof/d psi_N|
    over the pedestal [1 - width, 1] divided by that over the band of the same
    width just inside it [1 - 2*width, 1 - width]. ~1 means no gradient break
    at the pedestal top; NaN if either band holds fewer than 2 points."""
    grad = np.abs(np.gradient(prof, psi_N))
    ped = psi_N >= 1.0 - width
    core = (psi_N >= 1.0 - 2.0 * width) & ~ped
    if ped.sum() < 2 or core.sum() < 2 or grad[core].mean() == 0.0:
        return np.nan
    return float(grad[ped].mean() / grad[core].mean())


def pick_ped_width(kprof_params, method=PED_WIDTH_METHOD):
    """Pedestal width (psi_N) from the n_e or T_e profile, whichever has the
    larger ped_contrast (the more defined pedestal). A profile whose width
    fit fails, or whose contrast is NaN, is only used if the other one is
    unusable too.

    Returns
    -------
    width : float
        Selected width (normalized poloidal flux).
    source : str
        'n_e' or 'T_e', the profile the width came from.
    info : dict
        Per-profile width, contrast, methods used and error (if any).
    """
    info = {}
    for key, psin_key in (('n_e', 'psin_ne'), ('T_e', 'psin_Te')):
        psi_N, prof = kprof_params[psin_key], kprof_params[key]
        try:
            res = ped_width_all(psi_N, prof, method=method)
        except Exception as e:
            info[key] = {'width': np.nan, 'contrast': np.nan, 'error': str(e)}
            continue
        info[key] = {
            'width': float(res['width']),
            'width_tanh': float(res['width_tanh']),
            'width_grad': float(res['width_grad']),
            'width_lin': float(res['width_lin']),
            'methods_used': list(res['methods_used']),
            'contrast': ped_contrast(psi_N, prof, res['width']),
        }
    usable = [k for k in info if np.isfinite(info[k]['width'])]
    if not usable:
        raise RuntimeError(f"pedestal width failed on n_e ({info['n_e']['error']}) "
                           f"and T_e ({info['T_e']['error']})")
    source = max(usable, key=lambda k: np.nan_to_num(info[k]['contrast'], nan=-np.inf))
    return info[source]['width'], source, info


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

Z_i = 1

summary = []
for pair in equilibria.itertuples():
    mhd_fp, kprof_fp = pair.mhd_fp, pair.kprof_fp
    equil_tag = pair.tag
    out_path = Path(output_dir) / equil_tag
    out_path.mkdir(parents=True, exist_ok=True)

    meta = {
        'tag': equil_tag,
        'shot': pair.shot,
        'tstart': pair.tstart, # s, metadb window
        'tend': pair.tend,
        'time': pair.time, # ms, g-file time = mean of the profile time slices
        'regime': pair.regime,
        'label': pair.label,
        'h98': pair.h98,
        'wmhd': pair.wmhd, # J
    }

    try:
        # ESCAPE_state inputs, plus the same profiles in calc_pressure_profile form
        # so out_dict['profiles'] and the input pedestal pressure match them.
        equil_params, kprof_params, cdf = load_equilibrium(mhd_fp, kprof_fp)
        profiles = kprof_to_profiles(kprof_params)
        psi_N_p, p_Pa, p_mode = calc_pressure_profile(profiles)

        # Zeff is a profile in the IDA netCDF, so take it per equilibrium.
        Z_eff = zeff_at(cdf, Path(kprof_fp).name)  # dimensionless, at psi_N = ZEFF_PSI_N

        P_heat, P_heat_method = heating_power(equil_params, kprof_params, pair)
        P_tot_e = P_heat * ELECTRON_FRAC # W, not used by EPEDNN

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

        # Input neped: n_e at the pedestal top psi_N = 1 - width, with width from
        # the n_e or T_e profile (whichever has the more defined pedestal)
        ped_wid_in, ped_wid_src, ped_wid_info = pick_ped_width(kprof_params)
        neped = float(np.interp(1.0 - ped_wid_in, state.psi_ne_eval, state.n_e)) # m^-3

        print(f'{equil_tag}: Z_eff = {Z_eff:.3f}, pressure: {p_mode}, '
              f'input ped width = {ped_wid_in:.4f} from {ped_wid_src} '
              f'(contrast n_e {ped_wid_info["n_e"]["contrast"]:.2f}, T_e {ped_wid_info["T_e"]["contrast"]:.2f}), '
              f'neped = {neped:.3e} m^-3')

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

        epednn_inputs = {'a': float(state.a), 'betan': betan, 'bt': float(abs(np.ravel(state.bt)[0])),
                         'delta': float(state.delta), 'ip': float(abs(state.Ip)), 'kappa': float(state.kappa),
                         'm': float(state.M_eff), 'neped': neped / 1e19, 'r': float(state.Rmajor),
                         'zeffped': float(state.Z_eff)}

        # Save model output
        out_dict = {
            'mhd_fp': str(mhd_fp),
            'kprof_fp': str(kprof_fp),
            **meta,
            'profiles': profiles,
            'neped': neped, # m^-3, EPEDNN input
            'ped_width_method': PED_WIDTH_METHOD,
            'input_ped_wid': ped_wid_in, # width proxy
            'input_ped_wid_source': ped_wid_src, # 'n_e' or 'T_e'
            'input_ped_wid_info': ped_wid_info, # per-profile widths and contrasts
            'betan': betan, # EPEDNN input, from the kinetic profiles
            'epednn_inputs': epednn_inputs,
            'EPEDNN_ped_h': ped_h_out, # Pa
            'EPEDNN_ped_wid': ped_wid,
            'input_ped_h': p_at_ped_top, # Pa, input profile at psi_N = 1 - EPEDNN_ped_wid
            'input_ped_h_proxy': p_at_ped_top_in, # Pa, input profile at psi_N = 1 - input_ped_wid
            'epednn_model': epednn_params['epednn_model'],
            'p_mode': p_mode,
            'Zeff': Z_eff,
            'P_heat': P_heat, # W, total
            'P_heat_method': P_heat_method,
            'P_tot_e': P_tot_e,
            'EPEDNN_runtime': t1-t0,
            'state': state_to_dict(state), # picklable ESCAPE state attributes
        }
        np.save(out_path / 'out_dict.npy', out_dict)

        print(f'  neped = {neped:.3e} m^-3, betan = {betan:.3f}: '
              f'p_ped = {ped_h_out/1e3:.2f} kPa, width = {ped_wid:.4f} '
              f'(input profile: {p_at_ped_top/1e3:.2f} kPa at 1 - EPEDNN width, '
              f'{p_at_ped_top_in/1e3:.2f} kPa at 1 - proxy width)')

        summary.append({
            **meta,
            'neped': neped,
            'input_ped_wid': ped_wid_in,
            'input_ped_wid_source': ped_wid_src,
            'contrast_ne': ped_wid_info['n_e']['contrast'],
            'contrast_Te': ped_wid_info['T_e']['contrast'],
            'width_ne': ped_wid_info['n_e']['width'],
            'width_Te': ped_wid_info['T_e']['width'],
            **{f'in_{k}': v for k, v in epednn_inputs.items() if k != 'neped'},
            'EPEDNN_ped_h': ped_h_out,
            'EPEDNN_ped_wid': ped_wid,
            'input_ped_h': p_at_ped_top,
            'input_ped_h_proxy': p_at_ped_top_in,
            'failed': False,
        })
    except Exception as e:
        print(f'{equil_tag}: failed ({e}), skipped')
        np.save(out_path / 'failed.npy', {'failed': True, **meta, 'error': str(e)})
        summary.append({**meta, 'failed': True, 'error': str(e)})

summary = pd.DataFrame(summary)
summary.to_csv(Path(output_dir) / 'summary.csv', index=False)
n_ok = int((~summary.failed).sum())
print(f'{n_ok} of {len(summary)} equilibria succeeded')
if n_ok:
    print(f'Pedestal width source: {summary.loc[~summary.failed, "input_ped_wid_source"].value_counts().to_dict()}')
print(f'Saved to {output_dir}')
