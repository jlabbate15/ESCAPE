# Run only EPEDNN

import sys
import csv
from collections import defaultdict
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(ROOT))
from src.solver_nondim import saarelma_connor_nondim
from ESCAPE.helpers.load_equil import initialize_inputs

out_dir = 'EPEDNN_pure_output_v3/'

gfile_fps = '/mnt/homes_global/jal2351/software/sc_inputs/gHighPerfHMode/'
kprof_fps = '/mnt/homes_global/jal2351/software/sc_inputs/OMFITnc_HighPerfHMode/'

# Fallback removed: equilibria without a fit ped-top are skipped (see loop below).
# Pedestal-top psiN from run_fit_pedestals.py (p_tot, gradient method).
# EPEDNN neped / zeffped must be taken here — NOT at a fixed psiN=0.95, and
# NOT with Zeff=1 (IDA ped Zeff is typically ~1.5-1.9).
PED_TOP_CSV = (
    '/mnt/homes_global/jal2351/software/sc_inputs/'
    'pedestal_tops_summary_HighPerfHMode.csv'
)

# Oak's per-profile pedestal fits. Needed because Oak_pedestal_fits.npy stores
# each profile's pedestal value only at THAT profile's own gradient knee
# (ped_eval_psin = grad_sym - grad_width/2), and the n_e knee is not the p_tot
# knee: over this set the n_e knee sits outboard of the p_tot knee in 94% of
# cases (median +0.012 in psi_N, up to +0.20). Re-evaluating Oak's fits at the
# common ped top keeps neped, zeffped and psiN_ped_exp on one flux surface.
OAK_FIT_CSV = (
    '/mnt/homes_global/jal2351/software/sc_inputs/'
    'pedestal_fits_snyder_short_Oak.csv'
)

epednn_model = 'EPED1'

ne_ped_source = 'Oak'

# Where the Oak density is evaluated:
#   'ped_top'      -> psiN_ped, the p_tot gradient knee used for zeffped and
#                     psiN_ped_exp (consistent; all pedestal-top quantities on
#                     the same psi_N)
#   'oak_ne_knee'  -> the n_e profile's own knee, i.e. Oak_pedestal_fits.npy
#                     ['n_e'] verbatim (legacy behaviour)
ne_ped_psin = 'ped_top'


def tnh0(c, x):
    """Modified tanh (Burrell/Groebner), same form as sc_inputs/fit_pedestals.py.

    c = [symmetry point, full width, height, offset, alpha]
    """
    c = np.asarray(c, dtype=float)
    z = 2.0 * (c[0] - x) / c[1]
    pz = 1.0 + c[4] * z
    mth = 0.5 * ((pz + 1.0) * np.tanh(z) + pz - 1.0)
    return 0.5 * ((c[2] - c[3]) * mth + c[2] + c[3])


def linfun(c, x):
    """Two-line corner fit, same form as sc_inputs/fit_pedestals.py.

    c = [pedestal top psiN, pedestal top value, inner slope, outer slope]
    """
    c = np.asarray(c, dtype=float)
    y = np.where(x <= c[0], c[2] * (c[0] - x) + c[1], c[3] * (x - c[0]) + c[1])
    return np.where(y < 0.0, 0.0, y)


def _f(row, key):
    """float(row[key]) with blanks/nan mapped to nan."""
    v = (row.get(key) or '').strip()
    if not v or v.lower() == 'nan':
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def load_oak_ne_fits(csv_path=OAK_FIT_CSV):
    """shot -> list of n_e fit records from Oak's per-profile fit CSV.

    Each record keeps the tanh and two-line parameters (so the fitted curves can
    be evaluated anywhere), the profile's own knee, and the stored ped_val_mean.
    """
    by_shot = defaultdict(list)
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f'Missing Oak fit CSV: {csv_path}')
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get('profile') != 'n_e':
                continue
            if row.get('bad') in ('1', 'True', 'true'):
                continue
            tanh_c = [_f(row, k) for k in
                      ('tanh_sym', 'tanh_width', 'tanh_height',
                       'tanh_offset', 'tanh_alpha')]
            lin_c = [_f(row, k) for k in
                     ('lin_top', 'lin_val', 'lin_slope_in', 'lin_slope_out')]
            by_shot[int(row['shot'])].append({
                'time_ms': _f(row, 'time'),
                'knee': _f(row, 'ped_eval_psin'),
                'ped_val_mean': _f(row, 'ped_val_mean'),
                # A method contributes only if it converged, exactly as
                # fit_pedestals.py decides which fits enter ped_val_mean.
                'tanh': tanh_c if (row.get('tanh_ok') == '1'
                                   and np.all(np.isfinite(tanh_c))) else None,
                'lin': lin_c if (row.get('lin_ok') == '1'
                                 and np.all(np.isfinite(lin_c))) else None,
                'grad_ok': row.get('grad_ok') == '1',
            })
    if not by_shot:
        raise RuntimeError(f'No usable n_e rows in {csv_path}')
    return by_shot


def oak_ne_fit_for_tag(equil_tag, oak_fits):
    """Oak n_e fit record for a g-file tag '125729.03589': shot, closest time."""
    shot_s, time_s = equil_tag.split('.', 1)
    rows = oak_fits.get(int(shot_s))
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r['time_ms'] - float(time_s)))


def oak_ne_at_psin(fit, psin, psi_prof, ne_prof):
    """Oak's n_e pedestal value at an arbitrary psi_N, in m^-3.

    Mean of the fitted curves that converged, each evaluated at `psin` — the
    same recipe fit_pedestals.py uses for ped_val_mean, but at a psi_N of our
    choosing. The gradient method has no closed form: it is the measured
    profile itself, so it enters as an interpolation of the IDA n_e.
    """
    vals = []
    if fit['tanh'] is not None:
        vals.append(float(tnh0(fit['tanh'], np.asarray([psin], dtype=float))[0]))
    if fit['grad_ok']:
        vals.append(float(np.interp(psin, psi_prof, ne_prof)))
    if fit['lin'] is not None:
        vals.append(float(linfun(fit['lin'], np.asarray([psin], dtype=float))[0]))
    return float(np.mean(vals)) if vals else np.nan


def load_ped_tops(csv_path=PED_TOP_CSV):
    """shot -> list of {time_ms, ped_top_grad, ped_top_1mw, grad_width}."""
    by_shot = defaultdict(list)
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f'Missing ped-top CSV: {csv_path}\n'
            f'Run sc_inputs/run_fit_pedestals.py first.'
        )
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get('bad') in ('1', 'True', 'true') or not row.get('ped_top_grad'):
                continue
            by_shot[int(row['shot'])].append({
                'time_ms': float(row['time_ms']),
                'ped_top_grad': float(row['ped_top_grad']),
                'ped_top_1mw': float(row['ped_top_1mw']),
                'grad_width': float(row['grad_width']),
            })
    if not by_shot:
        raise RuntimeError(f'No usable ped-top rows in {csv_path}')
    return by_shot


def ped_top_for_tag(equil_tag, ped_tops, prefer='ped_top_grad'):
    """Map g-file tag '125729.03589' -> nearest fit ped-top psiN, or None."""
    shot_s, time_s = equil_tag.split('.', 1)
    shot = int(shot_s)
    time_ms = float(time_s)  # '03589' -> 3589 ms
    rows = ped_tops.get(shot)
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r['time_ms'] - time_ms))[prefer]


def OMFITnc_load(filename):
    """
    Reads an OMFITnc file, averages kinetic profiles across the first dimension,
    and interpolates them onto a common, unified psi_N grid.

    Parameters:
        filename (str): Path to the NetCDF file.

    Returns:
        tuple: (psi_N_unified, Te_1d, ne_1d, p_ion_1d, Zeff_1d) as 1D numpy arrays.
        Te_1d in keV, ne_1d in m^-3, p_ion_1d in Pa, Zeff_1d dimensionless.

    Notes:
        IDA OMFITnc files typically lack T_i / n_i. Ion pressure is taken from
        p_ion when present (checked O(1) vs pe = ne*Te*e); otherwise pe is used.
    """
    from omfit_classes.omfit_nc import OMFITnc
    nc = OMFITnc(filename)
    e_charge = 1.602176634e-19  # C

    def _nc_array(var_name):
        """OMFITnc variables are SortedDicts; numeric data lives under 'data'."""
        var = nc[var_name]
        if hasattr(var, 'keys') and 'data' in var:
            return np.asarray(var['data'], dtype=float)
        return np.asarray(var, dtype=float)

    def _nc_has(var_name):
        return var_name in nc

    def _reduce_time(arr):
        """Average over a leading time axis if present."""
        arr = np.asarray(arr, dtype=float)
        if arr.ndim > 1:
            return np.mean(arr, axis=0)
        return arr

    # 1. Extract and average electron profiles
    Te_raw = _reduce_time(_nc_array('T_e'))
    ne_raw = _reduce_time(_nc_array('n_e'))

    # Convert Te to keV if the file stores eV (IDA OMFITnc files use eV)
    te_unit = ''
    if hasattr(nc['T_e'], 'keys') and 'unit' in nc['T_e']:
        te_unit = str(nc['T_e']['unit']).lower()
    if te_unit in ('ev', 'electron volt', 'electron-volt'):
        Te_raw = Te_raw / 1e3
    elif te_unit in ('kev',):
        pass
    elif np.nanmax(np.abs(Te_raw)) > 100:
        # Heuristic: values >> 100 are almost certainly eV, not keV
        Te_raw = Te_raw / 1e3

    # 2. Ion pressure from file if present; else fall back to pe after interpolation
    has_p_ion = _nc_has('p_ion')
    if has_p_ion:
        p_ion_raw = _reduce_time(_nc_array('p_ion'))  # Pa
    else:
        print(
            f"No p_ion found in {filename}. "
            f"Available keys: {[k for k in nc.keys() if not str(k).startswith('__')]}"
        )
        print('Using pe = ne*Te*e as p_ion')
        p_ion_raw = None

    # 3. Determine psi_N grids (files may use psi_n / psi_N / separate Te,ne grids)
    if _nc_has('psi_N_Te') and _nc_has('psi_N_ne'):
        psi_Te = _reduce_time(_nc_array('psi_N_Te'))
        psi_ne = _reduce_time(_nc_array('psi_N_ne'))
        psi_ion = (
            _reduce_time(_nc_array('psi_N_ion'))
            if _nc_has('psi_N_ion')
            else psi_ne
        )
    else:
        for psi_key in ('psi_n', 'psi_N', 'psiN', 'psin'):
            if _nc_has(psi_key):
                break
        else:
            raise KeyError(
                f"No psi_N grid found in {filename}. "
                f"Available keys: {[k for k in nc.keys() if not str(k).startswith('__')]}"
            )
        psi_shared = _reduce_time(_nc_array(psi_key))
        psi_Te = psi_ne = psi_ion = psi_shared

    # Keep only the closed-flux domain for interpolation onto [0, 1]
    def _clip_profile(psi, prof):
        psi = np.asarray(psi, dtype=float).ravel()
        prof = np.asarray(prof, dtype=float).ravel()
        order = np.argsort(psi)
        psi, prof = psi[order], prof[order]
        mask = (psi >= 0.0) & (psi <= 1.0)
        if np.count_nonzero(mask) < 2:
            mask = np.ones_like(psi, dtype=bool)
        # Drop duplicate psi points that break interp1d
        _, uniq = np.unique(psi[mask], return_index=True)
        uniq = np.sort(uniq)
        return psi[mask][uniq], prof[mask][uniq]

    psi_Te, Te_raw = _clip_profile(psi_Te, Te_raw)
    psi_ne, ne_raw = _clip_profile(psi_ne, ne_raw)
    if has_p_ion:
        psi_ion, p_ion_raw = _clip_profile(psi_ion, p_ion_raw)
        num_points = max(len(psi_Te), len(psi_ne), len(psi_ion))
    else:
        num_points = max(len(psi_Te), len(psi_ne))

    # 4. Unified evaluation grid on [0, 1]
    psi_N_unified = np.linspace(0.0, 1.0, num_points)

    # 5. Interpolate profiles onto the unified grid
    Te_unified = interp1d(
        psi_Te, Te_raw, kind='cubic', bounds_error=False, fill_value='extrapolate'
    )(psi_N_unified)
    ne_unified = interp1d(
        psi_ne, ne_raw, kind='cubic', bounds_error=False, fill_value='extrapolate'
    )(psi_N_unified)

    pe_unified = ne_unified * (Te_unified * 1e3) * e_charge  # Pa
    if has_p_ion:
        p_ion_unified = interp1d(
            psi_ion, p_ion_raw, kind='cubic', bounds_error=False, fill_value='extrapolate'
        )(psi_N_unified)
        # Sanity check: p_ion should be O(1) times electron pressure
        ratio_med = float(np.nanmedian(p_ion_unified / pe_unified))
        if not (0.3 < ratio_med < 5.0):
            raise ValueError(
                f"p_ion does not look like ion pressure in {filename}: "
                f"median(p_ion / (ne*Te*e)) = {ratio_med:.3g} (expected O(1))"
            )
    else:
        p_ion_unified = pe_unified

    # Zeff profile (needed for EPEDNN zeffped). Fall back to 1.0 if absent.
    if _nc_has('Zeff'):
        zeff_raw = _reduce_time(_nc_array('Zeff'))
        psi_z, zeff_raw = _clip_profile(psi_ne, zeff_raw)
        Zeff_unified = interp1d(
            psi_z, zeff_raw, kind='cubic', bounds_error=False, fill_value='extrapolate'
        )(psi_N_unified)
    else:
        Zeff_unified = np.ones_like(psi_N_unified)

    return psi_N_unified, Te_unified, ne_unified, p_ion_unified, Zeff_unified


ped_tops = load_ped_tops()
equil_num = len(list(Path(kprof_fps).glob('*.cdf')))
equilibria = initialize_inputs(equil_num, gfile_fps, kprof_fps, p_filetype='OMFITnc', select_equil="144515.04108")

Path(out_dir).mkdir(parents=True, exist_ok=True)

if ne_ped_source == 'Oak':
    if ne_ped_psin not in ('ped_top', 'oak_ne_knee'):
        raise ValueError(
            f"ne_ped_psin must be 'ped_top' or 'oak_ne_knee', got {ne_ped_psin!r}"
        )
    oak_ne_fits = load_oak_ne_fits()
    print(f'Loaded Oak n_e fits for {len(oak_ne_fits)} shots from {OAK_FIT_CSV}')
    print(f"neped evaluated at: {ne_ped_psin}")

i = 0
n_skip = 0
for mhd_fp, kprof_fp in equilibria:
    equil_tag = Path(mhd_fp).name[1:]
    psiN_ped = ped_top_for_tag(equil_tag, ped_tops, prefer='ped_top_grad')
    if psiN_ped is None or not np.isfinite(psiN_ped):
        print(f'  SKIP {equil_tag}: no fit ped-top in {PED_TOP_CSV}')
        n_skip += 1
        continue

    psi_N_unified, Te_unified, ne_unified, p_ion_unified, Zeff_unified = OMFITnc_load(
        kprof_fp
    )
    if ne_ped_source == 'John':
        ne_ped = np.interp(psiN_ped, psi_N_unified, ne_unified)  # m^-3
        ne_ped = ne_ped / 1e19  # m^-3 -> 10^19 m^-3 (EPEDNN input)
        psiN_neped = float(psiN_ped)
        ne_ped_oak_knee = np.nan
    elif ne_ped_source == 'Oak':
        # Match shot exactly; pick closest time (g-tag ms vs Oak float ms).
        # equil_tag: '125729.03589' ; Oak CSV shot/time: 125729 / 3589.5480
        fit = oak_ne_fit_for_tag(equil_tag, oak_ne_fits)
        if fit is None:
            print(f'  SKIP {equil_tag}: no Oak n_e fit for shot '
                  f'{equil_tag.split(".", 1)[0]}')
            n_skip += 1
            continue
        # Oak's stored value lives at the n_e profile's own knee, which is NOT
        # psiN_ped (the p_tot knee used for zeffped / psiN_ped_exp). Re-evaluate
        # the same fits at psiN_ped so every pedestal-top quantity fed to EPEDNN
        # refers to one flux surface.
        ne_ped_oak_knee = float(fit['ped_val_mean']) / 1e19  # 10^19 m^-3
        if ne_ped_psin == 'oak_ne_knee':
            psiN_neped = float(fit['knee'])
            ne_ped = ne_ped_oak_knee
        else:
            psiN_neped = float(psiN_ped)
            ne_ped = oak_ne_at_psin(fit, psiN_ped, psi_N_unified, ne_unified) / 1e19
        if not np.isfinite(ne_ped):
            print(f'  SKIP {equil_tag}: Oak ne_ped is nan')
            n_skip += 1
            continue
    zeff_ped = float(np.interp(psiN_ped, psi_N_unified, Zeff_unified))
    e_charge = 1.602176634e-19  # C
    # Total kinetic pressure profile [Pa]: p_ion + pe, both on psi_N_unified
    # pe = ne[m^-3] * Te[eV] * e; Te_unified is keV so *1e3
    kfile_pres = p_ion_unified + ne_unified * (Te_unified * 1e3) * e_charge

    # Run EPEDNN
    base_model = saarelma_connor_nondim(
        P_tot_e      = 5e6,  # W
        alpha_crit   = 1,
        C_KBM        = 1,
        De_chie_etg  = 1,
        nFC_x0       = 1e15,
        ncx_x0_ratio = 1,
        psi_N_inner_boundary = 0.85,
        mhd_fp       = mhd_fp,
        kprof_fp     = kprof_fp,
        kprof_loc    = 'OMFITnc',
        verbose      = False,
    )
    # feed_epednn maps self.Z_i -> zeffped. Override AFTER __init__ so ion
    # charge e_i (set from the constructor default Z_i=1 for D) stays correct.
    base_model.setup_epednn(model=epednn_model)
    pedestal_height, pedestal_width, betan = base_model.feed_epednn(
        model=epednn_model, ne_ped=ne_ped, EPEDNN_core='pfile', Z_eff=zeff_ped
    )
    np.save(
        Path(out_dir) / Path(equil_tag + '.npy'),
        {
            'pedestal_height': pedestal_height,
            'pedestal_width': pedestal_width,
            'betan': betan,
            'kfile_pres': kfile_pres,
            'psiN_kfilepres': psi_N_unified,
            'psiN_ped_exp': psiN_ped,
            'ne_ped_1e19': ne_ped,
            'psiN_neped': psiN_neped,             # where neped was evaluated
            'ne_ped_psin_mode': ne_ped_psin,
            'ne_ped_oak_knee_1e19': ne_ped_oak_knee,  # legacy value, for reference
            'zeff_ped': zeff_ped,
        },
    )
    i += 1
    if i % 25 == 0:
        print(f'  {i}/{len(equilibria)}  last={equil_tag}  '
              f'psiN_ped={psiN_ped:.4f}  zeff={zeff_ped:.3f}  '
              f'p_h={float(pedestal_height)*1e3:.2f} kPa')

print(f'Done: saved {i}/{len(equilibria)} equilibria to {out_dir} '
      f'(skipped {n_skip} without ped-top fit)')
