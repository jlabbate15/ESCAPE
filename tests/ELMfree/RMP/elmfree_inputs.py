"""Inputs for ESCAPE runs on one regime of the DIII-D ELM-free database.

Profiles:   sc_inputs/elmfree_prof/IDA_{shot}[_{tstart}_{tend}]_.cdf  (IDA, OMFITnc netCDF)
Equilibria: sc_inputs/gELM_freeDB/{shot}_{time_ms}/g{shot}.{...}       (EFIT g-files)
Labels:     sc_inputs/metadb.csv  (one row per shot time window: regime, label, h98, wmhd, ...)

Matching
--------
cdf -> metadb row : the filename window equals the row's (tstart, tend) [s].
                    Files without a window (IDA_{shot}_.cdf), or whose window
                    is not in metadb, take the row whose window holds all of
                    the file's time slices.
cdf -> g-file     : the g-file directory time is the mean of the cdf's time
                    slices [ms], so pairs are matched on (shot, mean time).
A cdf is kept only if every metadb row it matches has the requested regime.

Run as a script to print the match summary and write the pairs to
rmp_pairs.csv next to this file.
"""
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

SC_INPUTS = Path('/mnt/homes_global/jal2351/software/sc_inputs')
METADB_FP = SC_INPUTS / 'metadb.csv'
KPROF_DIR = SC_INPUTS / 'elmfree_prof'
GEQDSK_DIR = SC_INPUTS / 'gELM_freeDB'

CDF_RE = re.compile(r'^IDA_(\d+)_(?:([0-9.]+)_([0-9.]+)_)?\.cdf$')
GDIR_RE = re.compile(r'^(\d+)_([0-9.]+)$')
WINDOW_ATOL = 1e-3  # s, filename window vs metadb (tstart, tend); a few names carry float32 noise
SLICE_TOL = 1.0     # ms, slack when checking that a metadb window holds every time slice
GTIME_TOL = 0.01    # ms, g-file directory time vs mean cdf time slice
META_COLS = ['regime', 'label', 'source', 'h98', 'betan', 'wmhd', 'n1rms', 'n2rms',
             'tritop', 'tribot', 'kappa', 'q95']


# ---------------------------------------------------------------------------
# IDA netCDF reader
# ---------------------------------------------------------------------------
def read_ida_cdf(kprof_fp, time_average=True):
    """Read an IDA profile netCDF (OMFITnc output) from elmfree_prof.

    Every (time, psi_n) variable is returned. Files differ in which ones they
    carry: all have n_e, T_e, q, rho_tor, ...; most also have Zeff, the carbon
    channels (T_12C6, n_12C6), p_e and p_ion. Temperatures stored in eV are
    converted to keV.

    Parameters
    ----------
    kprof_fp : path-like
    time_average : bool
        True: average each profile over the time slices (NaN-aware), giving
        1-D arrays on psi_n. False: keep the (time, psi_n) arrays.

    Returns
    -------
    dict
        'psi_n' : normalized poloidal flux grid (0 to 1.2, so it includes the SOL)
        'time'  : ms, time slices
        'units' : {variable: unit}
        plus one entry per profile variable, e.g. 'n_e' [m^-3], 'T_e' [keV], 'Zeff' [-]
    """
    out = {'units': {}}
    with nc.Dataset(kprof_fp) as f:
        out['psi_n'] = np.asarray(f['psi_n'][:], dtype=float)
        out['time'] = np.asarray(f['time'][:], dtype=float)
        for name, var in f.variables.items():
            if var.dimensions != ('time', 'psi_n'):
                continue
            data = np.ma.filled(var[:].astype(float), np.nan)
            unit = getattr(var, 'unit', '')
            if unit == 'eV':
                data, unit = data / 1e3, 'keV'
            if time_average:
                with warnings.catch_warnings():  # all-NaN columns stay NaN
                    warnings.simplefilter('ignore', RuntimeWarning)
                    data = np.nanmean(data, axis=0)
            out[name] = data
            out['units'][name] = unit
    return out


def cdf_to_kprof(cdf):
    """kprof_params (helpers/parse_prof.kprof_load layout) from read_ida_cdf output.

    Restricted to closed flux, psi_N in [0, 1], on the file's own grid: IDA
    packs it toward the separatrix (~1e-3 spacing in the pedestal), which a
    uniform resampling would throw away. The grid runs past the separatrix,
    so the psi_N = 1 point is interpolated rather than extrapolated.
    """
    psi = cdf['psi_n']
    inside = psi < 1.0
    psi_N = np.append(psi[inside], 1.0)
    kprof = {}
    for key, psin_key in (('T_e', 'psin_Te'), ('n_e', 'psin_ne')):
        prof = np.append(cdf[key][inside], np.interp(1.0, psi, cdf[key]))
        good = np.isfinite(prof)
        kprof[key] = prof[good]               # keV / m^-3
        kprof[psin_key] = psi_N[good].copy()
    return kprof


# ---------------------------------------------------------------------------
# Database sorting
# ---------------------------------------------------------------------------
def find_pairs(regime='RMP', metadb_fp=METADB_FP, kprof_dir=KPROF_DIR,
               geqdsk_dir=GEQDSK_DIR, verbose=True):
    """Match profile cdfs to metadb rows and g-files, keeping one regime.

    Returns
    -------
    pandas.DataFrame
        One row per (g-file, cdf) pair, sorted by shot and time, with columns
        tag (g-file name without the leading 'g'), mhd_fp, kprof_fp, shot,
        tstart / tend [s] (metadb window), time [ms] (mean cdf time slice =
        g-file time), n_slices, and the metadb columns in META_COLS.
    """
    md = pd.read_csv(metadb_fp)
    n_windows = len(md[md.regime == regime].drop_duplicates(['shot', 'tstart', 'tend']))
    shots = set(md.loc[md.regime == regime, 'shot'])
    md = md[md.shot.isin(shots)]

    g_by_shot = defaultdict(list)
    for d in Path(geqdsk_dir).iterdir():
        m = GDIR_RE.match(d.name)
        if m is None or int(m.group(1)) not in shots:
            continue
        gfiles = sorted(d.glob('g*'))
        if len(gfiles) == 1:  # a few directories are empty
            g_by_shot[int(m.group(1))].append((float(m.group(2)), gfiles[0]))

    rows, skipped = [], Counter()
    for fp in sorted(Path(kprof_dir).glob('IDA_*.cdf')):
        m = CDF_RE.match(fp.name)
        if m is None or int(m.group(1)) not in shots:
            continue
        shot = int(m.group(1))
        with nc.Dataset(fp) as f:
            t = np.asarray(f['time'][:], dtype=float)  # ms

        shot_rows = md[md.shot == shot]
        cand = shot_rows.iloc[0:0]
        if m.group(2) is not None:
            t0, t1 = float(m.group(2)), float(m.group(3))
            cand = shot_rows[np.isclose(shot_rows.tstart, t0, atol=WINDOW_ATOL)
                             & np.isclose(shot_rows.tend, t1, atol=WINDOW_ATOL)]
        if cand.empty:  # no window in the name, or one metadb does not list exactly
            cand = shot_rows[(shot_rows.tstart * 1e3 <= t.min() + SLICE_TOL)
                             & (shot_rows.tend * 1e3 >= t.max() - SLICE_TOL)]
        regimes = set(cand.regime)
        if regime not in regimes:  # no window, or another regime of the same shot
            skipped['no metadb window' if not regimes else 'other regime'] += 1
            continue
        if len(regimes) > 1:
            skipped['ambiguous regime'] += 1
            if verbose:
                print(f'  {fp.name}: metadb windows disagree on regime {sorted(regimes)}, skipped')
            continue

        t_mean = t.mean()
        g_match = [g for t_g, g in g_by_shot[shot] if abs(t_g - t_mean) < GTIME_TOL]
        if len(g_match) != 1:
            skipped[f'{len(g_match)} g-files'] += 1
            if verbose:
                print(f'  {fp.name}: {len(g_match)} g-files at t = {t_mean:.4f} ms, skipped')
            continue

        meta = cand.iloc[0]  # duplicate metadb rows for one window share the regime
        rows.append({
            'tag': g_match[0].name[1:],
            'mhd_fp': str(g_match[0]),
            'kprof_fp': str(fp),
            'shot': shot,
            'tstart': meta.tstart,
            'tend': meta.tend,
            'time': t_mean,
            'n_slices': len(t),
            **meta[META_COLS].to_dict(),
        })

    pairs = pd.DataFrame(rows).sort_values(['shot', 'time'], ignore_index=True)
    if verbose:
        n_matched = len(pairs.drop_duplicates(['shot', 'tstart', 'tend']))
        print(f'{regime}: {n_windows} metadb windows, {len(pairs)} g/cdf pairs over '
              f'{pairs.shot.nunique()} shots ({n_windows - n_matched} windows have no profile file)')
        for reason, n in sorted(skipped.items()):
            print(f'  cdfs of {regime} shots skipped, {reason}: {n}')
    return pairs


def select_pairs(pairs, equil_num=None, select_equil=None):
    """Choose which pairs to run.

    Parameters
    ----------
    equil_num : int or None
        Number of pairs (None for all). One time per shot is taken before a
        second time from any shot, as in helpers/load_equil.initialize_inputs.
    select_equil : sequence of str or None
        Tags (g-file name, leading 'g' optional) returned in the order given;
        overrides equil_num. An unknown tag raises ValueError.
    """
    if select_equil is not None:
        tags = [s[1:] if s.startswith('g') else s for s in select_equil]
        missing = sorted(set(tags) - set(pairs.tag))
        if missing:
            raise ValueError(f'Equilibria not among the matched pairs: {missing}')
        return pairs.set_index('tag').loc[tags].reset_index()
    order = pairs.assign(_k=pairs.groupby('shot').cumcount())
    order = order.sort_values(['_k', 'shot', 'time']).drop(columns='_k')
    return order.head(equil_num).reset_index(drop=True) if equil_num else order.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Heating power
# ---------------------------------------------------------------------------
def chord_avg_ne(equil_params, psi_N, n_e, n_chord=2000):
    """n_e [m^-3] averaged along the horizontal chord through the magnetic axis,
    inside the LCFS (a stand-in for the interferometer line average)."""
    g = equil_params
    R = g['rleft'] + np.linspace(0.0, g['rdim'], g['nr'])
    Z = g['zmid'] + np.linspace(-0.5 * g['zdim'], 0.5 * g['zdim'], g['nz'])
    psiN = (g['psirz'] - g['psimag']) / (g['psibry'] - g['psimag'])  # psirz is (nz, nr)
    Rc = np.linspace(R[0], R[-1], n_chord)
    psi_c = RectBivariateSpline(Z, R, psiN)(g['zaxis'], Rc).ravel()
    # contiguous closed-flux segment around the axis
    ia = np.argmin(np.abs(Rc - g['raxis']))
    outside = np.flatnonzero(psi_c > 1.0)
    lo = outside[outside < ia].max() + 1 if np.any(outside < ia) else 0
    hi = outside[outside > ia].min() if np.any(outside > ia) else n_chord
    return float(np.mean(np.interp(psi_c[lo:hi], psi_N, n_e)))


def ipb98_heating_power(equil_params, kprof_params, W, H98=1.0, M=2.0):
    """Heating power [W] that IPB98(y,2) needs to hold stored energy W [J] at H98.

    W = H98 * 0.0562 Ip^0.93 B^0.15 P^0.69 n^0.41 M^0.19 R^1.97 eps^0.58 kappa_a^0.78
    (MJ, MA, T, MW, 10^19 m^-3, amu, m), inverted for P. Ip, B and the boundary
    shape come from the g-file, n from chord_avg_ne.
    """
    g = equil_params
    rb, zb = g['rzout'][:, 0], g['rzout'][:, 1]
    R = 0.5 * (rb.max() + rb.min())  # m, geometric major radius
    a = 0.5 * (rb.max() - rb.min())  # m, minor radius
    area = 0.5 * abs(np.dot(rb, np.roll(zb, 1)) - np.dot(zb, np.roll(rb, 1)))
    kappa_a = area / (np.pi * a**2)  # ~ V / (2 pi^2 R a^2)
    Ip = abs(g['ip']) / 1e6  # MA
    B = abs(g['bcentr']) * g['rcentr'] / R  # T, vacuum toroidal field at R
    n_bar = chord_avg_ne(g, kprof_params['psin_ne'], kprof_params['n_e']) / 1e19
    W98_per_P = (0.0562 * Ip**0.93 * B**0.15 * n_bar**0.41 * M**0.19
                 * R**1.97 * (a / R)**0.58 * kappa_a**0.78)
    return (W / 1e6 / (H98 * W98_per_P))**(1 / 0.69) * 1e6


if __name__ == '__main__':
    pairs = find_pairs('RMP')
    print(pairs.label.value_counts().to_string())
    out_fp = Path(__file__).resolve().parent / 'rmp_pairs.csv'
    pairs.to_csv(out_fp, index=False)
    print(f'Wrote {out_fp}')
