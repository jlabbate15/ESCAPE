"""Parse the DIII-D Dunsmore/Saarelma validation database into ESCAPE inputs.

The database (``Julio_db_for_Saarelma_input_downsampled.pkl``) is a dict keyed
``'{shot}_{time_ms}'``. Each entry holds fitted T_e [eV] / n_e [m^-3] profiles
on psi_N, scalar shape/plasma parameters (R_geo, r_minor, kappa, delta_*, q95,
ip [MA], bt [T]), heating powers [MW], Zeff, and LLAMA edge neutral
measurements (n0, ionization, emissivity on ``psi_edge``).

The database does not contain the 2D equilibrium (psi(R,Z), F(psi), boundary)
that ESCAPE_state needs; those come from the per-entry g-files in
``EQDSK_DIR`` (``sc_inputs/Jamie_inputs_twindow``), saved as
``{shot}.{time_ms}.npy`` pickles of OMFIT ``OMFITgeqdsk`` objects. ``load_gfile``
reads them without importing omfit_classes (which needs omas, tqdm, ...) and
returns the same dict as OpenFUSIONToolkit's ``read_eqdsk``, which is what
ESCAPE_state.set_equil_params consumes.

Each database entry averages over a 100 ms window [t_min, t_max]; EQDSK_DIR
holds g-files at both ends (t_min and t_min + 100 ms), which
``load_equilibrium`` time-averages into one equilibrium (``average_gfiles``).
If only one g-file of the pair exists, that one is used alone.
"""
import pickle
import re
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format
from matplotlib.path import Path as MplPath

DB_PATH = Path('/mnt/homes_global/jal2351/software/sc_inputs/Julio_db_for_Saarelma_input_downsampled.pkl')
EQDSK_DIR = Path('/mnt/homes_global/jal2351/software/sc_inputs/Jamie_inputs_twindow')


def load_db(db_path=DB_PATH):
    """Return the database dict {'{shot}_{time}': entry}."""
    with open(db_path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Equilibrium (pickled OMFITgeqdsk per entry; not in the database)
# ---------------------------------------------------------------------------
def find_gfiles(entry, eqdsk_dir=EQDSK_DIR):
    """Paths of the {shot}.{time_ms}.npy files in eqdsk_dir whose time lies in
    [t_min, t_max], sorted by time (normally the pair t_min, t_min + 100 ms;
    may be a single file). Empty list if there are none."""
    if eqdsk_dir is None:
        return []
    shot = int(entry['shot'])
    t_min, t_max = int(entry['t_min']), int(entry['t_max'])
    matches = []
    for fp in Path(eqdsk_dir).glob(f'{shot}.*.npy'):
        m = re.fullmatch(rf'{shot}\.(\d+)\.npy', fp.name)
        if m and t_min <= int(m.group(1)) <= t_max:
            matches.append((int(m.group(1)), fp))
    return [fp for _, fp in sorted(matches)]


class _OMFITStub(dict):
    """Stand-in for omfit_classes objects (OMFITgeqdsk, SortedDict,
    fluxSurfaces) so the pickles load without omfit_classes: their pickled
    state is a tuple whose dict members hold the g-file fields."""
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        for part in (state if isinstance(state, tuple) else (state,)):
            if isinstance(part, dict):
                self.update(part)


class _OMFITUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('omfit_classes'):
            return type(name, (_OMFITStub,), {})
        return super().find_class(module, name)


def load_gfile(fp):
    """Pickled OMFITgeqdsk (.npy) -> read_eqdsk-style dict (psi in Wb/rad as
    stored in the g-file, psirz with shape (nz, nr))."""
    with open(fp, 'rb') as f:
        version = npy_format.read_magic(f)
        npy_format._read_array_header(f, version)
        obj = _OMFITUnpickler(f).load()
    g = obj.item() if isinstance(obj, np.ndarray) else obj

    eq = {
        'case': str(g['CASE']),
        'nr': int(g['NW']), 'nz': int(g['NH']),
        'rdim': float(g['RDIM']), 'zdim': float(g['ZDIM']),
        'rcentr': float(g['RCENTR']), 'rleft': float(g['RLEFT']), 'zmid': float(g['ZMID']),
        'raxis': float(g['RMAXIS']), 'zaxis': float(g['ZMAXIS']),
        'psimag': float(g['SIMAG']), 'psibry': float(g['SIBRY']),
        'bcentr': float(g['BCENTR']), 'ip': float(g['CURRENT']),
        'fpol': np.asarray(g['FPOL'], dtype=float),
        'pres': np.asarray(g['PRES'], dtype=float),
        'ffprim': np.asarray(g['FFPRIM'], dtype=float),
        'pprime': np.asarray(g['PPRIME'], dtype=float),
        'psirz': np.asarray(g['PSIRZ'], dtype=float),
        'qpsi': np.asarray(g['QPSI'], dtype=float),
        'rzout': np.column_stack([g['RBBBS'], g['ZBBBS']]).astype(float),
        'rzlim': np.column_stack([g['RLIM'], g['ZLIM']]).astype(float),
    }
    if eq['psirz'].shape != (eq['nz'], eq['nr']):
        raise ValueError(f"{fp}: PSIRZ shape {eq['psirz'].shape} != (NH, NW) = ({eq['nz']}, {eq['nr']})")

    # The LCFS must lie inside the limiter. Some EFIT boundaries carry a
    # corrupt point (e.g. 184317.3250.npy has (0.539, -0.025) m, inside the
    # centre post), which would corrupt the R, a, kappa, delta ESCAPE takes
    # from rzout's extremes; drop such points.
    inside = MplPath(eq['rzlim']).contains_points(eq['rzout'])
    if not inside.all():
        print(f'  {Path(fp).name}: dropping {np.count_nonzero(~inside)} rzout point(s) outside the limiter: '
              f'{eq["rzout"][~inside].round(3).tolist()}')
        eq['rzout'] = eq['rzout'][inside]
    eq['nbbs'], eq['nlim'] = len(eq['rzout']), len(eq['rzlim'])
    return eq


def _resample_boundary(rz, n_seg):
    """Closed boundary -> 4*n_seg points: the four arcs between its
    outboard (max R), top (max Z), inboard (min R) and bottom (min Z)
    points, each resampled to n_seg points uniform in arc length. These
    extreme points (incl. the X-point) land on the same indices in every
    resampled boundary, so averaging boundaries pointwise keeps them sharp
    and the averaged R, a, kappa, delta are the averages of each file's."""
    if np.allclose(rz[0], rz[-1]):
        rz = rz[:-1]
    i0 = np.argmax(rz[:, 0])
    rz = np.roll(rz, -i0, axis=0)  # start at the outboard point
    if np.argmax(rz[:, 1]) > np.argmin(rz[:, 1]):
        rz = np.vstack([rz[:1], rz[:0:-1]])  # orient counter-clockwise: outboard -> top -> inboard -> bottom
    rz = np.vstack([rz, rz[:1]])
    knots = [0, np.argmax(rz[:, 1]), np.argmin(rz[:, 0]), np.argmin(rz[:, 1]), len(rz) - 1]
    arcs = []
    for a, b in zip(knots[:-1], knots[1:]):
        seg = rz[a:b + 1]
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(seg, axis=0).T))])
        su = np.linspace(0.0, s[-1], n_seg + 1)[:-1]  # next arc supplies the end point
        arcs.append(np.column_stack([np.interp(su, s, seg[:, 0]), np.interp(su, s, seg[:, 1])]))
    return np.vstack(arcs)


def average_gfiles(eqs):
    """Time-average read_eqdsk-style dicts from the same shot: every scalar
    and profile/psirz array is averaged. The boundaries (rzout) have
    different point counts,
    so each is resampled by arc length between its extreme points
    (_resample_boundary) before averaging. A single dict is returned
    unchanged."""
    if len(eqs) == 1:
        return eqs[0]
    ref = eqs[0]
    for e in eqs[1:]:
        for key in ('nr', 'nz', 'rdim', 'zdim', 'rleft', 'zmid', 'rcentr'):
            if not np.isclose(e[key], ref[key]):
                raise ValueError(f"cannot average g-files with different grids ({key}: {ref[key]} vs {e[key]})")
        for key in ('fpol', 'pres', 'ffprim', 'pprime', 'psirz', 'qpsi'):
            if e[key].shape != ref[key].shape:
                raise ValueError(f"cannot average g-files with different {key} shapes "
                                 f"({ref[key].shape} vs {e[key].shape})")

    eq = dict(ref)
    for key in ('raxis', 'zaxis', 'psimag', 'psibry', 'bcentr', 'ip'):
        eq[key] = float(np.mean([e[key] for e in eqs]))
    for key in ('fpol', 'pres', 'ffprim', 'pprime', 'psirz', 'qpsi'):
        eq[key] = np.mean([e[key] for e in eqs], axis=0)

    n_seg = -(-max(len(e['rzout']) for e in eqs) // 4)
    rzout = np.mean([_resample_boundary(e['rzout'], n_seg) for e in eqs], axis=0)
    eq['rzout'] = np.vstack([rzout, rzout[:1]])  # closed, like EFIT's RBBBS/ZBBBS
    if all(e['rzlim'].shape == ref['rzlim'].shape for e in eqs):
        eq['rzlim'] = np.mean([e['rzlim'] for e in eqs], axis=0)  # identical within a shot in practice
    eq['nbbs'], eq['nlim'] = len(eq['rzout']), len(eq['rzlim'])

    # case string carries the mean time, which gfile_scalars/check_gfile read
    times = [gfile_scalars(e)['time'] for e in eqs]
    if None not in times:
        eq['case'] = re.sub(r'\d+(\s*ms)', f'{round(np.mean(times))}\\1', ref['case'], count=1)
    return eq


# Allowed g-file vs database differences. Measured over all 55 entries:
# Ip <= 1.6%; |Bt| database/BCENTR = 1.021-1.033 (BCENTR is at RCENTR = 1.6955 m,
# the database bt evidently at another radius/signal); R_geo, r_minor <= 2 mm;
# kappa <= 0.005; delta_u/l <= 0.046 (boundary sampled at only ~90 points, so
# the top/bottom extreme is off by up to one point spacing); q95 <= 0.04.
CHECK_TOL = {
    'ip': ('rel', 0.03),
    'bt': ('rel', 0.05),
    'R_geo': ('abs', 0.01),  # m
    'r_minor': ('abs', 0.01),  # m
    'kappa': ('abs', 0.03),
    'delta_upper': ('abs', 0.06),
    'delta_lower': ('abs', 0.06),
    'q95': ('rel', 0.05),
}


def gfile_scalars(eq):
    """Database-comparable scalars from a read_eqdsk-style dict: shot and
    time from the EFIT case string, |Ip| [MA], |BCENTR| [T], boundary shape
    (same extreme-point definitions as ESCAPE_state.set_equil_params), q95."""
    b = eq['rzout']
    r_max, r_min = b[:, 0].max(), b[:, 0].min()
    R0, a = (r_max + r_min) / 2, (r_max - r_min) / 2
    top, bot = b[np.argmax(b[:, 1])], b[np.argmin(b[:, 1])]
    m_shot = re.search(r'#\s*(\d+)', eq['case'])
    m_time = re.search(r'(\d+)\s*ms', eq['case'])
    return {
        'shot': int(m_shot.group(1)) if m_shot else None,
        'time': int(m_time.group(1)) if m_time else None,  # ms
        'ip': abs(eq['ip']) / 1e6,
        'bt': abs(eq['bcentr']),
        'R_geo': R0,
        'r_minor': a,
        'kappa': (top[1] - bot[1]) / (2 * a),
        'delta_upper': (R0 - top[0]) / a,
        'delta_lower': (R0 - bot[0]) / a,
        'q95': abs(np.interp(0.95, np.linspace(0.0, 1.0, len(eq['qpsi'])), eq['qpsi'])),
    }


def check_gfile(eq, entry, tol=CHECK_TOL):
    """Compare a loaded g-file with its database entry. Returns a list of
    mismatch messages (empty if consistent)."""
    g = gfile_scalars(eq)
    issues = []
    if g['shot'] != int(entry['shot']):
        issues.append(f"shot {g['shot']} != database {int(entry['shot'])}")
    if g['time'] is None or not int(entry['t_min']) <= g['time'] <= int(entry['t_max']):
        issues.append(f"time {g['time']} ms outside database window {int(entry['t_min'])}-{int(entry['t_max'])} ms")
    for key, (kind, limit) in tol.items():
        ref = abs(float(entry[key]))
        diff = abs(g[key] - ref) / ref if kind == 'rel' else abs(g[key] - ref)
        if not diff <= limit:
            issues.append(f"{key}: g-file {g[key]:.4g} vs database {ref:.4g} "
                          f"({'rel' if kind == 'rel' else 'abs'} diff {diff:.3g} > {limit})")
    return issues


def load_equilibrium(entry, eqdsk_dir=EQDSK_DIR, strict=True):
    """Find, load, time-average and check the g-files for a database entry
    (the pair at t_min and t_min + 100 ms, or the single one present).
    Returns (list of paths, read_eqdsk-style dict). Raises FileNotFoundError
    if there is no g-file and ValueError if a g-file is from another shot
    or, when strict, the averaged equilibrium disagrees with the database."""
    fps = find_gfiles(entry, eqdsk_dir)
    if not fps:
        raise FileNotFoundError(f"no g-file for shot {int(entry['shot'])}, "
                                f"{int(entry['t_min'])}-{int(entry['t_max'])} ms in {eqdsk_dir}")
    eqs = [load_gfile(fp) for fp in fps]
    issues = []
    if len(eqs) > 1:
        # shot of each file (the average's case string is the first file's)
        for fp, e in zip(fps, eqs):
            g = gfile_scalars(e)
            if g['shot'] != int(entry['shot']):
                issues.append(f"{fp.name}: shot {g['shot']} != database {int(entry['shot'])}")
        if issues:
            raise ValueError('g-files disagree with the database: ' + '; '.join(issues))
    eq = average_gfiles(eqs)
    issues = check_gfile(eq, entry)
    names = '+'.join(fp.name for fp in fps)
    if issues:
        msg = f'{names} disagrees with the database: ' + '; '.join(issues)
        if strict:
            raise ValueError(msg)
        print(f'  WARNING: {msg}')
    return fps, eq


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------
def heating_power(entry, key='ptot'):
    """Heating power [W] from the database (stored in MW). 'ptot' is
    pinj + pech + pohm; 'pnet' additionally subtracts prad."""
    return float(entry[key]) * 1e6


# ---------------------------------------------------------------------------
# Kinetic profiles
# ---------------------------------------------------------------------------
def _on_unit_psi_grid(psi, prof):
    """Restrict a fitted profile to psi_N in [0, 1] on a uniform grid ending
    exactly at psi_N = 1 (ESCAPE takes n_e[-1] as the separatrix value). The
    fits extend to psi_N = 1.2, so this only interpolates inside the data."""
    psi = np.asarray(psi, dtype=float)
    prof = np.asarray(prof, dtype=float)
    good = np.isfinite(psi) & np.isfinite(prof)
    order = np.argsort(psi[good])
    psi, prof = psi[good][order], prof[good][order]
    if psi[0] > 0.0 or psi[-1] < 1.0:
        raise ValueError(f'fitted profile covers psi_N [{psi[0]:.3f}, {psi[-1]:.3f}], not [0, 1]')
    n = int(np.count_nonzero(psi <= 1.0))
    psi_u = np.linspace(0.0, 1.0, n)
    return psi_u, np.interp(psi_u, psi, prof)


def build_kprof(entry):
    """ESCAPE kprof_params from the fitted profiles: T_e [keV] and n_e [m^-3]
    on psi_N in [0, 1]. The database has no main-ion profiles; n_i and T_i are
    left out, so ESCAPE_state applies quasineutral_flag / T_rat_flag."""
    psin_Te, T_e = _on_unit_psi_grid(entry['te_fitted_psi'], entry['te_fitted_full'])
    psin_ne, n_e = _on_unit_psi_grid(entry['ne_fitted_psi'], entry['ne_fitted_full'])
    return {
        'T_e': T_e / 1e3,  # eV -> keV
        'psin_Te': psin_Te,
        'n_e': n_e,  # m^-3
        'psin_ne': psin_ne,
    }


def kprof_to_profiles(kprof_params):
    """kprof_params -> the layout calc_pressure_profile expects (p = 2*ne*Te)."""
    return {
        'psin_ne': kprof_params['psin_ne'],
        'n_e': kprof_params['n_e'],  # m^-3
        'psin_Te': kprof_params['psin_Te'],
        'T_e': kprof_params['T_e'],  # keV
        'units': {'ne': 'm^-3', 'Te': 'keV'},
    }


def experimental_data(entry):
    """Every database field for this entry, kept alongside the ESCAPE outputs
    for validation (raw/fitted profiles, LLAMA n0/ionization/emissivity,
    powers, ELM frequency, ...)."""
    return dict(entry)


if __name__ == '__main__':
    # Check every database entry has a g-file consistent with it
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eqdsk-dir', default=str(EQDSK_DIR), help='directory of {shot}.{time}.npy g-file pickles')
    args = parser.parse_args()

    db = load_db()
    print(f'{len(db)} entries in {DB_PATH.name}')
    bad = {}
    for tag, entry in db.items():
        kp = build_kprof(entry)
        z = float(entry['Zeff'])
        try:
            fps, eq = load_equilibrium(entry, args.eqdsk_dir)
            status = f"{'+'.join(fp.name for fp in fps)} OK"
            if len(fps) == 1:
                status += ' (no +100 ms partner, not averaged)'
        except (FileNotFoundError, ValueError) as e:
            bad[tag] = str(e)
            status = f'FAILED: {e}'
        zflag = f'  (Zeff={z:.2f} < 1)' if z < 1.0 else ''
        print(f"{tag}: T_e(sep)={kp['T_e'][-1]*1e3:.1f} eV, n_e(sep)={kp['n_e'][-1]:.3e} m^-3, "
              f"Zeff={z:.2f}, ptot={float(entry['ptot']):.2f} MW  {status}{zflag}")
    print(f'{len(db) - len(bad)}/{len(db)} entries have a consistent g-file')
