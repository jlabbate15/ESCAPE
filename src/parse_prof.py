import numpy as np
from scipy.interpolate import interp1d

def OMFITnc_load(self, filename):
    """
    Reads an OMFITnc file, averages T_e and n_e across the first dimension,
    and interpolates them onto a common, unified psi_N grid.

    Parameters:
        filename (str): Path to the NetCDF file.

    Returns:
        tuple: (psi_N_unified, Te_1d, ne_1d) as 1D numpy arrays.
        Te_1d in keV, ne_1d in m^-3.

    # Example usage:
    # psi_grid, Te_profile, ne_profile = self.OMFITnc_load('my_plasma_data.cdf')
    """
    from omfit_classes.omfit_nc import OMFITnc
    nc = OMFITnc(filename)

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

    # 1. Extract and average the physics variables
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

    # 2. Determine psi_N grids (files may use psi_n / psi_N / separate Te,ne grids)
    if _nc_has('psi_N_Te') and _nc_has('psi_N_ne'):
        psi_Te = _reduce_time(_nc_array('psi_N_Te'))
        psi_ne = _reduce_time(_nc_array('psi_N_ne'))
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
        psi_Te = psi_shared
        psi_ne = psi_shared

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
    num_points = max(len(psi_Te), len(psi_ne))

    # 3. Unified evaluation grid on [0, 1]
    psi_N_unified = np.linspace(0.0, 1.0, num_points)

    # 4. Interpolate both profiles onto the unified grid
    Te_interp_func = interp1d(
        psi_Te, Te_raw, kind='cubic', bounds_error=False, fill_value='extrapolate'
    )
    ne_interp_func = interp1d(
        psi_ne, ne_raw, kind='cubic', bounds_error=False, fill_value='extrapolate'
    )

    Te_unified = Te_interp_func(psi_N_unified)
    ne_unified = ne_interp_func(psi_N_unified)

    return psi_N_unified, Te_unified, ne_unified



def kprof_load(self,kprof_loc='pfile',kprof_fp=None,T_prof=None,T_prof_psi_N=None,manual_profs=None):
    """Load kinetic equilibrium parameters using method specified by kprof_loc flag. 
    Parameters that will be loaded include: T_e, n_e
    Calculates: dn_e/dx|x=-inf, T_i
    
    Parameters
    ----------
    self : object
        instance of saarelma_connor class
    kprof_loc : string
        which method to use to load kinetic parameters.
    kprof_fp : string
        filepath to kprof_loc-type file with kinetic parameters.
    T_prof : string
        temperature profile if using the EPEDNN model
    T_prof_psi_N : array
        psi_N values at which T_e is evaluated if using the EPEDNN model
    """

    # currently self.T_e = self.T_e_pfile, fix when cleaning up code
    if kprof_loc == 'pfile' and manual_profs is None: # use true pfile for n_e and T_e

        def read_pfile(path):
            data = {}
            key = ''
            with open(path) as f:
                for line in f:
                    if '3 N Z A' in line:
                        break
                    if line.startswith('201'):
                        key = line.split()[2]
                        data[key] = np.array([])
                        data[f'{key}_psi'] = np.array([])
                    else:
                        psi, dat, _ = line.split()
                        psi = float(psi)
                        dat = float(dat)
                        data[key] = np.append(data[key], dat)
                        data[f'{key}_psi'] = np.append(data[f'{key}_psi'], psi)
            return data

        # Extract profiles
        pf = read_pfile(kprof_fp)

        # Store pfile information
        self.T_e_pfile = pf['te(KeV)'] # T_e values (keV) evaluated at psi_Te_eval
        self.T_e = self.T_e_pfile # keV
        self.psi_Te_eval_pfile = pf['te(KeV)_psi'] # psi_N values at which T_e is evaluated
        self.psi_Te_eval = self.psi_Te_eval_pfile # psi_N values at which T_e is evaluated

        self.n_e_pfile = pf['ne(10^20/m^3)'] * 1e20 # n_e values (10^20/m^3 -> m^-3) evaluated at psi_ne_eval
        self.psi_ne_eval = pf['ne(10^20/m^3)_psi'] # psi_N values at which n_e is evaluated

    elif kprof_loc == 'pfile' and manual_profs is not None: # use manual profiles for T_e but not n_e
        def read_pfile(path):
            data = {}
            key = ''
            with open(path) as f:
                for line in f:
                    if '3 N Z A' in line:
                        break
                    if line.startswith('201'):
                        key = line.split()[2]
                        data[key] = np.array([])
                        data[f'{key}_psi'] = np.array([])
                    else:
                        psi, dat, _ = line.split()
                        psi = float(psi)
                        dat = float(dat)
                        data[key] = np.append(data[key], dat)
                        data[f'{key}_psi'] = np.append(data[f'{key}_psi'], psi)
            return data

        # Extract profiles
        pf = read_pfile(kprof_fp)

        # Store pfile information
        self.n_e_pfile = pf['ne(10^20/m^3)'] * 1e20 # n_e values (10^20/m^3 -> m^-3) evaluated at psi_ne_eval
        self.psi_ne_eval = pf['ne(10^20/m^3)_psi'] # psi_N values at which n_e is evaluated

        # Store manual profiles information
        self.T_e_pfile = manual_profs['Te'] # keV
        self.T_e = self.T_e_pfile # keV
        self.psi_Te_eval_pfile = manual_profs['psi_N_Te'] # psi_N values at which T_e is evaluated
        self.psi_Te_eval = self.psi_Te_eval_pfile # psi_N values at which T_e is evaluated

    elif kprof_loc == 'OMFITnc':
        psi_N_unified, self.T_e_pfile, self.n_e_pfile = self.OMFITnc_load(kprof_fp)
        self.T_e = self.T_e_pfile # keV
        self.psi_Te_eval_pfile = psi_N_unified
        self.psi_Te_eval = self.psi_Te_eval_pfile # psi_N values at which T_e is evaluated
        self.psi_ne_eval = psi_N_unified

    elif kprof_loc in ('manual psi_N grid', 'manual rho grid'):
        # Profiles supplied directly by the caller, on their own radial grid.
        #   'manual psi_N grid'  the grid IS psi_N and is used as given
        #   'manual rho grid'    legacy grid, rho treated as sqrt(psi_N), so
        #                        psi_N = psi_N_pres[-1] * rho**2. Only for inputs
        #                        genuinely on a rho grid (e.g. the digitized ARC
        #                        profiles); anything derived from polflux should
        #                        pass psi_N and use 'manual psi_N grid'.
        def _psi_grid(psi_key, rho_key, fallback_psi=None, fallback_rho=None):
            """psi_N for one profile, from its psi_N_* key or its legacy rho_* key."""
            if manual_profs.get(psi_key) is not None:
                return np.asarray(manual_profs[psi_key], dtype=float)
            if manual_profs.get(rho_key) is not None:
                rho = np.asarray(manual_profs[rho_key], dtype=float)
                return self.psi_N_pres[-1] * rho**2
            if fallback_psi is not None:
                return fallback_psi
            if fallback_rho is not None:
                return fallback_rho
            raise KeyError(
                f"manual_profs needs '{psi_key}' (or the legacy '{rho_key}')"
            )

        # Specify full T_e and corresponding psi_N profile
        self.T_e_pfile = manual_profs['Te'] # T_e values (keV) evaluated at psi_Te_eval
        self.T_e = self.T_e_pfile # keV
        self.psi_Te_eval_pfile = _psi_grid('psi_N_Te', 'rho_Te') # psi_N values at which T_e is evaluated
        self.psi_Te_eval = self.psi_Te_eval_pfile # psi_N values at which T_e is evaluated

        # Specify n_e'(psi_N=0.85) and n_e(psi_N=1.0) boundary conditions
        # n_e_pfile is only used for the cross_sections and the boundary conditions
        self.n_e_pfile = manual_profs['ne'] * 1e20 # n_e values (10^20/m^3 -> m^-3) evaluated at psi_ne_eval
        self.psi_ne_eval = _psi_grid('psi_N_ne', 'rho_ne') # psi_N values at which n_e is evaluated

        # Optional ion profiles. If omitted, Ti = T_rat * Te and ni = ne.
        if 'Ti' in manual_profs and manual_profs['Ti'] is not None:
            psi_Ti = _psi_grid('psi_N_Ti', 'rho_Ti', fallback_psi=self.psi_Te_eval)
            Ti_keV = np.asarray(manual_profs['Ti'], dtype=float)
            self.T_i_pfile = Ti_keV
            self.psi_Ti_eval_pfile = psi_Ti
            self.psi_Ti_eval = psi_Ti
            # Working Ti on the Te psi_N grid (velocities / FSA use Te grid).
            self.T_i = interp1d(
                psi_Ti, Ti_keV, kind='linear',
                bounds_error=False, fill_value='extrapolate'
            )(self.psi_Te_eval)
            self.T_i_from_profile = True
        if 'ni' in manual_profs and manual_profs['ni'] is not None:
            self.n_i_pfile = np.asarray(manual_profs['ni'], dtype=float) * 1e20  # 10^20/m^3 -> m^-3
            self.psi_ni_eval = _psi_grid('psi_N_ni', 'rho_ni', fallback_psi=self.psi_ne_eval)

    elif kprof_loc == 'manual EPEDNN loop':
        assert manual_profs is not None, 'T and n_e, n_CX, n_FC profiles must be provided if T_e_source is epednn'
        self.T_e_pfile = manual_profs['Te'] # keV
        self.T_e = self.T_e_pfile # keV
        self.psi_Te_eval_pfile = manual_profs['psi_N_Te'] # psi_N values at which T_e is evaluated
        self.psi_Te_eval = self.psi_Te_eval_pfile # psi_N values at which T_e is evaluated
        self.n_e_pfile = manual_profs['ne'] # m^-3; n_e values evaluated at psi_ne_eval
        self.psi_ne_eval = manual_profs['psi_N_n'] # psi_N values at which n_e is evaluated
        self.nCX_manual = manual_profs['nCX'] # m^-3; evaluated at manual_profs['psi_N_n']
        self.nFC_manual = manual_profs['nFC'] # m^-3; evaluated at manual_profs['psi_N_n']
        self.psi_N_n_manual = manual_profs['psi_N_n'] # psi_N values at which n_e, n_CX, n_FC are evaluated

    elif kprof_loc == 'manual profs':
        assert manual_profs is not None, 'T and n_e, profiles must be provided'
        self.T_e_pfile = manual_profs['Te'] # keV
        self.T_e = self.T_e_pfile # keV
        self.psi_Te_eval_pfile = manual_profs['psi_N_Te'] # psi_N values at which T_e is evaluated
        self.psi_Te_eval = self.psi_Te_eval_pfile # psi_N values at which T_e is evaluated
        self.n_e_pfile = manual_profs['ne'] # m^-3; n_e values evaluated at psi_ne_eval
        self.psi_ne_eval = manual_profs['psi_N_ne'] # psi_N values at which n_e is evaluated

    else:
        assert False, 'kprof_loc method not supported'

    self.T_e_K = self.T_e * 1e3 * 11604.52 # T_e values (K) evaluated at psi_Te_eval

    # Ion temperature: profile if provided above, else T_rat * Te.
    if not getattr(self, 'T_i_from_profile', False):
        if self.T_rat_flag:
            self.T_i = self.T_e * self.T_rat # keV
            self.T_i_pfile = self.T_e_pfile * self.T_rat
            self.psi_Ti_eval = self.psi_Te_eval
            self.psi_Ti_eval_pfile = self.psi_Te_eval_pfile
        else:
            raise NotImplementedError("T_rat_flag must be True when Ti is not provided")
    self.T_i_K = self.T_i * 1e3 * 11604.52 # K

    # Ion density: profile if provided above, else ni = ne (quasineutrality).
    if not hasattr(self, 'n_i_pfile'):
        self.n_i_pfile = np.asarray(self.n_e_pfile, dtype=float).copy()
        self.psi_ni_eval = self.psi_ne_eval
