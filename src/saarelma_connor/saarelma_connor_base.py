import os
import inspect
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.integrate import simpson, solve_bvp, cumulative_trapezoid
from scipy import constants
import matplotlib.pyplot as plt
from matplotlib.path import Path as _MplPath
from src.adas.adas_ionisation import scd_adas
from src.adas.adas_cx import scx_adas
from src.radas.radas_rates import scd_radas, scx_radas
try:
    from firedrake import (
        IntervalMesh, FunctionSpace, MixedFunctionSpace, Function,
        TestFunctions, TrialFunction, Constant, DirichletBC,
        dx, ds, split, solve, lhs, rhs,
        LinearVariationalProblem, LinearVariationalSolver,
        SpatialCoordinate,
    )
    _FIREDRAKE_AVAILABLE = True
except Exception as _firedrake_import_err:  # raise an error if Firedrake is not available
    _FIREDRAKE_AVAILABLE = False
    _FIREDRAKE_IMPORT_ERR = _firedrake_import_err


# Free parameters of the model (all four must be set before solving).
FREE_PARAM_NAMES = ("alpha_crit", "De_chie_etg", "C_KBM", "nFC_x0")


class SaarelmaConnorBase:
    """

    Description
    ----------
    Creates instance of a tokamak pedestal configuration that the Saarelma-Connor (S. Saarelma et al 2024 Nucl. Fusion 64 076025) model can be applied to
    Dependencies:
    - juliacall (install with pip install juliacall) for EPEDNN interfacing
    - EPEDNN (install by git clone)
    - OpenFUSIONToolkit -> TokaMaker

    Uses COCOS 7 coordinate convention (same as TokaMaker) as defined by https://crppwww.epfl.ch/~sauter/cocos/Sauter_COCOS_Tokamak_Coordinate_Conventions.pdf

    Parameters
    ----------
    Z_i : int
        Z of ions
    P_tot_e : float
        Total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox, W
    ne_x0 : float
        m^-3, electron density at the separatrix (boundary condition)
    psi_N_inner_boundary : float
        psi_N to choose the inner boundary boundary condition. If None, inner boundary is chosen based on nFC_threshold and nCX_threshold.
    nFC_threshold : float or None
        Fraction of the separatrix FC neutral density (nFC_x0) below which the
        inner boundary is placed.  Default 0.01 (1 %).  Set to None to disable. Not used if psi_N_inner_boundary!=None
    nCX_threshold : float or None
        Fraction of the peak estimated CX neutral density below which the inner
        boundary is placed.  Default 0.01 (1 %).  Set to None to disable. Not used if psi_N_inner_boundary!=None
    equil_params : dict
        Dictionary of required equilibrium parameters
    kprof_params : dict
        Dictionary of required kinetic profile parameters
    T_rat_flag : bool
        True if the temperature ratio is given, False if the temperature ratio is to be calculated
    T_rat : float
        Temperature ratio between ions and electrons, dimensionless
        Ignored if T_rat_flag is False
        Default is 1
    pol_norm : bool
        True if the poloidal flux is normalized by 2pi, False if the poloidal flux is not normalized by 2pi
    species : string
        Species of ions, currently supporting: D, D-T
    initial_guess : string
        Initial guess for the electron density profile, currently supporting: pfile
    verbose : bool
        True if verbose output is desired, False if verbose output is not desired
    """
    def __init__(
        self,
        Z_i = 1, # Z of ions
        P_tot_e = None, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
        ne_x0 = None, # m^-3, electron density at the separatrix (boundary condition, default is to use from pfile)
        psi_N_inner_boundary = 0.85, # normalized poloidal flux at the inner boundary (boundary condition); overridden by find_inner_boundary if nFC_threshold or nCX_threshold is set
        nFC_threshold = None, # fraction of nFC at the separatrix below which the inner boundary is placed (None to disable)
        nCX_threshold = None, # fraction of nCX at the separatrix below which the inner boundary is placed (None to disable)
        equil_params = None, # dictionary of required equilibrium parameters
        kprof_params = None, # dictionary of required kinetic profile (density, temperature) parameters
        T_rat_flag = True, # True if using a temperature ratio between ions and electrons, False if doing something else
        T_rat = 1,
        pol_norm = False, # True for when the poloidal flux is not normalized by 2pi. COCOS 7 convention is pol_norm=False, so poloidal flux is normalized by 2pi
        species = 'D', # species of ions, currently supporting: D, D-T
        initial_guess = 'pfile', # initial guess for the electron density profile, currently supporting: pfile, linear
        regime_flag = 'PT H-mode', # regime of the plasma, currently supporting: 'PT H-mode', 'NT'
        x_method = 'radas', # method to use for the cross-section rates, currently supporting: 'adas', 'radas'
        verbose = False,
    ):

        # User-specified flags
        self.regime_flag = regime_flag
        self.T_rat_flag = T_rat_flag
        self.T_rat = T_rat
        self.verbose = verbose
        if self.verbose:
            self.bvp_verbose = 2
        else:
            self.bvp_verbose = 0
        self.pol_norm = pol_norm
        self.psi_N_inner_boundary = psi_N_inner_boundary
        self.initial_guess = initial_guess

        # Inner boundary handling
        if psi_N_inner_boundary is not None:
            self.psi_N_inner_boundary = float(psi_N_inner_boundary)
            self.nFC_threshold = None
            self.nCX_threshold = None
        else:
            if nFC_threshold is not None:
                self.nFC_threshold = nFC_threshold
            else: 
                raise NotImplementedError("nFC_threshold must be specified")
            if nCX_threshold is not None:
                self.nCX_threshold = nCX_threshold
            else:
                raise NotImplementedError("nCX_threshold must be specified")

        # Other constants
        E_FC = 3 * 1.60218e-19, # J, Energy of Franck-Condon neutrals as defined in Mahdavi M.A., Maingi R., Groebner R.J., Leonard A.W., Osborne T.H. and Porter G. 2003 Phys. Plasmas 10 3984 J
        self.mu0 = 4 * np.pi * 10**-7 # N/A**2, vacuum magnetic permeability constant
        self.P_tot_e = P_tot_e
        M_e = 9.109e-31, # kg, mass of electron
        M_i = 1.673e-27, # kg, mass of hydrogen nuclei
        self.M_i = M_i
        if species == 'D':
            self.M_eff = 2.0
        elif species == 'D-T':
            self.M_eff = 2.5
        else:
            assert False, 'species must be D or D-T'

        # Load in equilibrium and kinetic profile quantities
        self.set_equil_params(eq=equil_params)
        self.set_kprof_params(kprof_params=kprof_params)
        
        # Calculate the magnetic field at each RZ grid point, sets self.B
        self.calc_B(self.rgrid,self.zgrid)

        # calculate the flux surface-averaged |grad(r)| and |grad(r)|^2 and some other quantities like r_psi (outboard midplane minor radius for each flux surface)
        self.calc_gradr() # only a function of geometry

        # Calculate velocities
        self.Z_i = Z_i
        self.e_i = Z_i * constants.e # C
        k_B = 1.38064852e-23 # J/K, Boltzmann constant
        self.V_th_i = np.sqrt(2*k_B*self.T_i_K/(M_i*self.M_eff)) # m/s, per psi_N_eval for Ti
        self.V_th_e = np.sqrt(2*k_B*self.T_e_K/M_e) # m/s, per psi_N_eval for Te
        self.V_FC = np.sqrt(8*E_FC/((np.pi**2) * M_i*self.M_eff)) # m/s
        self.V_cx = np.sqrt(2*k_B*self.T_i_K/(np.pi * M_i*self.M_eff)) # m/s, per psi_N_eval for Ti

        # Load in cross-section rates, which are only a function of temperature if we use an average density (predictive models could provide a guess density)
        self.cross_section_rates(species=species,x_method=x_method)

        # Setup for diffusion coefficient that does not include free parameters and n_e
        self.c_s = (self.e_i * self.T_e * 1e3 / (M_i * self.M_eff)) ** 0.5 # m/s, cs = (e*T_e/mD)^1/2, T_e in keV -> eV via 1e3, as defined in W. Guttenfelder et al 2021 Nucl. Fusion 61 056005
        V_th_i_rz = self.psi_rz_expand(self.V_th_i, psi_N_A='T_e')
        self.rho_s = V_th_i_rz*M_i*self.M_eff / (self.e_i * self.B) # m, known on each RZ grid point
        self.rho_s = self.fsa(self.rho_s,flux_surfaces='T_e') # m, known on each flux surface, outputs nan for psi_N < 0.01 or psi_N > 0.99
        valid = ~np.isnan(self.rho_s)
        self.rho_s = interp1d(self.psi_Te_eval[valid], self.rho_s[valid], kind='linear',bounds_error=False, fill_value='extrapolate')(self.psi_Te_eval) # removes nan values from rho_s

        # Interpolate quantities onto the pressure psi_N grid
        T_e_pres = interp1d(self.psi_Te_eval, self.T_e, kind='linear',
                            bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.T_e_pres = T_e_pres * (1e3) # eV, on psi_N_pres grid
        self.T_i_pres = interp1d(
            self.psi_Ti_eval, self.T_i, kind='linear',
            bounds_error=False, fill_value='extrapolate'
        )(self.psi_N_pres) * (1e3)  # eV
        self.n_e_pres = interp1d(self.psi_ne_eval, self.n_e, kind='linear',
                            bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.n_i_pres = interp1d(self.psi_ni_eval, self.n_i, kind='linear',
                            bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.c_s = interp1d(self.psi_Te_eval, self.c_s, kind='linear',
                       bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.rho_s = interp1d(self.psi_Te_eval, self.rho_s, kind='linear',
                         bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)

        grad_Te = np.gradient(self.T_e_pres * (1.60218e-19), self.r_psi) # gradient in J/m, T_e_pres is in eV
        self.D_ETG_x = P_tot_e / (self.S_plasma * abs(grad_Te)) # evaluated at each psi_N_pres, not including free parameter De_chie_etg and n_e
        self.D_NEO = 0.05 * (self.c_s * self.rho_s**2) / self.a

        # Outer Dirichlet boundary condition for electrons
        if ne_x0 is None:
            self.ne_x0 = self.n_e[-1]
            self.ne_x0_manual = False
        else:
            self.ne_x0 = ne_x0
            self.ne_x0_manual = True


    def calc_pressure_quantities_sc(self,n_e,x):
        """Calculate the pressure, alpha, and D_KBM on the psi_N_pres grid."""
        n_e = interp1d(x, n_e, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        _pres = n_e * self.T_e_pres * 1.60218e-19 # Pa
        _alpha = (
            -(2 * np.gradient(self.V_plasma, self.psi_pres) / ((2 * np.pi) ** 2))
            * self.mu0
            * np.gradient(_pres, self.psi_pres)
            * np.sqrt(self.V_plasma / (2 * self.Rmajor * np.pi ** 2))
        )
        self._D_KBM = np.where(
            _alpha > self.alpha_crit,
            self.C_KBM*(_alpha-self.alpha_crit)*(self.c_s*self.rho_s**2)/self.a,
            0)

    def update_free_params(self, alpha_crit=None, C_KBM=None, De_chie_etg=None, nFC_x0=None,
                           nFC_threshold=None, nCX_threshold=None,
                           psi_N_inner_boundary=None,
                           ncx_x0_ratio=None,
                           ne_inner=None, dne_dx_inner=None, ne_x0=None,
                           clear_solution=False):
        """Update free parameters.

        Call this instead of constructing a new instance when the MHD equilibrium
        is fixed.

        Parameters
        ----------
        alpha_crit : float
        C_KBM : float
        De_chie_etg : float
        nFC_x0 : float
        nFC_threshold : float or None, optional
            Override the FC-neutral threshold used by find_inner_boundary.
            If omitted, keeps the value from construction.
        nCX_threshold : float or None, optional
            Override the CX-neutral threshold used by find_inner_boundary.
            If omitted, keeps the value from construction.
        psi_N_inner_boundary : float, optional
            If given, the inner boundary is placed at this psi_N value directly
            and the adaptive threshold-based logic is disabled (both thresholds
            forced to None for this run).
        ncx_x0_ratio : float, optional
            Override the ratio of nCX at the separatrix to nFC at the separatrix (used in coupled solver).
            If omitted, keeps the value from construction.
        ne_inner : float, optional
            Override the electron density at the inner boundary.
            If omitted, keeps the value from construction.
        dne_dx_inner : float, optional
            Override the derivative of the electron density at the inner boundary.
            If omitted, keeps the value from construction.
        ne_x0 : float, optional
            Override the electron density at the outer boundary.
            If omitted, keeps the value from construction.
        clear_solution : bool, default True
            If True, drop cached BVP solution attributes from a previous
            :meth:`solve` call.  Set False inside the Picard loop.
        """
        
        # Set/update free parameters alpha_crit, C_KBM, De_chie_etg, nFC_x0, psi_N_inner_boundary
        if alpha_crit is not None:
            self.alpha_crit = alpha_crit
        if C_KBM is not None: 
            self.C_KBM = C_KBM
        if De_chie_etg is not None:
            self.De_chie_etg = De_chie_etg
        if nFC_x0 is not None:
            self.nFC_x0 = nFC_x0
        if ncx_x0_ratio is not None:
            self.ncx_x0_ratio = ncx_x0_ratio
            self.nCX_x0 = self.ncx_x0_ratio * self.nFC_x0

        if psi_N_inner_boundary is not None:
            self.psi_N_inner_boundary = float(psi_N_inner_boundary)
            self.nFC_threshold = None
            self.nCX_threshold = None
        else:
            if nFC_threshold is not None:
                self.nFC_threshold = nFC_threshold
            if nCX_threshold is not None:
                self.nCX_threshold = nCX_threshold

        if ne_inner is not None:
            self.ne_inner = ne_inner
        if dne_dx_inner is not None:
            self.dne_dx_inner = dne_dx_inner
        if ne_x0 is not None:
            self.ne_x0 = ne_x0

        if clear_solution:
            for _attr in ('sol', 'sol_first', 'x_sol', 'ne_sol', 'dne_dx_sol',
                          'exp_term_arr', 'nFC_sol', 'integral_from_0'):
                if hasattr(self, _attr):
                    delattr(self, _attr)

        self.invalidate_firedrake_cache()


    def check_free_params(self):
        """Raise if any of the four free parameters is unset.

        The model is undefined without all of ``FREE_PARAM_NAMES``; catching
        it here gives a readable error instead of a ``None`` propagating into
        ``construct_C_ETG`` or the KBM gate.
        """
        missing = [n for n in FREE_PARAM_NAMES
                   if getattr(self, n, None) is None]
        if missing:
            raise ValueError(
                "The Saarelma-Connor model needs all four free parameters; "
                f"missing: {', '.join(missing)}.  Set them on the "
                "constructor, with update_free_params(), or pass "
                "free_params={'alpha_crit': ..., 'De_chie_etg': ..., "
                "'C_KBM': ..., 'nFC_x0': ...} to the solver."
            )

    def apply_free_params(self, free_params=None):
        """Optionally update the free parameters, then validate them.

        Shared by every solver entry point so that ``free_params`` means the
        same thing everywhere.  Going through :meth:`update_free_params`
        (rather than assigning the attributes directly) is what keeps the
        derived quantities -- ``nCX_x0``, the inner-boundary thresholds, the
        cached Firedrake objects -- consistent with the new values.

        Parameters
        ----------
        free_params : dict or None
            Any subset of ``{'alpha_crit', 'C_KBM', 'De_chie_etg',
            'nFC_x0'}``; omitted entries keep their current value.  Any
            further keyword accepted by :meth:`update_free_params` (e.g.
            ``psi_N_inner_boundary``, ``ne_x0``) may be included and is
            forwarded.  None skips the update and only validates.
        """
        if free_params is not None:
            fp = dict(free_params)
            self.update_free_params(
                fp.pop("alpha_crit", self.alpha_crit),
                fp.pop("C_KBM", self.C_KBM),
                fp.pop("De_chie_etg", self.De_chie_etg),
                fp.pop("nFC_x0", self.nFC_x0),
                **fp,
            )
        self.check_free_params()


    def inner_boundary_limits(self, outer_threshold=None, safety_margin=0.01,
                              x_res=100, psi_N_default=0.85):
        """Return (psi_N_inner_limit, psi_N_outer_limit) — the valid range for
        psi_N_inner_boundary given current parameters (D_ped, nFC_x0).

        The OUTER limit (closest to separatrix, largest psi_N) is determined by
        the nFC/nCX threshold logic in ``find_inner_boundary``.
        The INNER limit (deepest in core, smallest psi_N) is found by scanning
        from the separatrix inward until ``dn_e/dx`` is no longer
        steeper than ``safety_margin * min(dn_e/dx)``.

        Parameters
        ----------
        outer_threshold : float, optional
            Threshold applied to BOTH nFC_threshold and nCX_threshold when
            computing the outer limit.  Default: keep current thresholds.
        safety_margin : float, default ``0.01``
            Fraction of the most negative ``dn_e/dx`` used as the inner
            slope cutoff (see ``psi_inner_safety_margin`` in scan notebooks).
        x_res : int
            Resolution passed to ``setup_solver_grids`` if it has not yet been
            called on this instance.
        psi_N_default : float, default ``0.85``
            Default psi_N value used when no valid inner boundary is found.
        """
        # Ensure form-factor / solver-grid setup has been done.
        if not hasattr(self, 'fFC'):
            self.form_factor(type='FC')
        if not hasattr(self, 'fCX'):
            self.form_factor(type='cx')
        if not hasattr(self, 'x_init') or not hasattr(self, 'S_i_pres'):
            self.setup_solver_grids(res=x_res)

        # ---- OUTER limit ----------------------------------------------------
        saved_thr_fc = self.nFC_threshold
        saved_thr_cx = self.nCX_threshold
        saved_psi    = self.psi_N_inner_boundary
        saved_x_in   = self.x_inner

        if outer_threshold is not None:
            self.nFC_threshold = float(outer_threshold)
            self.nCX_threshold = float(outer_threshold)

        # Start from the default so find_inner_boundary doesn't compound onto
        # a previously narrowed value.
        self.find_inner_boundary()
        psi_N_outer = float(self.psi_N_inner_boundary)

        # Restore
        self.nFC_threshold = saved_thr_fc
        self.nCX_threshold = saved_thr_cx
        self.psi_N_inner_boundary = saved_psi
        self.x_inner = saved_x_in

        # ---- INNER limit ----------------------------------------------------
        dne_dx = np.gradient(self.n_e_pres, self.x_init)
        slope_cut = float(safety_margin) * float(np.min(dne_dx))
        psi_N_inner = None
        for i in range(len(dne_dx)):
            j = len(dne_dx) - i - 1  # separatrix -> core
            if dne_dx[j] >= slope_cut:
                psi_N_inner = float(self.psi_N_pres[j])
                break
        if psi_N_inner is None:
            if self.verbose:
                print(
                    "No valid inner boundary found, defaulting to psi_N=0.85 "
                    "for inner boundary"
                )
            psi_N_inner = psi_N_default

        # Guarantee monotonic ordering (inner <= outer) even with edge cases.
        if psi_N_inner > psi_N_outer:
            psi_N_inner, psi_N_outer = psi_N_outer, psi_N_inner

        return psi_N_inner, psi_N_outer

    def find_boundary_points(self,eq):
        """Find the top/bottom/inboard/outboard extrema of the separatrix.

        Parameters
        ----------
        eq : dict
            Equilibrium dictionary returned by ``read_eqdsk``.  Must contain
            keys: ``nr``, ``nz``, ``rleft``, ``rdim``, ``zmid``, ``zdim``,
            ``psirz``, ``raxis``, ``zaxis``, ``psimag``, ``psibry``.
            Optionally ``rzout`` (boundary R,Z points).

        Returns
        -------
        result : dict
            ``'top'``      – ``(R, Z)`` of the upper boundary point
            ``'bottom'``   – ``(R, Z)`` of the lower boundary point
            ``'outboard'``  – ``(R, Z)`` of the outboard boundary point
            ``'inboard'``   – ``(R, Z)`` of the inboard boundary point
        """

        psi = eq['psirz']
        psibry = eq['psibry']

        if 'rzout' in eq and eq['rzout'] is not None and len(eq['rzout']) > 0:
            bdy = eq['rzout']
        else:
            nr = eq['nr']
            nz = eq['nz']
            r = np.linspace(eq['rleft'], eq['rleft'] + eq['rdim'], nr)
            z = np.linspace(eq['zmid'] - eq['zdim']/2, eq['zmid'] + eq['zdim']/2, nz)
            import matplotlib
            matplotlib.use('Agg')
            fig, ax = plt.subplots()
            cs = ax.contour(r, z, psi, levels=[psibry])
            bdy = np.vstack(cs.allsegs[0])
            plt.close(fig)
        itop = np.argmax(bdy[:, 1])
        ibot = np.argmin(bdy[:, 1])
        iout = np.argmax(bdy[:, 0])
        iin  = np.argmin(bdy[:, 0])
        top      = (bdy[itop, 0], bdy[itop, 1])
        bottom   = (bdy[ibot, 0], bdy[ibot, 1])
        outboard = (bdy[iout, 0], bdy[iout, 1])
        inboard  = (bdy[iin,  0], bdy[iin,  1])

        return {
            'top': top,
            'bottom': bottom,
            'outboard': outboard,
            'inboard': inboard,
        }

    def plasma_surface_area_and_volume(self):
        """Compute the plasma surface area and enclosed volume at each flux surface.

        For each psi in np.linspace(psimag, psibry, len(self.pres)), extracts
        the flux surface contour from the 2D psi grid, then computes the
        toroidal surface area (Pappus' theorem) and volume (exact
        piecewise-linear revolution integral) of the surface of revolution.

        Parameters
        ----------
        self : object
            instance of saarelma_connor class

        Sets
        ----
        self.S_plasma : ndarray, shape (n_psi,)
            Toroidal surface area (m^2) at each flux surface.
        self.V_plasma : ndarray, shape (n_psi,)
            Enclosed toroidal volume (m^3) at each flux surface.
        """

        n_psi = len(self.psi_pres)

        self.S_plasma = np.zeros(n_psi)
        self.V_plasma = np.zeros(n_psi)

        # Magnetic axis, used to pick the closed core contour (not an open
        # SOL / divertor-leg segment) at each psi level.  On double-null
        # (e.g. SPARC / ARC) equilibria a given psi level also produces open
        # contours that can be *longer* than the closed core surface, so the
        # previous "longest segment" heuristic silently returned garbage
        # volumes / areas (non-monotonic V, sign-flipping dV/dpsi).  Reuse the
        # same selector as fsa() / calc_gradr().
        R_axis = self.eq['raxis']
        Z_axis = self.eq['zaxis']

        # Extract the flux surface contour from the 2D psi grid
        fig, ax = plt.subplots()
        for i in range(n_psi):
            ax.cla()
            cs = ax.contour(self.rgrid, self.zgrid, self.psi_RZ,
                            levels=[self.psi_pres[i]])

            segs = cs.allsegs[0]
            seg = self._select_core_contour(segs, R_axis, Z_axis)
            if seg is None:
                # No closed contour around the axis at this psi level
                # (e.g. exactly at / beyond the separatrix); filled from
                # valid neighbours below.
                self.S_plasma[i] = np.nan
                self.V_plasma[i] = np.nan
                continue

            R = seg[:, 0]
            Z = seg[:, 1]

            # Close the contour so the integral spans a full 2*pi
            if not (np.isclose(R[0], R[-1]) and np.isclose(Z[0], Z[-1])):
                R = np.append(R, R[0])
                Z = np.append(Z, Z[0])

            dZ = np.diff(Z)
            dR = np.diff(R)
            R_i  = R[:-1]
            R_ip = R[1:]

            # Toroidal volume:  V = (pi/3) |sum (Z_{i+1}-Z_i)(R_i^2 + R_i*R_{i+1} + R_{i+1}^2)|
            # Exact integral of pi*R^2 dZ for piecewise-linear boundary segments
            self.V_plasma[i] = (np.pi / 3.0) * abs(np.sum(dZ * (R_i**2 + R_i * R_ip + R_ip**2))) # m^3, volume enclosed by the plasma per poloidal flux

            # Poloidal cross-section area: Shoelace formula - general to any polygon (Pappus' theorem)
            dl = np.sqrt(dR**2 + dZ**2)
            self.S_plasma[i] = 2.0 * np.pi * np.sum(0.5 * (R_i + R_ip) * dl) # m^2, total surface area of plasma

        plt.close(fig)

        # Fill NaN entries (surfaces where no closed core contour was found,
        # typically right at / beyond the separatrix on diverted equilibria)
        # by extrapolating from the nearest valid neighbours, mirroring
        # calc_gradr().
        for arr in (self.S_plasma, self.V_plasma):
            valid = np.isfinite(arr)
            if valid.any() and not valid.all():
                arr[:] = interp1d(self.psi_N_pres[valid], arr[valid],
                                  kind='linear', bounds_error=False,
                                  fill_value='extrapolate')(self.psi_N_pres)

    def calc_B(self,R_eval,Z_eval):
        """Calculate magnetic field at some point in the plasma
            Always use (rho,theta,var_zeta) coordinate convention as defined by https://crppwww.epfl.ch/~sauter/cocos/Sauter_COCOS_Tokamak_Coordinate_Conventions.pdf

           Note: sigma_Bb is not important for this model, we will always use sigma_Bp=1.
        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        R_eval : float or array
            radial location at which to evaluate the magnetic field
        Z_eval : float or array
            vertical location at which to evaluate the magnetic field
        """

        F = self.eq['fpol']
        psi_F = np.linspace(self.eq['psimag'], self.eq['psibry'], len(F))

        e_Bp = 1 if self.pol_norm else 0

        r = self.rgrid
        z = self.zgrid
        spl = RectBivariateSpline(z, r, self.eq['psirz'])

        R_eval_arr = np.atleast_1d(R_eval)
        Z_eval_arr = np.atleast_1d(Z_eval)

        psi = spl(Z_eval_arr, R_eval_arr, grid=False)
        dpsi_dR = spl(Z_eval_arr, R_eval_arr, dx=0, dy=1, grid=False) # specifying dx, dy specifies the derivative order in the respective direction
        dpsi_dZ = spl(Z_eval_arr, R_eval_arr, dx=1, dy=0, grid=False)
        F_interp = interp1d(psi_F, F, kind='linear', bounds_error=False, fill_value="extrapolate")
        B_R = (1 / ((2*np.pi)**e_Bp)) * dpsi_dZ / R_eval_arr # R component of the magnetic field
        B_Z = (1 / ((2*np.pi)**e_Bp)) * -dpsi_dR / R_eval_arr # Z component of the magnetic field
        B_phi = (F_interp(psi) / R_eval_arr) # T, toroidal magnetic field

        self.B = np.sqrt(B_R**2 + B_Z**2 + B_phi**2) # T, total magnetic field at each R_eval, Z_eval
        return self.B, [B_R, B_Z, B_phi]

    def set_equil_params(self,eq):
        self.eq = eq

        bdry = self.find_boundary_points(eq=eq)

        rmax_top = bdry['top'][0]
        rmax_bottom = bdry['bottom'][0]
        zmax_top = bdry['top'][1]
        zmax_bottom = bdry['bottom'][1]
        rmax_outboard = bdry['outboard'][0]
        rmax_inboard = bdry['inboard'][0]
        z_outboard = bdry['outboard'][1]
        # zmax_inboard = bdry['inboard'][1]

        # Geometric parameters
        self.Raxis = self.eq['raxis'] # m, location of magnetic axis relative to device rotational line of toroidal symmetry
        self.Rmajor = (rmax_outboard + rmax_inboard) / 2 # m
        self.a = (rmax_outboard - rmax_inboard) / 2 # minor radius # m
        delta_u = (self.Rmajor - rmax_top) / self.a
        delta_l = (self.Rmajor - rmax_bottom) / self.a
        self.delta = (delta_u + delta_l) / 2 # dimensionless, total triangularity
        self.kappa = (zmax_top - zmax_bottom) / (2*self.a) # dimensionless, elongation

        # Plasma parameters (skip the magnetic axis to avoid degenerate zero-area/volume flux surface)
        self.Ip = self.eq['ip'] / 1e6 # MA, Plasma current
        self.psi_pres = np.linspace(self.eq['psimag'], self.eq['psibry'], len(self.eq['pres']))[1:]
        self.psi_N_pres = (self.psi_pres - self.eq['psimag']) / (self.eq['psibry'] - self.eq['psimag'])
        # self.pres_gfile = self.eq['pres'][1:] # pressure is NOT an input to this model but using this for plotting - want to use pfile pressure instead

        # Grids
        self.rgrid = np.linspace(self.eq['rleft'],self.eq['rleft']+self.eq['rdim'],self.eq['nr']) # m, 1D R grid
        self.zgrid = np.linspace(self.eq['zmid']-self.eq['zdim']/2,self.eq['zmid']+self.eq['zdim']/2,self.eq['nz']) # m, 1D Z grid
        self.psi_RZ = self.eq['psirz'] # 2D poloidal flux array at each RZ grid point
        self.psi_RZ_N = (self.psi_RZ - self.eq['psimag']) / (self.eq['psibry'] - self.eq['psimag']) # normalized poloidal flux at each RZ grid point
        # self.rsep_mid = (((rmax_outboard - self.Raxis)**2) + ((z_outboard - self.eq['zaxis'])**2))**5 # separatrix radius at midplane

        self.plasma_surface_area_and_volume()

    def set_kprof_params(self,kprof_params):


        #-------- electron information (required) --------#
        self.T_e = kprof_params['T_e'] # keV
        self.n_e = kprof_params['n_e'] # m^(-3)
        self.psi_Te_eval = kprof_params['psin_Te'] # psi_N values at which T_e is evaluated
        self.psi_ne_eval = kprof_params['psin_ne'] # psi_N values at which n_e is evaluated
        
        self.T_e_K = self.T_e * 1e3 * 11604.52 # T_e values (K) evaluated at psi_Te_eval


        #-------- ion information (not required) --------#

        # ion temperatures
        if 'T_i' in kprof_params and 'psin_Ti' in kprof_params:
            self.T_i_K = kprof_params['T_i'] * 1e3 * 11604.52 # K
            self.psi_Ti_eval = kprof_params['psin_Ti']
        elif self.T_rat_flag == True:
            self.T_i = self.T_e * self.T_rat # keV
            self.psi_Ti_eval = self.psi_Te_eval
        else:
            raise NotImplementedError("if T_i is not provided, must specify T_rat_flag")
        self.T_i_K = self.T_i * 1e3 * 11604.52 # K

        # ion densities
        if 'n_i' in kprof_params and 'psin_ni' in kprof_params:
            self.n_i = kprof_params['n_i'] # m^(-3)
            self.psi_ni_eval = kprof_params['psin_ni']
        elif self.quasineutral_flag == True:
            self.n_i = self.n_e # m^(-3)
            self.psi_ni_eval = self.psi_ne_eval
        else:
            raise NotImplementedError("if n_i is not provided, must use quasi-neutrality")

    def cross_section_rates(self,species='D',x_method='radas'):
        """Calculate ionization and charge-exchange rate coefficients.

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        species : string
            species of ions: 'D' or 'D-T' (D-T averages D and T RADAS rates)
        x_method : string
            'radas' (ADF11 from local RADAS dumps) or 'adas' (bundled ADAS files)
        """

        if x_method == 'adas':
            if species == 'D':
                
                # charge-exchange cross-section
                '''sigma_cx_perE = np.array([3.81*10**(-18), 3.85*10**(-18), 3.44*10**(-18), 2.71*10**(-18), 1.74*10**(-18), 8.10*10**(-20), 9.56*10**(-22), 1.46*10**(-23)]) # m^2
                E = np.array([3.23*10**(-16), 9.68*10**(-16), 3.23*10**(-15), 6.45*10**(-15), 9.68*10**(-15), 3.23*10**(-14), 9.68*10**(-14), 2.26*10**(-13)]) # J
                sigma_cx_interp = interp1d(E, sigma_cx_perE, kind='linear',fill_value='extrapolate',bounds_error=False)
                self.sigma_cx = sigma_cx_interp(0.5 * (self.M_i*self.M_eff) * self.V_cx**2) # m^2, charge-exchange cross-section'''

                # ionization rate coefficient profile: scd_adas(n_e, T_e[eV]) at each psi_Te_eval point.
                # also including CX rate coefficients from ADAS ADF11 https://open.adas.ac.uk/detail/adf11/ccd96/ccd96_d.dat
                # n_e is given on psi_ne_eval, so interpolate it onto psi_Te_eval first.
                n_e_at_Te = interp1d(self.psi_ne_eval, self.n_e, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')(self.psi_Te_eval)
                n_e_input = np.mean(n_e_at_Te)
                T_e_eV = self.T_e * 1e3 # keV -> eV
                self.S_i = np.array([
                    # scd_adas(n_e_at_Te[i], T_e_eV[i]) for i in range(len(self.psi_Te_eval))
                    scd_adas(n_e_input, T_e_eV[i]) for i in range(len(self.psi_Te_eval))
                ]) # m^3/s, on psi_Te_eval
                self.S_cx = scx_adas(np.ones_like(T_e_eV) * n_e_input, T_e_eV)

            else:
                assert False, 'species not supported for x_method=adas (use D)'
        elif x_method == 'radas':
            # RADAS ADF11 SCD (effective ionisation) and CCD (CX cross-coupling).
            # Same evaluation pattern as adas: mean ne, rate vs Te profile.
            # Tables are 2D (ne, Te); ne is fixed at the profile mean.
            n_e_at_Te = interp1d(self.psi_ne_eval, self.n_e, kind='linear',
                                bounds_error=False, fill_value='extrapolate')(self.psi_Te_eval)
            ne_arr = n_e_at_Te
            # n_e_input = float(np.mean(n_e_at_Te))
            # ne_arr = np.ones_like(T_e_eV) * n_e_input
            T_e_eV = self.T_e * 1e3  # keV -> eV

            if species == 'D':
                self.S_i = np.asarray(scd_radas(ne_arr, T_e_eV, isotope='D'), dtype=float)
                self.S_cx = np.asarray(scx_radas(ne_arr, T_e_eV, isotope='D'), dtype=float)
            elif species == 'D-T':
                S_i_D = np.asarray(scd_radas(ne_arr, T_e_eV, isotope='D'), dtype=float)
                S_i_T = np.asarray(scd_radas(ne_arr, T_e_eV, isotope='T'), dtype=float)
                S_cx_D = np.asarray(scx_radas(ne_arr, T_e_eV, isotope='D'), dtype=float)
                S_cx_T = np.asarray(scx_radas(ne_arr, T_e_eV, isotope='T'), dtype=float)
                self.S_i = 0.5 * (S_i_D + S_i_T)
                self.S_cx = 0.5 * (S_cx_D + S_cx_T)
            else:
                assert False, "species must be 'D' or 'D-T' for x_method=radas"
                            
        else:
            assert False, f"x_method must be 'radas' or 'adas', got {x_method!r}"


    def form_factor(self,x,type = 'ex'):
        """Calculate the form factor for FC or charge-exchange cases
        Currently just sets to 1, but can be updated to use a more sophisticated to account for poloidal asymmetries in the FC and CX neutral profiles.

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        x : array
            radial grid to evaluate the form factor on
        type : string
            type of form factor to calculate, supporting: FC, cx
        """
        assert type != 'FC' or type != 'cx', 'form factor must be for FC or cx'

        # grad(r) and nFC or nCX are needed

        if type == 'FC':
            self.fFC = np.ones_like(x)
        elif type == 'cx':
            self.fCX = np.ones_like(x)


    def setup_solver_grids(self,res = 100):
        """Setup the grids for the solver and calculates the flux surface-averaged |grad(r)| and |grad(r)|^2
        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        res : int, optional
            number of points to use in the radial grid if using the first method of defining the radial grid

        Returns
        -------
        self.x_prev : ndarray, shape (n_psi,)
            Radial grid (shifted so zero is at the separatrix) only at midplane, defined on the psi_N_pres grid.
        self.gradr_fsa : ndarray, shape (n_psi,)
            Flux-surface-averaged |grad(r)| at each psi_N_pres surface.
        self.gradr2_fsa : ndarray, shape (n_psi,)
            Flux-surface-averaged |grad(r)|^2 at each psi_N_pres surface.
        self.S_i_pres : ndarray, shape (n_psi,)
            Ionization cross-section at each psi_N_pres surface.
        self.S_cx_pres : ndarray, shape (n_psi,)
            Charge-exchange cross-section at each psi_N_pres surface.
        self.V_cx_pres : ndarray, shape (n_psi,)
            Volume of the charge-exchange neutral at each psi_N_pres surface.
        """

        # one method of defining the radial grid, requires uncommenting the rsep_mid definition in the mhd_load function
        # self.rmid = np.linspace(0, self.rsep_mid, res) # m, radial grid (shifted so zero is at the magnetic axis)
        # self.xmid = self.rmid - self.rsep_mid # m, radial grid (shifted so zero is at the separatrix)

        # another method of defining the radial grid
        # self.r_psi is the outboard midplane minor radius for each flux surface for the psi_N_pres grid
        self.x_init = self.r_psi - self.r_psi[-1] # m, radial grid (shifted so zero is at the separatrix) only at midplane, defined on the psi_N_pres grid
        self.x_prev = self.x_init.copy() # m, radial grid (shifted so zero is at the separatrix) only at midplane, defined on the psi_N_pres grid

        # interpolate quantities on Te grid to psi_N_pres grid
        self.S_i_pres = interp1d(self.psi_Te_eval, self.S_i, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.S_cx_pres = interp1d(self.psi_Te_eval, self.S_cx, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.V_cx_pres = interp1d(self.psi_Te_eval, np.abs(self.V_cx), kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)

        # note all 1D quantities are now defined on the psi_N_pres grid, which is the same as the x_prev grid

        self.x_inner = interp1d(self.psi_N_pres, self.x_prev, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_inner_boundary)

        # Equilibrium-only coefficient for the Connor-Hastie alpha when
        # written in the dp/dx form (the chain rule absorbs the dx/dpsi
        # Jacobian).  Precomputed on the parent (sorted) self.x_init grid
        # so that downstream Firedrake-DOF grids can use np.interp without
        # ever needing np.gradient on a possibly unsorted DOF array.
        # See calc_pressure_quantities and App. A.7 in the writeup.
        _dxdpsi_xinit = np.gradient(self.x_init, self.psi_pres)
        _dVdpsi_xinit = np.gradient(self.V_plasma, self.psi_pres)
        self._alpha_nodp_xinit = (
            -(2 * _dVdpsi_xinit / ((2 * np.pi) ** 2))
            * self.mu0
            * np.sqrt(self.V_plasma / (2 * self.Rmajor * np.pi ** 2))
            * _dxdpsi_xinit
        ) # alpha = self._alpha_nodp_xinit * dp/dx (with dp/dx evaluated on the same grid)

    def find_inner_boundary(self):
        """Adaptively locate the inner boundary by finding where the neutral
        densities (FC and/or CX) fall below user-supplied thresholds.

        Uses the n_e profile as a proxy for the pre-solve electron
        density to estimate nFC(x) via its exponential attenuation integral
        (Eq. 11) and nCX(x) via the algebraic closure (Eq. 12).  The inner
        boundary is placed at the outermost x (closest to the separatrix)
        where *both* active thresholds are satisfied simultaneously.

        If neither threshold is set (both are None), the method returns
        immediately without changing ``self.psi_N_inner_boundary`` or
        ``self.x_inner``.

        Parameters used from self
        -------------------------
        self.nFC_threshold : float or None
            nFC/nFC(x=0) must drop below this fraction.  None disables.
        self.nCX_threshold : float or None
            nCX/nCX_peak must drop below this fraction.  None disables.
        self.n_e_pres, self.x_init : array
            density on equil-defined grid and radial grid.
        self.S_i_pres, self.S_cx_pres : array
            Ionization and CX rate coefficients on psi_N_pres grid.
        self.D_ped, self.gradr2_fsa, self.V_cx_pres : array
            Diffusion and geometry arrays on psi_N_pres grid.
        self.V_FC, self.fFC, self.fCX : float
            FC neutral speed and form factors.
        self.nFC_x0 : float
            FC neutral density at the separatrix (boundary condition).

        Updates
        -------
        self.psi_N_inner_boundary : float
            Updated to the adaptively found inner boundary psi_N.
        self.x_inner : float
            Updated to the corresponding physical x coordinate (m).
        """
        if self.nFC_threshold is None and self.nCX_threshold is None:
            self.x_inner = interp1d(self.psi_N_pres, self.x_init, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_inner_boundary)
            return  # use fixed psi_N_inner_boundary and corresponding x_inner

        ne   = self.n_e_pres
        Si   = self.S_i_pres
        Scx  = self.S_cx_pres
        Dped = self.D_NEO + self._D_KBM + (self.D_ETG_x / ne)
        gr2  = self.gradr2_fsa
        Vcx  = self.V_cx_pres      # array, local thermal CX speed
        Vfc  = abs(self.V_FC)      # scalar FC speed
        fFC  = self.fFC            # form factor (= 1 currently)
        fCX  = self.fCX            # form factor (= 1 currently)
        x    = self.x_init         # physical x, separatrix = 0, inward < 0

        # ------------------------------------------------------------------ #
        # nFC estimate: integrate from separatrix inward (descending x)       #
        # nFC(x) / nFC_x0 = exp( ∫_0^x  ne*(Si+Scx)/(fFC*Vfc)  dx' )        #
        # ------------------------------------------------------------------ #
        order_desc = np.argsort(x)[::-1]   # index order: separatrix → core
        x_desc     = x[order_desc]
        integrand  = (ne * (Si + Scx)) / (fFC * Vfc)
        cumint     = cumulative_trapezoid(integrand[order_desc], x_desc, initial=0.0)
        nFC_ratio_desc = np.exp(cumint)    # decays < 1 going inward

        # map back to original (psi_N ascending) order
        nFC_ratio = np.empty_like(nFC_ratio_desc)
        nFC_ratio[order_desc] = nFC_ratio_desc

        # ------------------------------------------------------------------ #
        # nCX estimate: algebraic closure (Eq. 12)                            #
        # nCX = -(gr2*Dped/(Vcx*fCX))*(dne/dx - dne_dx_inner)               #
        #        - (Vfc*fFC/(Vcx*fCX))*((Si+Scx/2)/(Si+Scx))*nFC            #
        # ------------------------------------------------------------------ #
        dne_dx = np.gradient(ne, x)
        # Use the innermost gradient as the dne_dx_neginf proxy
        dne_dx_inner_est = dne_dx[np.argmin(x)]

        f_arr     = gr2 * Dped
        flux_term = -(f_arr / (Vcx * fCX)) * (dne_dx - dne_dx_inner_est)
        fc_term   = -(Vfc * fFC / (Vcx * fCX)) * ((Si + Scx / 2) / (Si + Scx)) * (self.nFC_x0 * nFC_ratio)
        nCX_est   = flux_term + fc_term
        nCX_est   = np.maximum(nCX_est, 0.0)   # physical lower bound

        nCX_peak  = np.max(nCX_est)
        nCX_ratio = nCX_est / nCX_peak if nCX_peak > 0 else np.zeros_like(nCX_est)

        # ------------------------------------------------------------------ #
        # Find the outermost x (closest to separatrix) where both active      #
        # thresholds are satisfied.                                            #
        # Work in ascending-x order (core first, separatrix last).            #
        # ------------------------------------------------------------------ #
        asc      = np.argsort(x)
        x_asc    = x[asc]
        psi_asc  = self.psi_N_pres[asc]

        # build combined mask: only consider thresholds the user activated
        mask = np.ones(len(x), dtype=bool)
        if self.nFC_threshold is not None:
            mask &= nFC_ratio[asc] < self.nFC_threshold
        if self.nCX_threshold is not None:
            mask &= nCX_ratio[asc] < self.nCX_threshold

        crossing = np.where(mask)[0]

        if len(crossing) == 0:
            import warnings
            warnings.warn(
                f"Neutral densities never fall below the requested thresholds "
                f"(nFC_threshold={self.nFC_threshold}, nCX_threshold={self.nCX_threshold}) "
                f"across the available domain.  Keeping the fixed inner boundary "
                f"at psi_N = {self.psi_N_inner_boundary:.3f}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        # The outermost (highest x, most separatrix-side) crossing point that
        # still satisfies both thresholds. We scan inward from the separatrix
        # (highest x in ascending order = last element) and take the first hit.
        idx       = crossing[-1]
        x_new     = float(x_asc[idx])
        psi_new   = float(psi_asc[idx])

        self.psi_N_inner_boundary = psi_new
        print(f"psi_N_inner_boundary: {self.psi_N_inner_boundary:.4f}")
        self.x_inner = x_new
        print(f"x_inner: {self.x_inner:.4f} m")

    @staticmethod
    def _select_core_contour(segs, R_axis, Z_axis, closure_tol=0.05):
        """Pick the closed flux-surface contour that encloses the magnetic
        axis from a list of matplotlib contour segments.

        Previously the *longest* segment was used, but on double-null
        (e.g. SPARC) or diverted equilibria a given psi level also produces
        open SOL / divertor-leg / private-flux contours which can be longer
        than the closed core surface.  Integrating over one of those open,
        theta-folded curves silently corrupts the flux-surface average
        (it even produced *negative* <|grad r|^2>).

        Parameters
        ----------
        segs : list of (N, 2) ndarray
            Contour segments (R, Z) for one psi level.
        R_axis, Z_axis : float
            Magnetic axis position (m).
        closure_tol : float
            Segment counts as closed if the gap between its endpoints is
            below ``closure_tol`` times its perimeter.

        Returns
        -------
        seg : ndarray or None
            The longest closed segment enclosing the axis, or None if no
            segment qualifies (caller should treat the surface as invalid,
            e.g. NaN + neighbour fill).
        """
        candidates = []
        for s in segs:
            if len(s) < 4:
                continue
            perim = np.hypot(np.diff(s[:, 0]), np.diff(s[:, 1])).sum()
            if perim <= 0.0:
                continue
            gap = np.hypot(s[0, 0] - s[-1, 0], s[0, 1] - s[-1, 1])
            if gap > closure_tol * perim:
                continue  # open contour (SOL / divertor leg)
            if not _MplPath(s).contains_point((R_axis, Z_axis)):
                continue  # closed but not around the axis (island etc.)
            candidates.append(s)
        if not candidates:
            return None
        return max(candidates, key=lambda s: len(s))

    @staticmethod
    def _sort_dedup_close_theta(theta_c, R_c, Z_c, min_dtheta=1e-12):
        """Sort contour points by poloidal angle, drop near-duplicate
        angles, and close the contour over a full 2*pi.

        Near-duplicate angles (e.g. the coincident first/last vertices of
        a closed matplotlib contour, or point clusters near an X-point)
        create ~1e-16-wide intervals; Simpson's nonuniform weights blow up
        on the huge interval-length ratios and amplify round-off into
        O(1e-4) errors in the flux-surface average.

        Parameters
        ----------
        theta_c, R_c, Z_c : ndarray
            Poloidal angle and contour coordinates (unsorted).
        min_dtheta : float
            Minimum allowed angular spacing between consecutive points.

        Returns
        -------
        theta_c, R_c, Z_c : ndarray
            Sorted, deduplicated arrays with the closing point
            (theta[0] + 2*pi, R[0], Z[0]) appended.
        """
        idx = np.argsort(theta_c)
        theta_c, R_c, Z_c = theta_c[idx], R_c[idx], Z_c[idx]

        keep = np.concatenate(([True], np.diff(theta_c) > min_dtheta))
        theta_c, R_c, Z_c = theta_c[keep], R_c[keep], Z_c[keep]

        # Close the contour so the integral spans a full 2*pi; drop the
        # last point first if it would duplicate the closing point.
        if theta_c[0] + 2 * np.pi - theta_c[-1] <= min_dtheta:
            theta_c, R_c, Z_c = theta_c[:-1], R_c[:-1], Z_c[:-1]
        theta_c = np.append(theta_c, theta_c[0] + 2 * np.pi)
        R_c = np.append(R_c, R_c[0])
        Z_c = np.append(Z_c, Z_c[0])
        return theta_c, R_c, Z_c

    def calc_gradr(self):
        """Compute <|grad(r)|> at each flux surface.

        r is defined as the outboard-midplane minor radius for each flux
        surface, making it a proper flux-surface label (one value per surface).
        By the chain rule:
            |grad(r)| = |dr/dpsi| * |grad(psi)|
        This varies poloidally because |grad(psi)| is larger where flux
        surfaces are compressed (inboard side) and smaller where they are
        spread apart (outboard side).

        The flux surface average is:
            <|grad(r)|> = ∮ R² |grad(r)| dθ / ∮ R² dθ

        Sets
        ----
        self.r_psi : ndarray, shape (n_psi,)
            Outboard midplane minor radius for each flux surface (m).
        self.gradr_c : ndarray, shape (n_psi,)
            |grad(r)| at each contour point on each flux surface.
        self.gradr_fsa : ndarray, shape (n_psi,)
            Flux-surface-averaged |grad(r)| at each psi_N_pres surface.
        self.gradr2_fsa : ndarray, shape (n_psi,)
            Flux-surface-averaged |grad(r)|^2 at each psi_N_pres surface.
        """
        R_axis = self.eq['raxis']
        Z_axis = self.eq['zaxis']

        psi_spl = RectBivariateSpline(self.zgrid, self.rgrid, self.psi_RZ)
        n_psi = len(self.psi_N_pres)

        # r(psi): find outboard midplane crossing for each flux surface
        R_out = np.linspace(R_axis, self.rgrid[-1], 500)
        Z_mid = np.full_like(R_out, Z_axis)
        psi_mid = psi_spl(Z_mid, R_out, grid=False)
        sort_idx = np.argsort(psi_mid)
        psi_to_R = interp1d(psi_mid[sort_idx], R_out[sort_idx], kind='linear',
                            bounds_error=False, fill_value=np.nan)

        self.r_psi = np.zeros(n_psi)
        for i, psi_val in enumerate(self.psi_pres):
            self.r_psi[i] = psi_to_R(psi_val) - R_axis

        dr_dpsi = np.gradient(self.r_psi, self.psi_N_pres) # (m) / (dimensionless), change in r_midplane(psi_N) over psi_N

        self.gradr_fsa = np.zeros(n_psi)
        self.gradr2_fsa = np.zeros(n_psi)
        fig, ax = plt.subplots()
        for i, psi_val in enumerate(self.psi_pres):
            ax.cla()
            cs = ax.contour(self.rgrid, self.zgrid, self.psi_RZ,
                            levels=[psi_val])
            segs = cs.allsegs[0]
            seg = self._select_core_contour(segs, R_axis, Z_axis)
            if seg is None:
                # No closed contour around the axis at this psi level
                # (e.g. exactly at / beyond the separatrix); filled from
                # valid neighbours below.
                self.gradr_fsa[i] = np.nan
                self.gradr2_fsa[i] = np.nan
                continue
            R_c, Z_c = seg[:, 0], seg[:, 1]

            theta_c = np.arctan2(Z_c - Z_axis, R_c - R_axis) # theta at all points on contour
            theta_c, R_c, Z_c = self._sort_dedup_close_theta(theta_c, R_c, Z_c)

            # |grad(psi)| at each contour point from the equilibrium spline
            dpsi_dR = psi_spl(Z_c, R_c, dx=0, dy=1, grid=False) # value at each point on the contour
            dpsi_dZ = psi_spl(Z_c, R_c, dx=1, dy=0, grid=False) # value at each point on the contour
            grad_psi_mag = np.sqrt(dpsi_dR**2 + dpsi_dZ**2) # value at each point on the contour

            # |grad(r)| = |dr/dpsi| * |grad(psi)| at each contour point on each flux surface i
            gradr_c = np.abs(dr_dpsi[i]) * grad_psi_mag

            den = simpson(R_c**2, theta_c)
            self.gradr_fsa[i] = simpson(R_c**2 * gradr_c, theta_c) / den
            self.gradr2_fsa[i] = simpson(R_c**2 * gradr_c**2, theta_c) / den
        plt.close(fig)

        # Fill NaN/zero entries (near-axis or separatrix edge cases) by extrapolating from the nearest valid neighbours.
        for arr in (self.r_psi, self.gradr_fsa, self.gradr2_fsa):
            valid = np.isfinite(arr) & (arr != 0)
            if valid.any() and not valid.all():
                arr[:] = interp1d(self.psi_N_pres[valid], arr[valid],
                                  kind='linear', bounds_error=False,
                                  fill_value='extrapolate')(self.psi_N_pres)
    
    def non_dimensionalize(self, x, y, L=None, n0=None):
        """Non-dimensionalize the BVP variables.

        Introduces xi = x / L and N = ne / n0 so that both the independent
        and dependent variables are O(1).

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        x : ndarray
            Physical radial grid (m).
        y : ndarray, shape (2, n_points)
            [ne, dne/dx] guess in physical units.
        L : float, optional
            Length scale (m).  Default ``|x_inner|``.
        n0 : float, optional
            Density scale (m^-3).  Default ``ne_x0``.

        Sets
        ----
        self._L, self._n0 : float
            Stored scales for later de-normalization.
        self.xi : ndarray
            Normalized grid (dimensionless, -1 to 0).
        self.N_guess : ndarray, shape (2, n_points)
            [N, dN/dxi] initial guess in normalized units.
        self.dNdxi_neginf : float
            Normalized Neumann BC value.
        """
        if L is None:
            L = abs(self.x_inner)
        if n0 is None:
            n0 = self.ne_x0
        self._L = L
        self._n0 = n0

        self.xi = x / L
        N = y[0] / n0
        dNdxi = y[1] * (L / n0)   # dne/dx = (n0/L)*dN/dxi  =>  dN/dxi = (L/n0)*dne/dx
        self.N_guess = np.vstack([N, dNdxi])

        self.dNdxi_neginf = self.dne_dx_neginf * (L / n0)
    
    def compute_post_solve_SC_neutrals(self):
        """Franck--Condon and charge-exchange densities after ``solve_simplified()``.

        Uses the same closures as ``find_inner_boundary`` (Eqs.~(11)--(12) in
        Saarelma et al., 2023): exponential attenuation of FC neutrals
        from the separatrix and the algebraic CX relation.

        Returns
        -------
        nFC, nCX : ndarray
            Neutral densities (m^-3) on ``x``.
        """
        x = self.x_sol
        dne_dx = self.dne_dx_sol

        ne_x = interp1d(self.x_sol, self.ne_sol, kind='linear', bounds_error=False, fill_value='extrapolate')

        D_ped = self.D_NEO + self._D_KBM + (self.D_ETG_x / ne_x(self.x_prev))

        Si = interp1d(self.x_init, self.S_i_pres, kind='linear',
                      bounds_error=False, fill_value='extrapolate')(x)
        Scx = interp1d(self.x_init, self.S_cx_pres, kind='linear',
                       bounds_error=False, fill_value='extrapolate')(x)
        gr2 = interp1d(self.x_init, self.gradr2_fsa, kind='linear',
                       bounds_error=False, fill_value='extrapolate')(x)
        Dped = interp1d(self.x_prev, D_ped, kind='linear',
                        bounds_error=False, fill_value='extrapolate')(x)
        Vcx = interp1d(self.x_init, self.V_cx_pres, kind='linear',
                       bounds_error=False, fill_value='extrapolate')(x)

        Vfc = abs(self.V_FC)
        fFC = self.fFC
        fCX = self.fCX

        integrand = self.ne_sol * (Si + Scx) / (fFC * Vfc)
        order_desc = np.argsort(x)[::-1]
        x_desc = x[order_desc]
        cumint = cumulative_trapezoid(integrand[order_desc], x_desc, initial=0.0)
        integral_from_0 = np.empty_like(cumint)
        integral_from_0[order_desc] = cumint
        nFC = self.nFC_x0 * np.exp(integral_from_0)

        flux_term = -(gr2 * Dped / (Vcx * fCX)) * (dne_dx - self.dne_dx_neginf)
        fc_term = -(Vfc * fFC / (Vcx * fCX)) * ((Si + Scx / 2) / (Si + Scx)) * nFC
        nCX = np.maximum(flux_term + fc_term, 0.0)
        return nFC, nCX
    
    def invalidate_firedrake_cache(self):
        """Drop cached Firedrake meshes, coefficients, and linear solvers.

        Called automatically by :meth:`update_free_params`.  Call manually
        after changing equilibrium inputs or other quantities that affect
        ``setup_solver_grids`` / ``calc_gradr``.
        """
        self._fd_cache = {}

    def _plot_profiles(self, x_dofs, ne, nFC, nCX, title=""):
        """Plot n_e, <n_FC>, <n_CX> vs x with a secondary psi_N axis on top.

        Used as a diagnostic from ``solve_firedrake`` (gated on ``verbose``)
        to visualise either the initial guess or any intermediate / final
        profile triple.

        Parameters
        ----------
        x_dofs : ndarray
            Per-DOF x coordinates (m), unsorted (as stored in dat.data).
        ne, nFC, nCX : ndarray
            Profiles in the same DOF order as ``x_dofs`` (m^-3).
        title : str
            Figure suptitle.
        """
        # Sort by ascending x for clean line plots (DOF order is not guaranteed
        # to be spatial).
        sort_idx = np.argsort(x_dofs) # if setup correctly, this should not do anything
        x_plot = x_dofs[sort_idx]
        profiles = [ne[sort_idx], nFC[sort_idx], nCX[sort_idx]]
        labels   = [r"$n_e$",
                    r"$\langle n_{FC} \rangle$",
                    r"$\langle n_{CX} \rangle$"]
        colours  = ["tab:blue", "tab:orange", "tab:green"]

        # x <-> psi_N mapping built off the parent grid (self.x_init is
        # monotonically increasing toward the separatrix; self.psi_N_pres
        # is the corresponding normalised poloidal flux).  Used by
        # secondary_xaxis to render psi_N on top of each panel.
        x_to_psiN = interp1d(self.x_init, self.psi_N_pres,
                            kind='linear', bounds_error=False,
                            fill_value='extrapolate')
        psiN_to_x = interp1d(self.psi_N_pres, self.x_init,
                            kind='linear', bounds_error=False,
                            fill_value='extrapolate')

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        if title:
            fig.suptitle(title, fontsize=12)
        for ax, prof, label, col in zip(axes, profiles, labels, colours):
            ax.plot(x_plot, prof, lw=2, color=col)
            ax.set_xlabel(r"$x$ (m)")
            ax.set_ylabel(f"{label} (m$^{{-3}}$)")
            ax.grid(True, alpha=0.3)
            secax = ax.secondary_xaxis('top',
                                    functions=(x_to_psiN, psiN_to_x))
            secax.set_xlabel(r"$\psi_N$")
        # plt.show()
        plt.savefig(f'{title}.png')

    @staticmethod
    def _build_petsc_solver_parameters(linear_solver="lu",
                                    ksp_rtol=1e-8,
                                    ksp_max_it=200):
        """PETSc options for SNES ``solve(F == 0, ...)`` (full Newton path)."""
        linear_solver = str(linear_solver).lower()
        if linear_solver not in ("lu", "gamg"):
            raise ValueError(
                f"linear_solver must be 'lu' or 'gamg', got {linear_solver!r}."
            )

        params = {
            "snes_type": "newtonls",
            "snes_max_it": 50,
            "snes_atol": 1.0e-8,
            "snes_rtol": 1.0e-8,
            "snes_linesearch_type": "bt",
            "mat_type": "aij",
        }

        if linear_solver == "lu":
            params.update({
                "ksp_type": "preonly",
                "pc_type": "lu",
            })
        else:
            params.update({
                "ksp_type": "gmres",
                "ksp_rtol": float(ksp_rtol),
                "ksp_max_it": int(ksp_max_it),
                "ksp_gmres_restart": 30,
                "pc_type": "gamg",
                "mg_levels_ksp_type": "richardson",
                "mg_levels_pc_type": "sor",
                "mg_levels_ksp_max_it": 5,
            })

        return params

    def _ensure_firedrake_coefficient_grids(self, x_res, force=False):
        """Run ``form_factor`` + ``setup_solver_grids`` once per ``x_res``."""
        key = int(x_res)
        if force or self._fd_cache.get("x_res") != key:
            self.setup_solver_grids(res=x_res)
            self.form_factor(type='FC',x=self.x_init)
            self.form_factor(type='cx',x=self.x_init)
            self._fd_cache["x_res"] = key

    def calc_pressure_quantities(self, n_e, average_alpha_pedestal=True):
        """Pressure, ``alpha``, and ``D_KBM`` on the ``x_dofs`` (from separatrix to x_inner) grid.

        When ``average_alpha_pedestal`` is True (default), use the mean of
        local ``alpha`` over the pedestal ``x in [x_inner, 0]`` in Eq.~(25)
        instead of a flux-surface-local ``alpha``.

        Implementation notes
        --------------------
        ``self._fd_cache["x_dofs"]`` is the per-DOF x coordinate as stored by
        Firedrake; for ``CG_p`` with ``p>=2`` the DOFs are grouped by entity
        (vertices first, then cell interiors) and so the array is **not**
        monotonic in space.  ``np.gradient`` interprets consecutive samples
        as adjacent in x, so we must sort to spatial order before any
        finite-difference call and unsort the outputs that are subsequently
        written into ``Function.dat.data`` (which expects DOF order).

        The equilibrium-only piece of ``alpha_nodp`` is precomputed on the
        parent ``self.x_init`` grid in :meth:`setup_solver_grids`; here we
        only need ``np.interp`` (which does not require its first argument
        to be sorted).  The single ``np.gradient`` left at runtime is
        ``dpdx``, whose dependence on ``n_e`` forces the sort.
        """
        x_dofs_arr = self._fd_cache["x_dofs"]
        sort_idx = np.argsort(x_dofs_arr)
        unsort_idx = np.argsort(sort_idx) # inverse permutation: a[sort][unsort] == a
        x = x_dofs_arr[sort_idx]
        n_e_sorted = np.asarray(n_e)[sort_idx]

        # All quantities below are evaluated in sorted-x (spatial) order.
        T_e = np.interp(x, self.x_init, self.T_e_pres)
        T_i = np.interp(x, self.x_init, self.T_i_pres)
        self.T_e_xdofs = T_e
        G_KBM_grid = self.C_KBM * (self.c_s * self.rho_s**2) / self.a # on psi_Te_eval/x_init grid
        G_KBM = np.interp(x, self.x_init, G_KBM_grid)
        alpha_nodp = np.interp(x, self.x_init, self._alpha_nodp_xinit)

        _pres = n_e_sorted * (T_e + T_i) * 1.60218e-19 # Pa = J/m^3, neglecting neutral pressures and assuming quasi-neutral plasma
        dpdx = np.gradient(_pres, x) # x is now monotonic
        _alpha = alpha_nodp * dpdx # standard Connor-Hastie alpha, sorted-x order

        if average_alpha_pedestal:
            alpha_bar = float(np.mean(_alpha))
            self.alpha_bar_ped = alpha_bar
            gate = alpha_bar > self.alpha_crit
            D_KBM_sorted = np.where(gate, (alpha_bar - self.alpha_crit) * G_KBM, 0.0)
            A_KBM_sorted = np.where(gate, -G_KBM * self.alpha_crit, 0.0)
            B_KBM_sorted = np.where(gate, G_KBM * alpha_nodp, 0.0)
        else:
            gate = _alpha > self.alpha_crit
            D_KBM_sorted = np.where(gate, (_alpha - self.alpha_crit) * G_KBM, 0.0)
            A_KBM_sorted = np.where(gate, -G_KBM * self.alpha_crit, 0.0)
            B_KBM_sorted = np.where(gate, G_KBM * alpha_nodp, 0.0)

        # Outputs assigned to Firedrake Function.dat.data in solve_coupled
        # must be in DOF order: undo the sort.
        self._D_KBM = D_KBM_sorted[unsort_idx]
        self._A_KBM = A_KBM_sorted[unsort_idx]
        self._B_KBM = B_KBM_sorted[unsort_idx]

        # Diagnostic attributes; keep DOF order so they line up with x_dofs.
        self.G_KBM = G_KBM[unsort_idx]
        self.alpha_nodp = alpha_nodp[unsort_idx]
        self.pres = _pres[unsort_idx]

    def _firedrake_mesh_key(self, x_left, mesh_n, fe_degree):
        return (round(float(x_left), 12), int(mesh_n), int(fe_degree))

    def construct_C_ETG(self): # on x_init grid
        DeltaT_e = np.gradient(self.T_e_pres * (1.60218e-19), self.x_init) # gradient in J/m, T_e_pres is in eV
        self.C_ETG = self.De_chie_etg * self.P_tot_e / (self.S_plasma * abs(DeltaT_e)) # evaluated at x_init

    def _ne_on_parent_grid(self, ne_mesh_data):
        """Interpolate FE ``n_e`` DOF values onto ``self.x_init``."""
        x_dofs = self._fd_cache["x_dofs"]
        order = np.argsort(x_dofs)
        return np.interp(
            self.x_init, x_dofs[order], ne_mesh_data[order],
            left=np.nan, right=np.nan,
        )

    def _ensure_firedrake_discretization(self, x_left, mesh_n, fe_degree, force=False):
        """Build or reuse mesh, spaces, and coefficient Functions on the mesh."""
        mesh_key = self._firedrake_mesh_key(x_left, mesh_n, fe_degree)
        if (not force
                and self._fd_cache.get("mesh_key") == mesh_key
                and "mesh" in self._fd_cache):
            return (
                self._fd_cache["mesh"],
                self._fd_cache["V"],
                self._fd_cache["W"],
                self._fd_cache["x_dofs"],
                self._fd_cache["g_fd"],
                self._fd_cache["Si_fd"],
                self._fd_cache["Scx_fd"],
                self._fd_cache["Vcx_fd"],
            )

        mesh = IntervalMesh(mesh_n, x_left, 0.0)
        V = FunctionSpace(mesh, "CG", fe_degree)
        W = MixedFunctionSpace([V, V, V])

        x_coord_func = Function(V).interpolate(SpatialCoordinate(mesh)[0])
        x_dofs = x_coord_func.dat.data.copy() # value of x at each finite element node

        def _make_func(arr, name=""): # interpolate from self.x_init to x_dofs
            f = Function(V, name=name)
            f.dat.data[:] = np.interp(x_dofs, self.x_init, arr)
            return f

        g_fd = _make_func(self.gradr2_fsa, "gradr2_fsa")
        Si_fd = _make_func(self.S_i_pres, "S_i")
        Scx_fd = _make_func(self.S_cx_pres, "S_cx")
        Vcx_fd = _make_func(abs(self.V_cx_pres), "V_cx")

        self._fd_cache.pop("u", None)
        self._fd_cache.pop("u_prev", None)
        self._fd_cache.update({
            "mesh_key": mesh_key,
            "mesh": mesh,
            "V": V,
            "W": W,
            "x_dofs": x_dofs,
            "g_fd": g_fd,
            "Si_fd": Si_fd,
            "Scx_fd": Scx_fd,
            "Vcx_fd": Vcx_fd,
        })
        return mesh, V, W, x_dofs, g_fd, Si_fd, Scx_fd, Vcx_fd

    def _get_or_create_mixed_solution(self, W, force=False):
        """Return cached ``(u, u_prev)`` on ``W``, or allocate new Functions."""
        if (not force
                and self._fd_cache.get("W") is W
                and "u" in self._fd_cache
                and "u_prev" in self._fd_cache):
            return self._fd_cache["u"], self._fd_cache["u_prev"]

        u = Function(W, name="u")
        u_prev = Function(W, name="u_prev")
        self._fd_cache["W"] = W
        self._fd_cache["u"] = u
        self._fd_cache["u_prev"] = u_prev
        return u, u_prev

    def fsa(self,A,flux_surfaces='T_e'):
        """Flux surface average a quantity as defined by ⟨A⟩= int(R^2Adθ)/ int(R^2dθ) in S. Saarelma et al 2023 Nucl. Fusion 63 052002

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        A : array
            2D array of A values at each R_grid, Z_grid.
        flux_surfaces : string
            which flux surface to average over, supporting: T_e, psi_N_pres
             
        """

        R_axis = self.eq['raxis']
        Z_axis = self.eq['zaxis']

        if flux_surfaces == 'T_e':
            psi_N_vals = self.psi_Te_eval
        elif flux_surfaces == 'psi_N_pres':
            psi_N_vals = self.psi_N_pres
        else:
            assert False, 'valid flux_surfaces method must be provided'

        A_clean = np.where(np.isfinite(A), A, 0.0) # replace nan values with 0.0
        A_spl = RectBivariateSpline(self.zgrid, self.rgrid, A_clean)
        fsa_A = np.full(len(psi_N_vals), np.nan)

        fig, ax = plt.subplots()
        for i, psi_val in enumerate(psi_N_vals):
            if psi_val <= 0.01 or psi_val >= 0.99:
                continue

            ax.cla()
            cs = ax.contour(self.rgrid, self.zgrid, self.psi_RZ_N,
                            levels=[psi_val])

            segs = cs.allsegs[0]
            # Closed contour around the axis = the real flux surface
            # (not open SOL/divertor legs or islands).
            seg = self._select_core_contour(segs, R_axis, Z_axis)
            if seg is None:
                continue  # stays NaN; callers interpolate over gaps
            R_c, Z_c = seg[:, 0], seg[:, 1] # R, Z coordinates of the contour

            # poloidal angle measured from the magnetic axis
            # R_c_ax = (((R_c - R_axis)**2) + ((Z_c - Z_axis)**2))**0.5
            # theta_c = np.arcsin( (Z_c - Z_axis) / R_c_ax )
            theta_c = np.arctan2(Z_c - Z_axis, R_c - R_axis) # theta at all points on contour
            theta_c, R_c, Z_c = self._sort_dedup_close_theta(theta_c, R_c, Z_c)

            A_c = A_spl(Z_c, R_c, grid=False)

            den = simpson(R_c**2, theta_c)
            if abs(den) < 1e-30:
                continue
            fsa_A[i] = simpson(R_c**2 * A_c, theta_c) / den

        plt.close(fig)

        return fsa_A

    def psi_rz_expand(self,A,psi_N_A='T_e'):
        """For A defined for each psi_N, expand to all R_grid, Z_grid.

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        A : array
            1D array of A values at each psi_N.
        psi_N_A : array
            1D array of psi_N values at which A is defined.

        Returns
        -------
        A_expanded : array
            2D array of A values at each R_grid, Z_grid.
             
        """

        # psi_N values at which A is defined
        if psi_N_A == 'T_e':
            psi_N_A = self.psi_Te_eval # 1D array of psi_N values at which T_e is evaluated
        else:
            assert False, 'valid psi_N_A method must be provided'

        # Interpolate: psi_N -> A, then evaluate on the 2D psi_N map
        A_interp = interp1d(psi_N_A, A, kind='linear',
                            bounds_error=False, fill_value=np.nan)
        return A_interp(self.psi_RZ_N)

    # ------------------------------------------------------------------
    # Unified solver interface
    # ------------------------------------------------------------------
    #
    # `solve()` and `_build_result_dict()` live on the base class, but the
    # solvers they reach are supplied by the two mixins, so they only work
    # on the assembled class (`src.saarelma_connor.saarelma_connor_api.saarelma_connor`).  Keeping
    # them here keeps the dispatch and the result schema in one place, next
    # to the shared setup the two models both rely on.

    #: Public model name -> the solver entry point it dispatches to.
    _MODEL_ENTRY_POINTS = {
        '3D': 'solve_coupled_nondim',   # coupled n_e / n_FC / n_CX
        '1D': 'solve_sc',               # original single-equation n_e model
    }

    #: Accepted spellings of the model names, lower-cased.
    _MODEL_ALIASES = {
        '3d': '3D', 'coupled': '3D', 'nondim': '3D',
        '1d': '1D', 'sc': '1D', 'simplified': '1D',
    }

    #: Optional per-solver attributes copied into result['diagnostics'].
    #: Read with getattr(..., None), so an attribute a given backend never
    #: sets simply comes back as None.
    _RESULT_DIAGNOSTIC_ATTRS = (
        'picard_info', 'kbm_info', 'grad_bc_info',
        'alpha_bar_ped', 'kbm_gate_on', 'dne_dx_inner_solved',
        'dne_dx_neginf',
        'hat_x_sol', 'hat_ne_sol', 'hat_nFC_sol', 'hat_nCX_sol',
        'E_sol',
    )

    @classmethod
    def _normalise_model(cls, model):
        """Map a user-supplied model name onto ``'1D'`` or ``'3D'``."""
        key = str(model).strip().lower()
        try:
            return cls._MODEL_ALIASES[key]
        except KeyError:
            raise ValueError(
                f"model must be one of '1D' (aliases: 'sc', 'simplified') or "
                f"'3D' (aliases: 'coupled', 'nondim'); got {model!r}."
            ) from None

    def _solve_signature_owners(self, model, backend=None):
        """Methods whose signatures define the kwargs accepted by `model`.

        ``solve_coupled_nondim`` declares every argument of both of its
        backends explicitly, so it is its own authority.  ``solve_sc`` only
        forwards ``**kwargs``, so the authority is the concrete backend --
        and the two differ (``bvp_*`` for scipy, ``fe_degree``/``ksp_*``
        for Firedrake).  With ``backend=None`` every backend of the model
        is returned, which is what the "did you mean the other model?"
        half of the error message needs.
        """
        if model == '3D':
            names = ['solve_coupled_nondim']
        elif backend is None:
            names = ['solve_sc_firedrake', 'solve_sc_scipy']
        else:
            names = {'firedrake': ['solve_sc_firedrake'],
                     'scipy': ['solve_sc_scipy']}.get(str(backend).strip().lower())
            if names is None:
                raise ValueError(
                    f"solver_structure must be 'firedrake' or 'scipy' for "
                    f"model='1D'; got {backend!r}."
                )
        return [getattr(self, name) for name in names if hasattr(self, name)]

    def _accepted_solve_kwargs(self, model, backend=None):
        """Keyword names accepted by `model` (optionally a single backend)."""
        accepted = set()
        for method in self._solve_signature_owners(model, backend):
            for name, param in inspect.signature(method).parameters.items():
                if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  inspect.Parameter.KEYWORD_ONLY):
                    accepted.add(name)
        return accepted

    def _check_solve_kwargs(self, model, backend, kwargs):
        """Reject kwargs the chosen model/backend cannot accept.

        The caller owns the ``SOLVE_KW`` dict it hands to :meth:`solve`, so a
        stray key has to fail loudly here rather than be silently dropped on
        the way into a solver that would then quietly use its default.
        """
        accepted = self._accepted_solve_kwargs(model, backend)
        unknown = sorted(set(kwargs) - accepted)
        if not unknown:
            return

        other = '1D' if model == '3D' else '3D'
        other_accepted = self._accepted_solve_kwargs(other)
        this_model_accepted = self._accepted_solve_kwargs(model)

        detail = []
        for name in unknown:
            if name in this_model_accepted:
                detail.append(f"{name!r} (valid for model={model!r}, but not "
                              f"with solver_structure={backend!r})")
            elif name in other_accepted:
                detail.append(f"{name!r} (belongs to model={other!r})")
            else:
                detail.append(f"{name!r} (not an argument of any solver)")

        raise TypeError(
            f"solve(model={model!r}, solver_structure={backend!r}) got "
            f"unsupported keyword argument(s): {', '.join(detail)}.\n"
            f"Accepted here: {', '.join(sorted(accepted))}."
        )

    def solve_scmodel(self, model='3D', **kwargs):
        """Solve the pedestal problem with the chosen physics model.

        Parameters
        ----------
        model : {'3D', '1D'}
            ``'3D'`` (aliases ``'coupled'``, ``'nondim'``) solves the
            coupled three-equation system for n_e, n_FC and n_CX via
            :meth:`solve_coupled_nondim`.  ``'1D'`` (aliases ``'sc'``,
            ``'simplified'``) solves the original single-equation
            Saarelma-Connor model for n_e via :meth:`solve_sc`.

            Note this counts *equations*, not spatial dimensions: both
            models are one-dimensional in space.
        solver_structure : {'firedrake', 'scipy'}, optional
            Discretisation backend, default ``'firedrake'``.  Accepted for
            both models, so a single kwargs dict can drive either;
            ``implementation`` is accepted as a synonym (it is the name
            :meth:`solve_sc` uses internally).
        **kwargs
            Passed to the chosen solver.  Arguments the target does not
            accept raise ``TypeError`` naming the offending key rather than
            being dropped.

        Returns
        -------
        dict
            See :meth:`_build_result_dict`.
        """
        model = self._normalise_model(model)
        kwargs = dict(kwargs)

        backend = kwargs.pop('solver_structure', None)
        if backend is None:
            raise ValueError("need to define solver structure")

        self._check_solve_kwargs(model, backend, kwargs)

        entry_point = self._MODEL_ENTRY_POINTS[model]
        try:
            solver = getattr(self, entry_point)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__} has no {entry_point!r}; model="
                f"{model!r} needs the corresponding solver mixin. Build the "
                f"model from src.saarelma_connor.saarelma_connor_api.saarelma_connor."
            ) from None

        backend_kw = 'solver_structure' if model == '3D' else 'implementation'
        return solver(**{backend_kw: backend}, **kwargs)

    def _build_result_dict(self, model, solver_structure):
        """Assemble the common result dictionary returned by every solver.

        Both models fill the same keys, so callers parse one schema
        regardless of which solver ran.

        Parameters
        ----------
        model : {'1D', '3D'}
            Which model produced the solution currently on the instance.
        solver_structure : {'firedrake', 'scipy'}
            Which backend produced it.

        Returns
        -------
        dict
            ``model``, ``solver_structure`` : str

            ``x`` : ndarray
                Radial grid (m), zero at the separatrix.
            ``ne``, ``dne_dx`` : ndarray
                Electron density (m^-3) and its gradient (m^-4) on ``x``.
            ``nFC``, ``nCX`` : ndarray
                Franck-Condon and charge-exchange neutral densities
                (m^-3) on ``x``.  Solved as unknowns by the 3D model;
                derived from the converged n_e by the 1D model.
            ``T_e_pres``, ``psi_N_pres`` : ndarray
                Electron temperature and normalised flux of the input
                profiles, passed through for downstream convenience.
            ``diagnostics`` : dict
                Per-solver extras (Picard/KBM/shooting info, the
                non-dimensional profiles); entries the backend never set
                are ``None``.
        """
        x = np.asarray(self.x_sol, dtype=float)
        ne = np.asarray(self.ne_sol, dtype=float)

        # Branch on `model` rather than on which attributes happen to
        # exist: solving 1D on an instance that previously ran 3D would
        # otherwise pick up that run's stale nFC_sol / nCX_sol.
        if model == '3D':
            nFC = np.asarray(self.nFC_sol, dtype=float)
            nCX = np.asarray(self.nCX_sol, dtype=float)
            # The coupled solver does not carry the gradient as an unknown.
            dne_dx = np.gradient(ne, x)
        else:
            # n_FC and n_CX are not unknowns of the single-equation model;
            # they follow from the converged n_e via Saarelma Eqs. (11)-(12).
            # Not compute_post_solve_SC_neutrals(): it assumes the solution
            # grid matches x_init and omits De_chie_etg from D_ETG.
            nFC, nCX = self._post_solve_neutrals_sc()
            nFC = np.asarray(nFC, dtype=float)
            nCX = np.asarray(nCX, dtype=float)
            dne_dx = np.asarray(self.dne_dx_sol, dtype=float)

        diagnostics = {name: getattr(self, name, None)
                       for name in self._RESULT_DIAGNOSTIC_ATTRS}

        self.result = {
            'model': model,
            'solver_structure': str(solver_structure),
            'x': x,
            'ne': ne,
            'dne_dx': dne_dx,
            'nFC': nFC,
            'nCX': nCX,
            'T_e_pres': np.asarray(self.T_e_pres, dtype=float),
            'psi_N_pres': np.asarray(self.psi_N_pres, dtype=float),
            'diagnostics': diagnostics,
        }
        return self.result


#: Backwards-compatible alias for the pre-refactor base-class name.  The
#: full model (base + both solver mixins) is src.saarelma_connor.saarelma_connor_api.saarelma_connor.
saarelma_connor_base = SaarelmaConnorBase