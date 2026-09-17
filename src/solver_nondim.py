"""Non-dimensional Firedrake solver for the coupled three-equation
Saarelma-Connor pedestal model.

Implements the foundational non-dimensionalisation derived in
Appendix A.8 of docs/Labbate_APAM9301_Final_06152026.tex:

    L  = |x_inner|                length scale (m)
    n0 = ne_x0                    density scale (m^-3)
    T0 = T_e(0) + T_i(0)          energy scale (J)
    tau = 1 / (n0 * S0)           time scale (s),   S0 = S_i(0)

with derived scales
    [V]_0 = L / tau = L * n0 * S0          velocity (m/s)
    [D]_0 = L^2 / tau = L^2 * n0 * S0      diffusivity (m^2/s)
    [p]_0 = n0 * T0                        pressure (Pa).

Hatted variables:
    hat_x = x / L,                     hat_x in [-1, 0]
    hat_n_e = n_e / n0,                hat_n_e(0) = 1
    hat_n_FC = <n_FC> / n0,            hat_n_FC(0) = nFC_x0/n0
    hat_n_CX = <n_CX> / n0             hat_n_CX(0) = nCX_x0/n0
    hat_T = (T_e + T_i) / T0
    hat_S_i = S_i / S0,   hat_S_CX = S_CX / S0
    hat_V_FC = |V_FC| / [V]_0,   hat_V_CX = |V_CX| / [V]_0
    hat_g = <|grad r|^2>           (already dimensionless)

Dimensionless transport coefficients:
    hat_C_ETG = C_ETG / (L^2 * n0^2 * S0)
    hat_D_NEO = D_NEO / (L^2 * n0 * S0)
    hat_A_KBM = A_KBM / (L^2 * n0 * S0)
    hat_B_KBM = B_KBM * T0 / (L^3 * S0)

The dimensionless residuals (Eqs. (eq:weak-hat-A8),
(eq:weak-hat-nFC-A8), (eq:weak-hat-nCX-A8) in App. A.8):

    F1_hat: int_{-1}^{0} hat_g [ hat_C_ETG/hat_n_e * hat_n_e'
              + hat_A_KBM * hat_n_e'
              + hat_B_KBM * hat_T * (hat_n_e')^2
              + hat_B_KBM * hat_n_e * hat_dT_dxhat * hat_n_e'
              + hat_D_NEO * hat_n_e' ] v_e' dxhat
            - int_{-1}^{0} hat_n_e * hat_S_i * (hat_n_FC + hat_n_CX) v_e dxhat
            + delta_N * hat_g(-1) * hat_D|_{-1} * hat_dne_dx_inner * v_e(-1) = 0

    F2_hat: int_{-1}^{0} [ hat_V_FC * d/dxhat[ f_FC * hat_g * hat_n_FC ]
              - hat_n_e * (hat_S_i + hat_S_CX) * hat_n_FC ] v_F dxhat = 0

    F3_hat: int_{-1}^{0} [ d/dxhat[ hat_V_CX * f_CX * hat_g * hat_n_CX ]
              - hat_n_e * (hat_S_i * hat_n_CX - 0.5 * hat_S_CX * hat_n_FC)
              ] v_C dxhat = 0

Placement of the n_e gradient boundary condition
================================================
``ne_grad_bc_loc`` selects which end of the domain carries the
prescribed dn_e/dx.

``"inner"`` (default, unchanged behaviour)
    Dirichlet n_e(0) = ne_x0 at the separatrix, and at hat_x = -1
    either Dirichlet n_e(x_inner) or the Neumann flux above
    (``ne_inner_bc``).  One condition at each end.

``"outer"``
    Both conditions sit at the separatrix -- Dirichlet n_e(0) = ne_x0
    *and* Neumann dn_e/dx|_{x=0} = dne_dx_outer -- and the inner
    boundary is left completely free.  This is the boundary-condition
    set of Saarelma et al. 2023 Sec. 2.3, where the separatrix density
    and its gradient (their Eq. (20), dn_e/dx|_0 = -n_e(0) /
    sqrt(D_SOL tau_par)) are the specified data and nothing is imposed
    at the pedestal top.

    A Galerkin discretisation cannot impose both directly: the strong
    Dirichlet condition at boundary id 2 makes v_e(0) = 0, so a ds(2)
    flux term vanishes identically.  The equivalent well-posed
    statement is that the *inner* flux is an unknown Lagrange
    multiplier fixed by the separatrix-gradient constraint, and that is
    what is solved here -- by secant iteration on the ds(1) slope
    ``hat_dne_dx_inner`` until hat_n_e'(0) matches the prescribed
    value.  Shooting on that one scalar (rather than enlarging the
    mixed space by a real-space multiplier) leaves the function space,
    the Picard loop and the analytic-neutral path untouched.
    Diagnostics land in ``self.grad_bc_info``.

Solver entry point: ``saarelma_connor.solve_coupled_nondim(...)``, or
equivalently ``saarelma_connor.solve(model='3D', ...)``.  Both return the
common result dictionary described in
:meth:`src.solver.SaarelmaConnorBase._build_result_dict`.

This module contributes :class:`NondimSolverMixin` to the single public
class :class:`src.solver_api.saarelma_connor`; all equilibrium / kinetic /
cross-section setup comes from :class:`src.solver.SaarelmaConnorBase`.
"""

import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp
from scipy.interpolate import interp1d

try:
    from firedrake import (
        IntervalMesh, FunctionSpace, MixedFunctionSpace, Function,
        TestFunctions, Constant, DirichletBC,
        dx, ds, split, solve, assemble, SpatialCoordinate,
        tanh as fd_tanh,
    )
    _FIREDRAKE_AVAILABLE = True
    _FIREDRAKE_IMPORT_ERR = None
except Exception as _firedrake_import_err:
    _FIREDRAKE_AVAILABLE = False
    _FIREDRAKE_IMPORT_ERR = _firedrake_import_err

from src import bc_ig_helpers as bcig


# Conversion constant (eV -> J)
_EV2J = 1.60218e-19


class NondimSolverMixin:
    """Non-dimensional coupled three-equation solver ("3D" model).

    A mixin over :class:`src.solver.SaarelmaConnorBase`; it is combined
    with the base and :class:`src.solver_sc.SCSolverMixin` into the single
    public :class:`src.solver_api.saarelma_connor`.  It defines no
    ``__init__`` -- every attribute it reads is set by the base
    constructor -- and it overrides nothing, so it only ever adds the
    ``*_nondim`` methods below.

    The solver assembles and solves the dimensionless residuals derived in
    Appendix~A.8 of the paper rather than their SI counterparts.

    Reference scales are computed lazily inside :meth:`solve_coupled`
    (after ``setup_solver_grids`` and any inner-boundary search are done).
    They are stored as instance attributes:

        self._L_nd, self._n0_nd, self._T0_nd, self._S0_nd  : 4 free scales
        self._tau_nd, self._V0_nd, self._D0_nd             : derived scales

    After convergence the SI-unit profiles are recovered and stored on
    the same attributes (``self.x_sol``, ``self.ne_sol``, ``self.nFC_sol``,
    ``self.nCX_sol``) as the parent solver, so all downstream code
    (plotting, EPED feed, etc.) keeps working without modification.
    The non-dimensional profiles are also kept available as
    ``self.hat_x_sol``, ``self.hat_ne_sol``, ``self.hat_nFC_sol``,
    ``self.hat_nCX_sol`` for diagnostics.
    """

    # ------------------------------------------------------------------
    # Reference scales
    # ------------------------------------------------------------------

    def _set_nondim_scales(self, verbose=False):
        """Compute and cache the four free reference scales of
        Eq.~(eq:nondim-scales-A8) plus the derived scales.

        Convention (see App.~A.8):

            L  = |x_inner|              (m)
            n0 = ne_x0                  (m^-3)         -- separatrix value
            T0 = (T_e(0)+T_i(0)) * J/eV (J)            -- separatrix value
            S0 = S_i(0)                 (m^3/s)        -- separatrix value
            tau    = 1 / (n0 * S0)      (s)
            [V]_0  = L / tau            (m/s)
            [D]_0  = L^2 / tau          (m^2/s)

        """
        if not hasattr(self, "x_inner") or self.x_inner is None:
            raise RuntimeError(
                "x_inner is not set.  Call setup_solver_grids and/or "
                "find_inner_boundary before _set_nondim_scales."
            )
        if not hasattr(self, "ne_x0"):
            raise RuntimeError("ne_x0 is not set on this instance.")

        self._L_nd  = float(abs(self.x_inner))
        self._n0_nd = float(self.ne_x0)

        # Te + Ti at separatrix.  T_e_pres / T_i_pres are stored in eV
        # (cf. solver.py L196--L200), and the separatrix is index -1
        # because x_init = r_psi - r_psi[-1] (cf. solver.py L781).
        T_e0_eV = float(self.T_e_pres[-1])
        T_i0_eV = float(self.T_i_pres[-1])
        self._T0_nd = (T_e0_eV + T_i0_eV) * _EV2J  # in J

        # S_i at separatrix on the pressure grid (built in
        # setup_solver_grids -> S_i_pres on psi_N_pres, last index = sep.)
        S0 = float(self.S_i_pres[-1])
        if not np.isfinite(S0) or S0 <= 0.0:
            raise RuntimeError(
                f"S_i(0) = {S0!r} m^3/s is not a usable separatrix "
                "ionisation rate; cannot non-dimensionalise."
            )
        self._S0_nd = S0

        # Derived scales
        self._tau_nd = 1.0 / (self._n0_nd * self._S0_nd)
        self._V0_nd  = self._L_nd / self._tau_nd
        self._D0_nd  = (self._L_nd ** 2) / self._tau_nd

        # Pre-compute scales that show up repeatedly inside the residuals
        # (purely for readability of the rest of the class):
        #     ETG_scale = L^2 * n0^2 * S0   (= n0 * [D]_0)
        #     KBMB_scale = L^3 * S0 / T0    (= [B_KBM])
        self._ETG_scale_nd  = (self._L_nd ** 2) * (self._n0_nd ** 2) * self._S0_nd
        self._KBMB_scale_nd = (self._L_nd ** 3) * self._S0_nd / self._T0_nd

        if verbose:
            print(
                "[nondim] reference scales:\n"
                f"  L  = {self._L_nd:.4e} m\n"
                f"  n0 = {self._n0_nd:.4e} m^-3\n"
                f"  T0 = {self._T0_nd:.4e} J  ({(T_e0_eV + T_i0_eV):.3f} eV)\n"
                f"  S0 = {self._S0_nd:.4e} m^3/s\n"
                f"  tau    = {self._tau_nd:.4e} s\n"
                f"  [V]_0  = {self._V0_nd:.4e} m/s\n"
                f"  [D]_0  = {self._D0_nd:.4e} m^2/s"
            )

    # ------------------------------------------------------------------
    # SI <-> non-dim helpers
    # ------------------------------------------------------------------

    def _to_hat_x(self, x_si):
        """Map an SI x array (m) onto hat_x = x / L."""
        return np.asarray(x_si) / self._L_nd

    def _from_hat_x(self, hat_x):
        """Map hat_x back to SI x = L * hat_x."""
        return np.asarray(hat_x) * self._L_nd

    def _interp_si_to_hat(self, hat_x_dofs, si_array_on_xinit):
        """Evaluate an SI quantity defined on ``self.x_init`` (sorted)
        at the dimensionless mesh DOFs ``hat_x_dofs``.

        Returns the SI value at L * hat_x_dofs (i.e. *no* rescaling).
        Callers rescale to hat-units as needed.

        This basically just interpolates the SI array to the x_dofs grid.
        Because x_dofs is the same size as hat_x_dofs, once you are on x_dofs you don't need to interpolate again.
        """
        x_si = self._from_hat_x(hat_x_dofs)
        return np.interp(x_si, self.x_init, np.asarray(si_array_on_xinit)) # interpolates x_init to x_dofs grid

    # ------------------------------------------------------------------
    # Mesh and frozen non-dim coefficient Functions
    # ------------------------------------------------------------------

    def _ensure_firedrake_discretization_nondim(self, mesh_n, fe_degree, force=False):
        """Build (or reuse) the dimensionless mesh on hat_x in [-1, 0]
        and the equilibrium-only non-dim coefficient Functions.

        Cache keys live alongside the parent's ``self._fd_cache`` keys
        but use a ``_nd`` suffix to avoid colliding with any cached
        SI-version state.

        Returns
        -------
        mesh, V, W, hat_x_dofs, hat_g_fd, hat_Si_fd, hat_Scx_fd, hat_Vcx_fd
        """
        if not _FIREDRAKE_AVAILABLE:
            raise ImportError(
                "Firedrake is not available in this environment.  "
                "Install Firedrake (https://www.firedrakeproject.org/) "
                "to use this solver.  Original import error:\n"
                f"  {_FIREDRAKE_IMPORT_ERR}"
            )

        mesh_key = ("nd", int(mesh_n), int(fe_degree))
        if (not force
                and self._fd_cache.get("mesh_key_nd") == mesh_key
                and "mesh_nd" in self._fd_cache):
            return (
                self._fd_cache["mesh_nd"],
                self._fd_cache["V_nd"],
                self._fd_cache["W_nd"],
                self._fd_cache["hat_x_dofs"],
                self._fd_cache["hat_g_fd"],
                self._fd_cache["hat_Si_fd"],
                self._fd_cache["hat_Scx_fd"],
                self._fd_cache["hat_Vcx_fd"],
            )

        mesh = IntervalMesh(int(mesh_n), -1.0, 0.0)
        V = FunctionSpace(mesh, "CG", int(fe_degree))
        W = MixedFunctionSpace([V, V, V])

        x_coord_func = Function(V).interpolate(SpatialCoordinate(mesh)[0])
        hat_x_dofs = x_coord_func.dat.data.copy()

        def _make_func(arr, name=""):
            f = Function(V, name=name)
            f.dat.data[:] = arr
            return f

        # hat_g(hat_x) = <|grad r|^2>(L*hat_x)  -- already dimensionless
        hat_g_arr = self._interp_si_to_hat(hat_x_dofs, self.gradr2_fsa) # move gradr2_fsa to x_dofs grid
        hat_g_fd = _make_func(hat_g_arr, "hat_g")

        # hat_S_i(hat_x), hat_S_CX(hat_x): rate coefficients / S0
        hat_Si_arr  = self._interp_si_to_hat(hat_x_dofs, self.S_i_pres)  / self._S0_nd
        hat_Scx_arr = self._interp_si_to_hat(hat_x_dofs, self.S_cx_pres) / self._S0_nd
        hat_Si_fd  = _make_func(hat_Si_arr,  "hat_S_i")
        hat_Scx_fd = _make_func(hat_Scx_arr, "hat_S_cx")

        # hat_V_CX(hat_x) = |V_CX|(L*hat_x) / [V]_0
        hat_Vcx_arr = self._interp_si_to_hat(hat_x_dofs, np.abs(self.V_cx_pres)) / self._V0_nd
        hat_Vcx_fd  = _make_func(hat_Vcx_arr, "hat_V_cx")

        # Drop any mixed solution that was cached against a previous mesh.
        self._fd_cache.pop("u_nd", None)
        self._fd_cache.pop("u_prev_nd", None)
        self._fd_cache.update({
            "mesh_key_nd": mesh_key,
            "mesh_nd": mesh,
            "V_nd": V,
            "W_nd": W,
            "hat_x_dofs": hat_x_dofs,
            "hat_g_fd": hat_g_fd,
            "hat_Si_fd": hat_Si_fd,
            "hat_Scx_fd": hat_Scx_fd,
            "hat_Vcx_fd": hat_Vcx_fd,
        })
        return mesh, V, W, hat_x_dofs, hat_g_fd, hat_Si_fd, hat_Scx_fd, hat_Vcx_fd

    def _get_or_create_mixed_solution_nondim(self, W, force=False):
        """Return cached ``(hat_u, hat_u_prev)`` or allocate them."""
        if (not force
                and self._fd_cache.get("W_nd") is W
                and "u_nd" in self._fd_cache
                and "u_prev_nd" in self._fd_cache):
            return self._fd_cache["u_nd"], self._fd_cache["u_prev_nd"]

        u = Function(W, name="hat_u")
        u_prev = Function(W, name="hat_u_prev")
        self._fd_cache["W_nd"] = W
        self._fd_cache["u_nd"] = u
        self._fd_cache["u_prev_nd"] = u_prev
        return u, u_prev

    # ------------------------------------------------------------------
    # Pressure / KBM coefficients in non-dim units
    # ------------------------------------------------------------------

    def calc_pressure_quantities_nondim(self, hat_n_e,
                                        gate_mode=None,
                                        hat_x_dofs=None):
        """Non-dim version of :meth:`saarelma_connor.calc_pressure_quantities`.

        Computes the pedestal-averaged Connor-Hastie alpha in physical
        (SI) units (because alpha is a physical quantity defined by an
        SI integral) and then rescales the resulting transport
        contributions A_KBM, B_KBM, D_KBM into their dimensionless
        counterparts hat_A_KBM, hat_B_KBM, hat_D_KBM.

        Parameters
        ----------
        hat_n_e : array_like, shape (n_dofs,)
            Dimensionless electron density at the mesh DOFs (DOF order,
            not necessarily monotonic in hat_x).
        gate_mode : {None, "average", "majority", "local"}
            How the KBM on/off gate is decided:

            ``"average"``
                Gate on the pedestal-averaged alpha:
                ``gate = mean(alpha) > alpha_crit`` (single on/off for
                the whole grid).  When on, Saarelma et al. (2023) Eq. 25
                diffusivity is used on the whole grid:
                ``D_KBM = (alpha_bar - alpha_crit) * G_KBM``.
            ``"majority"``
                Gate on a majority vote of the local alpha: on for the
                whole grid iff more than half of the pedestal grid
                points have alpha > alpha_crit.  When on, the local
                (A, B) KBM coefficient structure is applied at every
                point (no pointwise gate).
            ``"local"``
                Pointwise gate (legacy behaviour): each grid point is
                gated by its own local alpha.

        Sets
        ----
        self._hat_A_KBM, self._hat_B_KBM, self._hat_D_KBM : ndarray
            Dimensionless KBM coefficients in DOF order.
        self.alpha_bar_ped : float
            Pedestal-averaged alpha (same definition as the parent).
        self.alpha_frac_above : float
            Fraction of pedestal grid points with alpha > alpha_crit.
        self.kbm_gate_on : bool or ndarray
            Gate state: a scalar bool for "average"/"majority", a
            pointwise bool array (DOF order) for "local".
        self.alpha_local_ped : ndarray
            Local Connor-Hastie alpha in DOF order (diagnostic).
        """
        # The Firedrake path leaves this None and uses the cached FE DOFs;
        # the scipy path passes its own (ascending) collocation grid.
        if hat_x_dofs is None:
            hat_x_dofs = self._fd_cache["hat_x_dofs"]
        hat_x_dofs = np.asarray(hat_x_dofs, dtype=float)
        sort_idx = np.argsort(hat_x_dofs)
        unsort_idx = np.argsort(sort_idx)
        hat_x = hat_x_dofs[sort_idx]
        hat_n_e_sorted = np.asarray(hat_n_e)[sort_idx]

        # Reconstruct SI quantities on the sorted hat-x grid for the
        # alpha calculation.
        x_si = self._from_hat_x(hat_x)              # m
        n_e_sorted = hat_n_e_sorted * self._n0_nd   # m^-3
        T_e_eV = np.interp(x_si, self.x_init, self.T_e_pres)        # eV
        T_i_eV = np.interp(x_si, self.x_init, self.T_i_pres)        # eV
        self.T_e_xdofs = T_e_eV                                     # diagnostic
        G_KBM_grid = self.C_KBM * (self.c_s * self.rho_s ** 2) / self.a
        G_KBM = np.interp(x_si, self.x_init, G_KBM_grid)            # m^2/s
        alpha_nodp = np.interp(x_si, self.x_init, self._alpha_nodp_xinit)

        # Connor-Hastie alpha = alpha_nodp * dp/dx (SI)
        _pres = n_e_sorted * (T_e_eV + T_i_eV) * _EV2J # Pa, neglecting neutral pressures and assuming quasi-neutral plasma
        dpdx = np.gradient(_pres, x_si)                # Pa/m
        _alpha = alpha_nodp * dpdx                     # dimensionless

        # Resolve the gate mode
        gate_mode = str(gate_mode).lower()
        if gate_mode not in ("average", "majority", "local"):
            raise ValueError(
                f"gate_mode must be 'average', 'majority', or 'local', "
                f"got {gate_mode!r}."
            )

        alpha_bar = float(np.mean(_alpha))
        self.alpha_bar_ped = alpha_bar
        self.alpha_frac_above = float(np.mean(_alpha > self.alpha_crit))
        self.alpha_local_ped = _alpha[unsort_idx]

        if gate_mode == "average":
            # Saarelma et al. (2023) Eqs. 24--25: whole-grid gate on
            # alpha_bar, diffusivity locked to (alpha_bar - alpha_crit).
            gate = alpha_bar > self.alpha_crit
            self.kbm_gate_on = bool(gate)
            D_KBM_si = np.where(gate, (alpha_bar - self.alpha_crit) * G_KBM, 0.0)
            # A/B kept as diagnostics / unused by Picard-"average"
            # (that path freezes D into the A-slot with B = 0).
            A_KBM_si = np.where(gate, -G_KBM * self.alpha_crit, 0.0)
            B_KBM_si = np.where(gate, G_KBM * alpha_nodp, 0.0)
        elif gate_mode == "majority":
            # Whole-grid on/off from a majority vote of local alpha;
            # when on, freeze the local A/B structure everywhere, does NOT use a global gate like "average".
            gate = self.alpha_frac_above > 0.5
            self.kbm_gate_on = bool(gate)
            D_KBM_si = np.where(gate, (_alpha - self.alpha_crit) * G_KBM, 0.0)
            A_KBM_si = np.where(gate, -G_KBM * self.alpha_crit, 0.0)
            B_KBM_si = np.where(gate, G_KBM * alpha_nodp, 0.0)
        else:  # local (pointwise legacy gate)
            gate = _alpha > self.alpha_crit
            self.kbm_gate_on = gate[unsort_idx]
            D_KBM_si = np.where(gate, (_alpha - self.alpha_crit) * G_KBM, 0.0)
            A_KBM_si = np.where(gate, -G_KBM * self.alpha_crit, 0.0)
            B_KBM_si = np.where(gate, G_KBM * alpha_nodp, 0.0)

        # Rescale to hat units (App. A.8 Eqs. (eq:hat-A-KBM), (eq:hat-B-KBM)):
        #   hat_A = A / [D]_0
        #   hat_B = B * T0 / (L^3 * S0) = B / KBMB_scale
        #   hat_D_KBM = D_KBM / [D]_0
        hat_A_KBM_sorted = A_KBM_si / self._D0_nd
        hat_B_KBM_sorted = B_KBM_si / self._KBMB_scale_nd
        hat_D_KBM_sorted = D_KBM_si / self._D0_nd
        self.D_KBM_si = D_KBM_si[unsort_idx]

        self._hat_A_KBM = hat_A_KBM_sorted[unsort_idx]
        self._hat_B_KBM = hat_B_KBM_sorted[unsort_idx]
        self._hat_D_KBM = hat_D_KBM_sorted[unsort_idx]

        # Diagnostics in DOF order (mirrors the parent class)
        self.G_KBM = G_KBM[unsort_idx]
        self.alpha_nodp = alpha_nodp[unsort_idx]
        self.pres = _pres[unsort_idx]

    # ------------------------------------------------------------------
    # Semi-analytic neutral solve (integrating factor on a fine sub-grid)
    # ------------------------------------------------------------------

    def _solve_neutrals_analytic_nondim(self, hat_ne_dofs, n_sub=4001):
        """Solve the FC and CX neutral equations exactly for a *given*
        electron density, on a dense sub-grid that resolves the neutral
        ionisation boundary layer even when the FE mesh does not.

        Both neutral equations are first-order linear ODEs in x once
        ``n_e(x)`` is frozen, so they admit integrating-factor solutions
        that are positive by construction (no CG oscillations, no
        negative densities).  This is the same mathematics as the
        ``nFC_ic='solve'`` / ``nCX_ic='solve'`` initial-guess branches,
        but evaluated on ``n_sub`` points instead of the FE DOFs and
        including the |grad r|^2 / form-factor variation exactly.

        FC neutrals (Saarelma-Connor Eq. 9):

            |V_FC| d/dx [ f_FC g n_FC ] = n_e (S_i + S_CX) n_FC

        With u = f_FC g n_FC and tau = -x (inward distance >= 0):

            u(tau) = u(0) * exp( -int_0^tau  n_e (S_i+S_CX)
                                             / (|V_FC| f_FC g)  dtau' )

        CX neutrals (Eq. 10), with w = |V_CX| f_CX g n_CX:

            dw/dtau = -P(tau) w + Q(tau),
            P = n_e S_i / (|V_CX| f_CX g),
            Q = (1/2) n_e S_CX n_FC,

        integrated with a per-step exponential integrator
        (w_{k+1} = w_k e^{-P dtau} + (Q/P)(1 - e^{-P dtau})), which is
        unconditionally stable and positivity-preserving regardless of
        how stiff P dtau is.

        Parameters
        ----------
        hat_ne_dofs : array_like
            Dimensionless n_e at the FE mesh DOFs (DOF order).
        n_sub : int, default 4001
            Number of points of the dense integration sub-grid on
            [x_inner, 0].  Choose so that the sub-grid spacing is well
            below the minimum neutral penetration depth
            lambda = |V_FC| / (n_e (S_i + S_CX)).

        Returns
        -------
        hat_nFC_dofs, hat_nCX_dofs : ndarray
            Dimensionless neutral densities at the FE mesh DOFs (DOF
            order), interpolated in log-space from the sub-grid so the
            boundary-layer decay is preserved.
        """
        hat_x_dofs = self._fd_cache["hat_x_dofs"]
        x_dofs = self._from_hat_x(hat_x_dofs)          # m, DOF order
        order = np.argsort(x_dofs)
        x_sorted = x_dofs[order]                        # ascending [-L, 0]
        ne_sorted = np.asarray(hat_ne_dofs)[order] * self._n0_nd  # m^-3

        # Dense sub-grid, descending from the separatrix (x = 0) inward.
        x_fine = np.linspace(0.0, x_sorted[0], int(n_sub))
        tau = -x_fine                                   # >= 0, ascending
        ne_f = np.clip(np.interp(x_fine, x_sorted, ne_sorted), 0.0, None)
        Si_f = np.interp(x_fine, self.x_init, self.S_i_pres)
        Scx_f = np.interp(x_fine, self.x_init, self.S_cx_pres)
        g_f = np.interp(x_fine, self.x_init, self.gradr2_fsa)
        fFC_f = np.interp(x_fine, self.x_init, self.fFC)
        fCX_f = np.interp(x_fine, self.x_init, self.fCX)
        Vcx_f = np.interp(x_fine, self.x_init, np.abs(self.V_cx_pres))
        Vfc = abs(self.V_FC)

        # ---- FC neutrals: closed-form integrating factor ----
        # I_FC(tau) = int_0^tau n_e (S_i + S_CX) / (Vfc f_FC g) dtau' >= 0
        I_FC = cumulative_trapezoid(
            ne_f * (Si_f + Scx_f) / (Vfc * fFC_f * g_f), tau, initial=0.0,
        )
        ln_u0 = np.log(fFC_f[0] * g_f[0] * self.nFC_x0)
        ln_nFC_f = ln_u0 - I_FC - np.log(fFC_f * g_f)

        # ---- CX neutrals: exponential-integrator march in tau ----
        P_f = ne_f * Si_f / (Vcx_f * fCX_f * g_f)       # 1/m
        Q_f = 0.5 * ne_f * Scx_f * np.exp(ln_nFC_f)     # source, w-units/m
        w = np.empty_like(tau)
        w[0] = Vcx_f[0] * fCX_f[0] * g_f[0] * self.nCX_x0
        dtau = np.diff(tau)
        Pm = 0.5 * (P_f[:-1] + P_f[1:])
        Qm = 0.5 * (Q_f[:-1] + Q_f[1:])
        for k in range(len(dtau)):
            a = Pm[k] * dtau[k]
            if a > 1e-12:
                E = np.exp(-a)
                w[k + 1] = w[k] * E + (Qm[k] / Pm[k]) * (1.0 - E)
            else:
                w[k + 1] = w[k] + Qm[k] * dtau[k]
        ln_nCX_f = (
            np.log(np.clip(w, 1e-300, None)) - np.log(Vcx_f * fCX_f * g_f)
        )

        # ---- Back to the FE DOFs (log-space interpolation) ----
        tau_dofs = -x_dofs                              # DOF order
        # np.interp needs ascending xp: tau is ascending by construction.
        nFC_dofs = np.exp(np.interp(tau_dofs, tau, ln_nFC_f))
        nCX_dofs = np.exp(np.interp(tau_dofs, tau, ln_nCX_f))

        # Diagnostics (SI, sub-grid)
        self.x_neutrals_analytic = x_fine
        self.nFC_analytic = np.exp(ln_nFC_f)
        self.nCX_analytic = np.exp(ln_nCX_f)

        return nFC_dofs / self._n0_nd, nCX_dofs / self._n0_nd

    # ------------------------------------------------------------------
    # Weak forms
    # ------------------------------------------------------------------

    def _kbm_inline_terms(
        self, hat_ne, hat_T_fd, hat_dT_dx_fd,
        hat_alpha_nodp_fd, hat_G_KBM_fd,
        alpha_crit_c, gate_eps_c,
        boundary=False, hat_dne_dx_inner_c=None,
    ):
        """Build the inline (trial-dependent) UFL expressions for the
        Connor-Hastie ``alpha``, the smoothed KBM gate, and the
        dimensionless KBM coefficients ``hat_A_KBM`` and ``hat_B_KBM``.

        Mathematics (see App.~A.7--A.8 of docs/Labbate_APAM9301_Final):

            alpha(hat_x, hat_n_e)
                = hat_alpha_nodp(hat_x) *  d/dhat_x [ hat_n_e * hat_T ]
                = hat_alpha_nodp * ( hat_T * hat_n_e' + hat_n_e * hat_T' ),

            gate_eps(z)
                = 0.5 * (1 + tanh(z / gate_eps))    ~ Heaviside(z),

            hat_A_KBM(hat_n_e) = - gate_eps(alpha - alpha_crit) * alpha_crit * hat_G_KBM,
            hat_B_KBM(hat_n_e) = + gate_eps(alpha - alpha_crit) * hat_alpha_nodp * hat_G_KBM.

        These are full UFL expressions so SNES (Newton) automatically
        captures their dependence on the trial ``hat_n_e``.

        Parameters
        ----------
        boundary : bool, default False
            If True, use the prescribed inner-boundary slope
            ``hat_dne_dx_inner_c`` for hat_n_e' inside ``alpha`` (this
            is needed in the Neumann-flux ds(1) contribution to F1).
            Otherwise use the volume trial derivative ``hat_ne.dx(0)``.
        """
        
        if boundary:
            ne_dx_here = hat_dne_dx_inner_c
        else:
            ne_dx_here = hat_ne.dx(0)
        alpha_ufl = hat_alpha_nodp_fd * (
            hat_T_fd * ne_dx_here + hat_ne * hat_dT_dx_fd
        )
        gate_ufl = 0.5 * (1.0 + fd_tanh((alpha_ufl - alpha_crit_c) / gate_eps_c)) # smoothed Heaviside gate
        hat_A_KBM_ufl = -gate_ufl * alpha_crit_c * hat_G_KBM_fd
        hat_B_KBM_ufl =  gate_ufl * hat_alpha_nodp_fd * hat_G_KBM_fd
        return alpha_ufl, gate_ufl, hat_A_KBM_ufl, hat_B_KBM_ufl

    def _build_f1_weak_form_nondim(
            self, hat_ne, hat_g_fd,
            hat_C_ETG_fd, hat_D_NEO_fd,
            hat_A_KBM_term, hat_B_KBM_term,
            hat_T_fd, hat_dT_dx_fd,
            hat_Si_fd, hat_nFC, hat_nCX,
            v_e, hat_dne_dx_inner_c,
            hat_A_KBM_bc_term=None, hat_B_KBM_bc_term=None):
        """Dimensionless n_e weak form -- Eq. (eq:weak-hat-A8) of App. A.8.

        Integrals are over hat_x in [-1, 0] (the Firedrake mesh).  The
        Neumann boundary contribution at hat_x = -1 (boundary id 1) is
        always present -- the prescribed inner flux in the "inner"
        pathway, the free (secant-driven) inner flux in the "outer"
        pathway -- evaluated using the Dirichlet-type expansion of
        hat_D consistent with the parent solver.

        ``hat_A_KBM_term`` and ``hat_B_KBM_term`` are UFL expressions
        (or Functions) carrying the KBM coefficients.  In
        the ``"inline"`` path they are UFL expressions that depend on
        the trial ``hat_n_e``.

        ``hat_A_KBM_bc_term`` / ``hat_B_KBM_bc_term`` are the analogous
        expressions evaluated with the prescribed Neumann slope (used
        only inside the ds(1) boundary integrand).  If left None the
        volume expressions are reused.
        """
        if hat_A_KBM_bc_term is None:
            hat_A_KBM_bc_term = hat_A_KBM_term
        if hat_B_KBM_bc_term is None:
            hat_B_KBM_bc_term = hat_B_KBM_term

        ne_dx = hat_ne.dx(0)

        flux_a33 = (
            (hat_C_ETG_fd / hat_ne) * ne_dx
            + hat_A_KBM_term * ne_dx
            + hat_B_KBM_term * hat_T_fd * ne_dx * ne_dx
            + hat_B_KBM_term * hat_ne * hat_dT_dx_fd * ne_dx
            + hat_D_NEO_fd * ne_dx
        )

        F1 = (
            hat_g_fd * flux_a33 * v_e.dx(0)
            - hat_ne * hat_Si_fd * (hat_nFC + hat_nCX) * v_e
        ) * dx

        # Boundary id 1 = left endpoint = inner boundary = hat_x = -1.
        if True:
            # The flux at hat_x = -1 reuses the same expansion of hat_D
            # but with the prescribed slope substituted for hat_n_e'.
            hat_D_bc = (
                hat_C_ETG_fd / hat_ne
                + hat_A_KBM_bc_term
                + hat_B_KBM_bc_term * hat_T_fd * hat_dne_dx_inner_c
                + hat_B_KBM_bc_term * hat_ne * hat_dT_dx_fd
                + hat_D_NEO_fd
            )
            F1 = F1 + hat_g_fd * hat_D_bc * hat_dne_dx_inner_c * v_e * ds(1)

        return F1

    def _build_f2_weak_form_nondim(
            self, hat_ne, hat_nFC, hat_g_fd, hat_Si_fd, hat_Scx_fd,
            hat_VFC_const, fFC_fd, v_F):
        """Dimensionless FC-neutral weak form -- Eq. (eq:weak-hat-nFC-A8)."""
        flux_FC = hat_VFC_const * (fFC_fd * hat_g_fd * hat_nFC).dx(0)
        return (flux_FC - hat_ne * (hat_Si_fd + hat_Scx_fd) * hat_nFC) * v_F * dx

    def _build_f3_weak_form_nondim(
            self, hat_ne, hat_nFC, hat_nCX, hat_g_fd,
            hat_Si_fd, hat_Scx_fd, hat_Vcx_fd, fCX_fd, half, v_C):
        """Dimensionless CX-neutral weak form -- Eq. (eq:weak-hat-nCX-A8)."""
        flux_CX = (hat_Vcx_fd * fCX_fd * hat_g_fd * hat_nCX).dx(0)
        rhs = hat_ne * (hat_Si_fd * hat_nCX - half * hat_Scx_fd * hat_nFC)
        return (flux_CX - rhs) * v_C * dx

    # ------------------------------------------------------------------
    # Separatrix-gradient boundary condition (ne_grad_bc_loc="outer")
    # ------------------------------------------------------------------

    # Boundary-condition resolution and initial guesses live in
    # src/bc_ig_helpers.py so the 1D/3D x firedrake/scipy solvers share
    # exactly one implementation; see that module's docstring.

    def _shoot_outer_grad_nondim(
        self, F, u, bcs, snes_params,
        hat_ne_curr, hat_slope_c, hat_target, tol, max_it, verbose,
    ):
        """Solve ``F == 0`` subject to hat_n_e'(0) == ``hat_target``.

        The inner boundary carries no condition of its own in this mode,
        so its flux slope ``hat_slope_c`` (the ds(1) Constant, which the
        inline-KBM boundary terms share) is the free unknown.  A damped
        secant iteration drives the residual

            r(s) = hat_n_e'(0; s) - hat_target

        to zero.  The seed is the inner slope, which can be far from the
        answer (the free flux routinely lands on the other side of zero)
        and the system need not be solvable for every slope in between,
        so each step is capped at four times the previous accepted one
        and backtracked, halving, whenever SNES fails.

        Every attempt restarts from the *same* reference state (the
        iterate this routine was handed) rather than from the previous
        attempt's answer.  That is deliberate.  Chaining warm starts
        makes r(s) path dependent, and once two slopes are close the warm
        start already passes SNES's convergence test, so it returns at
        iteration zero with the density untouched and the secant sees a
        bit-identical residual for two different slopes and concludes the
        gradient does not respond at all.  Restarting from a fixed
        reference makes r(s) a genuine function of s, which is what the
        secant needs; the cost is a few more Newton steps per attempt.

        Returns a diagnostics dict (also stored by the caller in
        ``self.grad_bc_info``).
        """
        # ds(2) has unit measure in 1D, so this assembles to hat_n_e'(0).
        grad_sep_form = hat_ne_curr.dx(0) * ds(2)
        scale = max(abs(hat_target), 1e-30)   # tol is relative to target
        history = []
        n_solves = 0
        n_backtracks = 0

        # Fixed restart state, and SNES settings that stop it exiting on
        # the step-size test before it has done any work.
        ref = u.copy(deepcopy=True)
        shoot_params = dict(snes_params)
        shoot_params["snes_stol"] = 0.0

        def _attempt(s):
            """Solve at slope ``s`` from the reference; return (r, ok)."""
            nonlocal n_solves
            u.assign(ref)
            hat_slope_c.assign(s)
            n_solves += 1
            try:
                solve(F == 0, u, bcs=bcs, solver_parameters=shoot_params)
            except Exception:
                u.assign(ref)         # discard the failed line search
                return None, False
            return float(assemble(grad_sep_form)) - hat_target, True

        s = float(hat_slope_c)
        r, ok = _attempt(s)
        if not ok:
            raise RuntimeError(
                f"[grad-bc] the nonlinear solve failed at the seed slope "
                f"hat_dne/dxhat(-1) = {s:.3e}; the separatrix-gradient mode "
                "has nothing to iterate from.  Try a different "
                "initial_guess or dne_dx_inner (which seeds the search)."
            )

        s_prev = r_prev = None
        step_prev = None
        converged = False
        for it in range(1, int(max_it) + 1):
            history.append({"iteration": it, "hat_slope": s, "residual": r})
            if verbose:
                print(f"[grad-bc] it {it:3d}: hat_dne/dxhat(-1) = {s:.6e}, "
                      f"hat_dne/dxhat(0) - target = {r:.3e}")
            if abs(r) <= tol * scale:
                converged = True
                break

            if s_prev is None:
                # Bootstrap the secant with a finite-difference step.
                step = 0.05 * abs(s) if s != 0.0 else 1e-3
            else:
                denom = r - r_prev
                if denom == 0.0:
                    raise RuntimeError(
                        "[grad-bc] secant iteration stalled: the separatrix "
                        "gradient did not respond to a change in the inner "
                        "flux.  In ne_grad_bc_loc='outer' mode nothing "
                        "may be imposed at the inner boundary."
                    )
                step = -r * (s - s_prev) / denom
                if step_prev is not None and abs(step) > 4.0 * abs(step_prev):
                    step = np.sign(step) * 4.0 * abs(step_prev)

            # Backtrack while SNES fails.
            ok = False
            for _ in range(12):
                r_new, ok = _attempt(s + step)
                if ok:
                    break
                n_backtracks += 1
                step *= 0.5
            if not ok:
                raise RuntimeError(
                    "[grad-bc] the nonlinear solve kept failing while "
                    f"searching for the separatrix gradient (last good slope "
                    f"hat_dne/dxhat(-1) = {s:.3e}, residual {r:.3e}).  The "
                    "system may have no solution near the slope this "
                    "condition demands; try kbm_treatment='picard' with "
                    "picard_relax < 1 or a coarser grad_bc_tol."
                )

            s_prev, r_prev = s, r
            s, r = s + step, r_new
            step_prev = step

        if not converged:
            raise RuntimeError(
                f"[grad-bc] the separatrix-gradient condition did not "
                f"converge in {max_it} iterations (last relative residual "
                f"{abs(history[-1]['residual']) / scale:.3e}, tolerance "
                f"{tol:.3e}); consider raising grad_bc_max_it or relaxing "
                f"grad_bc_tol."
            )
        hat_slope_c.assign(s)
        return {
            "converged":    converged,
            "iterations":   len(history),
            "solves":       n_solves,
            "backtracks":   n_backtracks,
            "hat_slope":    s,
            "rel_residual": abs(r) / scale,
            "history":      history,
        }

    # ------------------------------------------------------------------
    # Shared boundary-condition / initial-guess resolution
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # scipy (solve_bvp) implementation of the coupled three-equation model
    # ------------------------------------------------------------------

    @staticmethod
    def _check_solver_structure(solver_structure):
        """Validate and normalise the discretisation-choice flag."""
        s = str(solver_structure).lower()
        if s not in ("firedrake", "scipy"):
            raise ValueError(
                f"solver_structure must be 'firedrake' or 'scipy', got "
                f"{solver_structure!r}."
            )
        return s

    def _scipy_coefficients_nondim(self):
        """Linear interpolants of the frozen non-dim coefficients in hat_x.

        Every coefficient lives on ``self.x_init`` in SI; each returned
        callable takes hat_x and applies ``x = L * hat_x`` internally, so
        the ODE right-hand side never has to think about units.  Linear
        interpolation matches the ``np.interp`` sampling the Firedrake
        path uses to fill its coefficient Functions, so the two
        discretisations see the same coefficient reconstruction.
        """
        def _mk(arr, scale=1.0):
            f = interp1d(self.x_init, np.asarray(arr, dtype=float) * scale,
                         kind='linear', bounds_error=False,
                         fill_value='extrapolate')
            return lambda hat_x: f(self._from_hat_x(hat_x))

        return {
            'g':      _mk(self.gradr2_fsa),
            'Si':     _mk(self.S_i_pres,  1.0 / self._S0_nd),
            'Scx':    _mk(self.S_cx_pres, 1.0 / self._S0_nd),
            'Vcx':    _mk(np.abs(self.V_cx_pres), 1.0 / self._V0_nd),
            'fFC':    _mk(self.fFC),
            'fCX':    _mk(self.fCX),
            'D_NEO':  _mk(self.D_NEO,  1.0 / self._D0_nd),
            'C_ETG':  _mk(self.C_ETG,  1.0 / self._ETG_scale_nd),
            'VFC':    abs(self.V_FC) / self._V0_nd,
        }

    def solve_coupled_nondim_scipy(self,
                                   x_res=200,
                                   ne_inner_bc="neumann",
                                   ne_grad_bc_loc="inner",
                                   bc_origin=None,
                                   ne_inner=None,
                                   dne_dx_inner=None,
                                   dne_dx_outer=None,
                                   initial_guess="tanh",
                                   tanh_width=None,
                                   tanh_center=None,
                                   nFC_ic="solve",
                                   nCX_ic="solve",
                                   picard_gate_mode="average",
                                   picard_max_it=50,
                                   picard_rtol=1e-8,
                                   picard_relax=1.0,
                                   bvp_tol=1e-6,
                                   bvp_max_nodes=50000,
                                   reuse_setup=True,
                                   scale_ne_inner=None,
                                   ne_floor=1e-8,
                                   verbose=None):
        """scipy (``solve_bvp``) solver for the coupled three-equation model.

        Same physics, non-dimensionalisation and boundary conditions as
        :meth:`solve_coupled_nondim`, discretised by collocation instead
        of finite elements.  Reached through
        ``solve_coupled_nondim(solver_structure="scipy", ...)``.

        Formulation
        ===========
        Equations (8)-(10) of the writeup form a *fourth*-order system
        (second order in hat_n_e, first order in each neutral), which is
        exactly Saarelma et al. 2023 Sec. 2.3's "fourth order system
        requiring four boundary-conditions".  ``solve_bvp`` wants an
        explicit first-order system, so the state carries the three
        **fluxes** rather than the derivatives:

            Y = [ hat_n_e,  Phi,  U,  W ]

            Phi = hat_f * hat_n_e'          electron flux
            U   = f_FC * hat_g * hat_n_FC   FC neutral flux / hat_V_FC
            W   = hat_V_CX * f_CX * hat_g * hat_n_CX   CX neutral flux

        with the conductance ``hat_f(hat_x, hat_n_e) = hat_g (hat_D_NEO
        + hat_D_KBM) + hat_g hat_C_ETG / hat_n_e``.  The system is then

            hat_n_e' = Phi / hat_f
            Phi'     = -hat_n_e hat_S_i (hat_n_FC + hat_n_CX)
            U'       =  hat_n_e (hat_S_i + hat_S_CX) hat_n_FC / hat_V_FC
            W'       =  hat_n_e (hat_S_i hat_n_CX - hat_S_CX hat_n_FC / 2)

        recovering hat_n_FC = U / (f_FC hat_g) and hat_n_CX = W /
        (hat_V_CX f_CX hat_g).  Writing it this way means **no
        coefficient derivatives appear anywhere** -- unlike
        ``solve_sc_scipy``, which expands d/dx[f n_e'] and therefore
        needs df/dx off the coarse p-file grid (the accuracy-limiting
        step noted in ``solver_sc``'s module docstring).  Here the
        divergence form is kept intact, as in the Firedrake weak form.

        Boundary conditions
        ===================
        Four residuals, placed exactly as in the Firedrake path (see
        ``ne_grad_bc_loc`` in the module docstring).  ``solve_bvp`` only
        needs the right *number* of residuals and does not care which end
        they are evaluated at, so ``ne_grad_bc_loc="outer"`` -- both
        hat_n_e conditions at the separatrix -- needs no shooting here.

        KBM treatment
        =============
        Only the Picard-frozen gate is supported: hat_D_KBM is refrozen
        from the latest hat_n_e between outer iterations, which is the
        Saarelma Eqs. 24-25 treatment and matches
        ``solve_sc_scipy``.  ``picard_gate_mode="average"`` puts the
        whole frozen diffusivity in the D-slot with hat_B_KBM = 0, so
        hat_f depends on hat_x and hat_n_e only.  The Firedrake path's
        ``kbm_treatment="inline"`` (smoothed local gate) has hat_f
        depending on hat_n_e' as well, which would make Phi -> hat_n_e'
        a per-point root-find; that is not implemented here.

        Initial guess
        =============
        n_e comes from ``bc_ig_helpers.build_ne_initial_guess``
        (``initial_guess``, ``tanh_width``, ``tanh_center``) and the two
        neutral profiles from
        ``bc_ig_helpers.build_neutral_initial_guess`` (``nFC_ic``,
        ``nCX_ic``) -- the same shared helpers, with the same options,
        that the Firedrake path uses, so the two discretisations start
        from identical SI profiles.  ``nFC_ic="solve"`` /
        ``nCX_ic="solve"`` integrate the neutral equations analytically
        at the frozen n_e guess; ``nFC_ic``/``nCX_ic``
        ``="manual EPEDNN loop"`` read the tabulated EPED-NN profiles,
        and ``nCX_ic="scale nFC"`` scales the FC guess by
        nCX_x0 / nFC_x0.  The SI neutral densities are converted to the
        flux variables U and W before being handed to ``solve_bvp``.

        Returns
        -------
        dict
            The same result schema as :meth:`solve_coupled_nondim`, in SI
            units -- see
            :meth:`src.solver.SaarelmaConnorBase._build_result_dict`.
        """
        v = self.verbose if verbose is None else bool(verbose)
        force_setup = not reuse_setup

        bcig.reject_legacy_ne_inner_bc(ne_inner_bc)
        ne_grad_bc_loc = bcig.check_ne_bc_loc(ne_grad_bc_loc)
        self.ne_grad_bc_loc = ne_grad_bc_loc
        picard_gate_mode = str(picard_gate_mode).lower()
        if picard_gate_mode != "average":
            raise NotImplementedError(
                "solver_structure='scipy' supports picard_gate_mode='average' "
                f"only (got {picard_gate_mode!r}).  'majority' freezes the "
                "local A/B KBM structure, which makes the conductance depend "
                "on hat_n_e' and turns Phi -> hat_n_e' into a per-point "
                "root-find; use solver_structure='firedrake' for it."
            )
        picard_relax = float(picard_relax)
        if not (0.0 < picard_relax <= 1.0):
            raise ValueError(
                f"picard_relax must be in (0, 1], got {picard_relax}."
            )

        # Equilibrium / kinetic / ETG setup -- pure numpy, no Firedrake.
        self._ensure_firedrake_coefficient_grids(x_res, force=force_setup)
        self.construct_C_ETG()

        self.x_inner = np.interp(self.psi_N_inner_boundary, self.psi_N_pres, self.x_init)

        # Only the two conditions belonging to ne_grad_bc_loc are looked up.
        nebcs = bcig.resolve_ne_bcs(
            self, ne_grad_bc_loc, bc_origin=bc_origin,
            dne_dx=(dne_dx_outer if ne_grad_bc_loc == "outer" else dne_dx_inner),
            ne_inner=ne_inner, scale_ne_inner=scale_ne_inner,
        )
        self.ne_bcs = nebcs

        self._set_nondim_scales(verbose=v)
        L, n0 = self._L_nd, self._n0_nd

        # Non-dim BC values
        # hat_dne_dx_bc is the single Neumann value; ne_grad_bc_loc says
        # which end of the domain it is imposed at.
        hat_ne_x0     = nebcs.ne_outer / n0        # = 1 by construction
        hat_nFC_x0    = self.nFC_x0 / n0
        hat_nCX_x0    = self.nCX_x0 / n0
        hat_dne_dx_bc = L * nebcs.dne_dx / n0

        if v:
            print(nebcs.describe(prefix="[nondim scipy] "))
            print(f"[nondim scipy] hat_nFC(0)       = {hat_nFC_x0:.3e}")
            print(f"[nondim scipy] hat_nCX(0)       = {hat_nCX_x0:.3e}")

        # Collocation grid (ascending, hat_x in [-1, 0]) and coefficients.
        hat_x = np.linspace(-1.0, 0.0, int(x_res))
        x_si = self._from_hat_x(hat_x)
        c = self._scipy_coefficients_nondim()
        VFC = c['VFC']
        ne_floor = float(ne_floor)

        # ---- Initial guess ------------------------------------------------
        ne_init = bcig.build_ne_initial_guess(
            self, x_si, initial_guess, nebcs,
            tanh_width=tanh_width, tanh_center=tanh_center,
        )
        self.ne_init = ne_init
        N_g = np.clip(ne_init / n0, ne_floor, None)

        # KBM diffusivity frozen from the initial guess (Saarelma Eq. 25).
        self.calc_pressure_quantities_nondim(N_g, gate_mode="average",
                                             hat_x_dofs=hat_x)
        hat_D_KBM = self._hat_D_KBM.copy()

        def conductance(hat_x_q, N_q, D_KBM_q):
            """hat_f = hat_g (hat_D_NEO + hat_D_KBM) + hat_g hat_C_ETG / N."""
            g_q = c['g'](hat_x_q)
            return g_q * (c['D_NEO'](hat_x_q) + D_KBM_q) + g_q * c['C_ETG'](hat_x_q) / N_q

        # Neutral seeds from the shared helper in src/bc_ig_helpers.py --
        # the *same* routine, on the same SI grid, that the firedrake path
        # uses, so the two discretisations start from identical profiles
        # and support the identical nFC_ic / nCX_ic options.  The SI
        # densities are then converted into the flux variables this
        # collocation system actually carries,
        #     U = f_FC hat_g hat_n_FC,   W = hat_V_CX f_CX hat_g hat_n_CX.
        nFC_init, nCX_init = bcig.build_neutral_initial_guess(
            self, x_si, ne_init, nFC_ic=nFC_ic, nCX_ic=nCX_ic,
        )
        self.nFC_init = nFC_init
        self.nCX_init = nCX_init

        g_q, fFC_q, fCX_q = c['g'](hat_x), c['fFC'](hat_x), c['fCX'](hat_x)
        Vcx_q = c['Vcx'](hat_x)
        U_g = (nFC_init / n0) * fFC_q * g_q
        W_g = (nCX_init / n0) * Vcx_q * fCX_q * g_q
        Phi_g = conductance(hat_x, N_g, hat_D_KBM) * np.gradient(N_g, hat_x)
        Y_guess = np.vstack([N_g, Phi_g, U_g, W_g])

        # ---- Residuals ----------------------------------------------------
        # D_KBM is frozen per Picard iteration, so it enters the ODE through
        # an interpolant rebuilt each time round the loop.
        D_KBM_x = interp1d(hat_x, hat_D_KBM, kind='linear',
                           bounds_error=False, fill_value='extrapolate')

        def ode(xq, Y):
            N, Phi, U, W = Y
            N_s = np.clip(N, ne_floor, None)
            g_q = c['g'](xq)
            fFC_q, fCX_q, Vcx_q = c['fFC'](xq), c['fCX'](xq), c['Vcx'](xq)
            Si_q, Scx_q = c['Si'](xq), c['Scx'](xq)
            f_q = g_q * (c['D_NEO'](xq) + D_KBM_x(xq)) + g_q * c['C_ETG'](xq) / N_s
            nFC = U / (fFC_q * g_q)
            nCX = W / (Vcx_q * fCX_q * g_q)
            return np.vstack([
                Phi / f_q,
                -N * Si_q * (nFC + nCX),
                N * (Si_q + Scx_q) * nFC / VFC,
                N * (Si_q * nCX - 0.5 * Scx_q * nFC),
            ])

        # Separatrix (hat_x = 0) conversions between density and flux.
        g0, fFC0, fCX0, Vcx0 = (float(c['g'](0.0)), float(c['fFC'](0.0)),
                                float(c['fCX'](0.0)), float(c['Vcx'](0.0)))
        U_x0 = hat_nFC_x0 * fFC0 * g0
        W_x0 = hat_nCX_x0 * Vcx0 * fCX0 * g0

        def bc(Ya, Yb):
            # Yb is the separatrix (hat_x = 0), Ya the inner boundary (-1).
            res = [Yb[0] - hat_ne_x0,      # n_e(0)   = ne_x0
                   Yb[2] - U_x0,           # n_FC(0)  = nFC_x0
                   Yb[3] - W_x0]           # n_CX(0)  = nCX_x0
            if ne_grad_bc_loc == "outer":
                # Fourth condition also at the separatrix: n_e'(0) prescribed,
                # i.e. Phi(0) = hat_f(0, n_e(0)) * hat_dne_dx_bc.
                f0 = conductance(0.0, max(Yb[0], ne_floor), float(D_KBM_x(0.0)))
                res.append(Yb[1] - f0 * hat_dne_dx_bc)
            else:                                   # Neumann at the inner end
                fin = conductance(-1.0, max(Ya[0], ne_floor), float(D_KBM_x(-1.0)))
                res.append(Ya[1] - fin * hat_dne_dx_bc)
            return np.array(res)

        # ---- Picard loop on the frozen KBM diffusivity --------------------
        picard_history = []
        picard_converged = False
        n_picard = 0
        prev_gate = bool(self.kbm_gate_on)
        x_prev, Y_prev = hat_x, Y_guess
        sol = None

        for it in range(1, int(picard_max_it) + 1):
            n_picard = it
            D_KBM_x = interp1d(hat_x, hat_D_KBM, kind='linear',
                               bounds_error=False, fill_value='extrapolate')

            Y_start = np.vstack([np.interp(hat_x, x_prev, Y_prev[k])
                                 for k in range(4)])
            sol = solve_bvp(ode, bc, hat_x, Y_start,
                            tol=bvp_tol, max_nodes=int(bvp_max_nodes),
                            verbose=0)
            if not sol.success:
                raise RuntimeError(
                    f"[nondim scipy] Picard iteration {it} BVP failed: "
                    f"{sol.message}"
                )

            N_new = np.interp(hat_x, sol.x, sol.y[0])
            dn_rel = (np.max(np.abs(N_new - np.interp(hat_x, x_prev, Y_prev[0])))
                      / max(np.max(np.abs(N_new)), 1e-300))
            x_prev, Y_prev = sol.x, sol.y

            # Refreeze the gate / diffusivity from the new profile.
            self.calc_pressure_quantities_nondim(
                np.clip(N_new, ne_floor, None), gate_mode="average",
                hat_x_dofs=hat_x,
            )
            gate_now = bool(self.kbm_gate_on)
            hat_D_KBM = (picard_relax * self._hat_D_KBM
                         + (1.0 - picard_relax) * hat_D_KBM)

            picard_history.append({
                "iteration":        it,
                "alpha_bar":        float(self.alpha_bar_ped),
                "alpha_frac_above": float(self.alpha_frac_above),
                "kbm_gate_on":      gate_now,
                "dne_rel":          float(dn_rel),
            })
            if v:
                print(f"[nondim scipy] picard it {it:3d}: "
                      f"|dne|_rel = {dn_rel:.3e}, "
                      f"alpha_bar = {self.alpha_bar_ped:.4f}, "
                      f"KBM {'ON' if gate_now else 'OFF'}")
            if dn_rel < float(picard_rtol) and gate_now == prev_gate:
                picard_converged = True
                break
            prev_gate = gate_now

        self.picard_info = {
            "converged":  picard_converged,
            "iterations": n_picard,
            "gate_mode":  picard_gate_mode,
            "history":    picard_history,
        }
        self.kbm_info = {
            "treatment":  "picard",
            "alpha_crit": float(self.alpha_crit),
            "picard_gate_mode": picard_gate_mode,
            "picard_converged": picard_converged,
            "picard_iterations": n_picard,
            "solver_structure": "scipy",
        }
        if not picard_converged:
            raise RuntimeError(
                f"[nondim scipy] Picard loop did not converge in "
                f"{picard_max_it} iterations "
                f"(last |dne|_rel = {picard_history[-1]['dne_rel']:.3e}); "
                "consider increasing picard_max_it or setting "
                "picard_relax < 1."
            )

        # ---- Recover the densities and store, SI + hat --------------------
        hat_x_sol = sol.x
        N_sol, Phi_sol, U_sol, W_sol = sol.y
        g_s, fFC_s, fCX_s = (c['g'](hat_x_sol), c['fFC'](hat_x_sol),
                             c['fCX'](hat_x_sol))
        Vcx_s = c['Vcx'](hat_x_sol)
        nFC_sol = U_sol / (fFC_s * g_s)
        nCX_sol = W_sol / (Vcx_s * fCX_s * g_s)
        f_sol = conductance(hat_x_sol, np.clip(N_sol, ne_floor, None),
                            D_KBM_x(hat_x_sol))

        self.hat_x_sol   = hat_x_sol
        self.hat_ne_sol  = N_sol
        self.hat_nFC_sol = nFC_sol
        self.hat_nCX_sol = nCX_sol

        self.x_sol   = hat_x_sol * L
        self.ne_sol  = N_sol   * n0
        self.nFC_sol = nFC_sol * n0
        self.nCX_sol = nCX_sol * n0
        self.dne_dx_sol = (Phi_sol / f_sol) * (n0 / L)     # SI m^-4
        self.sol = sol

        if v:
            print(
                f"[nondim scipy] solved on {hat_x_sol.size} collocation nodes.\n"
                f"  n_e  in [{self.ne_sol.min():.3e}, {self.ne_sol.max():.3e}] m^-3\n"
                f"  n_FC in [{self.nFC_sol.min():.3e}, {self.nFC_sol.max():.3e}] m^-3\n"
                f"  n_CX in [{self.nCX_sol.min():.3e}, {self.nCX_sol.max():.3e}] m^-3"
            )

        return self._build_result_dict('3D', 'scipy')

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def solve_coupled_nondim(self,
                      x_res=20,
                      fe_degree=2,
                      solver_structure="firedrake",
                      ne_inner_bc="neumann",
                      ne_grad_bc_loc="inner",
                      bc_origin=None,
                      ne_inner=None,
                      dne_dx_inner=None,
                      dne_dx_outer=None,
                      grad_bc_tol=1e-8,
                      grad_bc_max_it=25,
                      grad_bc_seed=None,
                      initial_guess="tanh",
                      tanh_width=None,
                      tanh_center=None,
                      linear_solver="lu",
                      ksp_rtol=1e-8,
                      ksp_max_it=200,
                      reuse_setup=True,
                      scale_ne_inner=None,
                      nCX_ic="solve",
                      nFC_ic="solve",
                      kbm_treatment="inline",
                      kbm_gate_eps=None,
                      picard_gate_mode="average",
                      picard_max_it=50,
                      picard_rtol=1e-8,
                      picard_relax=1.0,
                      neutrals_treatment="fem",
                      n_neutral_sub=4001,
                      bvp_tol=1e-6,
                      bvp_max_nodes=50000,
                      ne_floor=1e-8,
                      verbose=None):
        """Non-dimensional Firedrake solver for the coupled three-equation
        Saarelma--Connor neutral-transport pedestal model.

        Equivalent to :meth:`saarelma_connor.solve_coupled` but assembles
        and solves the dimensionless residuals derived in App.~A.8
        (foundational scaling, no free $\\Lambda$ prefactor).  Inputs and
        outputs are in SI units; the rescaling is invisible to the
        caller.

        Boundary conditions
        ===================
        ``ne_grad_bc_loc="inner"`` (default) -- one condition per end:

            hat_n_e (hat_x = 0)        = 1
            hat_n_FC(hat_x = 0)        = nFC_x0 / n0
            hat_n_CX(hat_x = 0)        = nCX_x0 / n0
            hat_n_e (hat_x = -1)       = ne_inner / n0   (Dirichlet option)
         OR hat_n_e'(hat_x = -1)       = L * dne_dx_inner / n0   (Neumann option)

        ``ne_grad_bc_loc="outer"`` -- both n_e conditions at the
        separatrix (Saarelma et al. 2023 Sec. 2.3), inner boundary free:

            hat_n_e (hat_x = 0)        = 1
            hat_n_e'(hat_x = 0)        = L * dne_dx_outer / n0
            hat_n_FC(hat_x = 0)        = nFC_x0 / n0
            hat_n_CX(hat_x = 0)        = nCX_x0 / n0
            (nothing imposed at hat_x = -1)

        Parameters
        ----------
        See :meth:`saarelma_connor.solve_coupled`.  All physical inputs
        are in SI units.

        solver_structure : {"firedrake", "scipy"}, default "firedrake"
            Which discretisation solves the coupled system.  ``"scipy"``
            forwards to :meth:`solve_coupled_nondim_scipy` (collocation
            via ``solve_bvp``); it supports ``picard_gate_mode="average"``
            only.  Firedrake-only arguments (``fe_degree``,
            ``linear_solver``, ``ksp_*``, ``kbm_treatment="inline"``,
            ``neutrals_treatment``, ``grad_bc_*``) do not apply there.
        bvp_tol, bvp_max_nodes : float, int
            ``solve_bvp`` controls, used only by ``solver_structure="scipy"``.
        ne_floor : float
            Lower clip on hat_n_e keeping hat_C_ETG / hat_n_e finite in
            the scipy driver.
        ne_grad_bc_loc : {"inner", "outer"}, default "inner"
            Which end of the domain carries the prescribed dn_e/dx; see
            the module docstring.  ``"outer"`` frees the inner boundary
            entirely and forces ``ne_inner_bc="neumann"`` internally
            (the ds(1) flux becomes the unknown that the separatrix
            gradient determines), so a requested inner Dirichlet
            condition is ignored with a ``RuntimeWarning``.
        dne_dx_outer : float or None
            SI separatrix slope dn_e/dx|_{x=0} (m^-4) used when
            ``ne_grad_bc_loc="outer"``.  Taken from the same origins as
            the inner slope: given explicitly it wins for any
            ``bc_origin`` (this is how Saarelma Eq. (20) is fed in);
            left None it is read off the p-file gradient at x = 0, which
            requires a p-file-backed ``bc_origin``.
        grad_bc_tol : float, default 1e-8
            Relative tolerance on hat_n_e'(0) - target for the
            ``"outer"`` secant iteration.
        grad_bc_max_it : int, default 25
            Maximum secant iterations for the ``"outer"`` mode.

        tanh_width is in x units

        scale_ne_inner is a scaling factor for the inner boundary density for testing purposes

        Sets
        ----
        SI-unit attributes (parent-class compatibility):
            self.x_sol, self.ne_sol, self.nFC_sol, self.nCX_sol
            self.u_fd  (mixed Firedrake Function -- now stores hat values)
            self.W_fd, self.V_fd

        Non-dim attributes (diagnostics):
            self.hat_x_sol, self.hat_ne_sol, self.hat_nFC_sol, self.hat_nCX_sol
            self._L_nd, self._n0_nd, self._T0_nd, self._S0_nd
            self._tau_nd, self._V0_nd, self._D0_nd

        KBM treatment
        =============
        ``kbm_treatment`` controls how the piecewise KBM gate
        (Heaviside on alpha - alpha_crit) is handled.

        ``"inline"`` (default, new)
            ``hat_A_KBM`` and ``hat_B_KBM`` are written directly as UFL
            expressions of the trial hat_n_e (App. A.7--A.8 of the
            writeup), with the discontinuous indicator replaced by the
            smoothed Heaviside

                gate(z) = 0.5 * (1 + tanh(z / kbm_gate_eps))

            so the residual is C^infty in hat_n_e and SNES Newton can
            differentiate it analytically.  The Connor-Hastie alpha is
            evaluated locally as

                alpha(hat_x, hat_n_e) = hat_alpha_nodp(hat_x)
                                        * d/dhat_x [ hat_n_e * hat_T ].

            The gate therefore updates self-consistently within every
            Newton step; on convergence the model is on its own KBM
            branch by construction.

        ``"picard"``
            Outer Picard (fixed-point) loop following Saarelma et al.
            (2023): each Picard iteration the KBM gate and coefficients
            are evaluated from the *current* hat_n_e (the initial guess
            on the first pass, thereafter the previously solved
            profile), then frozen while SNES solves the three-field
            system.  The loop repeats until the density profile is
            unchanged to ``picard_rtol`` and the gate state is stable
            (or ``picard_max_it`` is hit).  ``picard_gate_mode`` selects
            both the whole-grid gate criterion and the frozen flux
            structure:

            ``picard_gate_mode="average"`` (default)
                KBM on iff pedestal-averaged alpha exceeds alpha_crit
                (Eq. 24).  When on, freeze
                ``hat_D_KBM = (alpha_bar - alpha_crit) * hat_G`` into
                the A-slot with B = 0 (Eq. 25 diffusivity).
            ``picard_gate_mode="majority"``
                KBM on iff more than half of the pedestal grid points
                have local alpha > alpha_crit.  When on, freeze the
                local A/B KBM structure everywhere (no pointwise gate).

        ``kbm_gate_eps`` : float or None, default None
            Smoothing width of the Heaviside (inline treatment only).
            ``None`` -> 5% of ``alpha_crit`` (with a 1e-3 floor).
            Smaller eps makes the gate sharper but harder for Newton
            to converge.

        ``picard_max_it`` : int, default 50
            Maximum number of Picard iterations ("picard" only).

        ``picard_rtol`` : float, default 1e-8
            Relative L2 tolerance on the change in hat_n_e between
            Picard iterations ("picard" only).

        ``picard_relax`` : float in (0, 1], default 1.0
            Under-relaxation factor for the frozen KBM coefficient
            update: coeff_new = relax * coeff_computed + (1 - relax) *
            coeff_old.  Use < 1 if the gate flip-flops between iterations.
            1.0 means this effect is disabled.

        After a "picard" solve, ``self.picard_info`` records the
        iteration history (alpha_bar, gate state, profile change) and
        whether the loop converged.

        Neutrals treatment
        ==================
        ``neutrals_treatment`` controls how the FC / CX neutral
        equations are solved.

        ``"fem"`` (default)
            n_FC and n_CX are solved as part of the mixed three-field
            Firedrake system (legacy behaviour).  Requires the FE mesh
            to resolve the neutral penetration depth
            lambda = |V_FC| / (n_e (S_i + S_CX)); on neutral-opaque
            devices (e.g. SPARC, lambda ~ 0.2 mm) an unresolved layer
            produces oscillatory / negative neutral densities and
            Newton line-search failures.

        ``"analytic"`` (requires ``kbm_treatment="picard"``)
            Each Picard iteration, n_FC and n_CX are solved *exactly*
            (integrating factor / per-step exponential integrator) on a
            dense ``n_neutral_sub``-point sub-grid for the current n_e
            (see :meth:`_solve_neutrals_analytic_nondim`), then frozen
            in the n_e equation.  The mixed-system residuals for n_FC /
            n_CX are replaced by trivial pinning forms so the rest of
            the machinery (mixed space, BCs, output extraction) is
            unchanged.  Positivity of the neutrals is guaranteed and
            ``x_res`` only needs to resolve the *electron* profile.

        ``n_neutral_sub`` : int, default 4001
            Number of sub-grid points for the analytic neutral solve.
            Choose so that L / n_neutral_sub << min(lambda).

        Returns
        -------
        dict
            The common solver result schema, in SI units -- see
            :meth:`src.solver.SaarelmaConnorBase._build_result_dict`.
        """
        if not _FIREDRAKE_AVAILABLE:
            raise ImportError(
                "Firedrake is not available in this environment.  "
                "Install Firedrake to use this solver.  Original import "
                f"error:\n  {_FIREDRAKE_IMPORT_ERR}"
            )

        solver_structure = self._check_solver_structure(solver_structure)
        if solver_structure == "scipy":
            # Collocation instead of finite elements; see
            # solve_coupled_nondim_scipy for the flux-variable formulation.
            return self.solve_coupled_nondim_scipy(
                x_res=x_res,
                ne_inner_bc=ne_inner_bc,
                ne_grad_bc_loc=ne_grad_bc_loc,
                bc_origin=bc_origin,
                ne_inner=ne_inner,
                dne_dx_inner=dne_dx_inner,
                dne_dx_outer=dne_dx_outer,
                initial_guess=initial_guess,
                tanh_width=tanh_width,
                tanh_center=tanh_center,
                nFC_ic=nFC_ic,
                nCX_ic=nCX_ic,
                picard_gate_mode=picard_gate_mode,
                picard_max_it=picard_max_it,
                picard_rtol=picard_rtol,
                picard_relax=picard_relax,
                bvp_tol=bvp_tol,
                bvp_max_nodes=bvp_max_nodes,
                reuse_setup=reuse_setup,
                scale_ne_inner=scale_ne_inner,
                ne_floor=ne_floor,
                verbose=verbose,
            )

        v = self.verbose if verbose is None else bool(verbose)
        force_setup = not reuse_setup

        # Equilibrium-only quantities (FSA |grad r|^2, S_i_pres, etc.) and
        # ETG coefficient -- exactly the same setup as the parent class.
        self._ensure_firedrake_coefficient_grids(x_res, force=force_setup)
        self.construct_C_ETG()

        # Inner boundary location (in SI x); same logic as solver.py.
        if self.psi_N_inner_boundary is None:
            raise ValueError(
                "need to specify psi_N_inner_boundary. auto psi_N inner boundary method is no longer offered."
            )
        else:
            self.x_inner = np.interp(self.psi_N_inner_boundary, self.psi_N_pres, self.x_init)

        bcig.reject_legacy_ne_inner_bc(ne_inner_bc)
        ne_grad_bc_loc = bcig.check_ne_bc_loc(ne_grad_bc_loc)
        self.ne_grad_bc_loc = ne_grad_bc_loc

        # Read off ne(x_inner), dne/dx(x_inner) in SI -- same conventions
        # as solver.py.solve_coupled, shared with the scipy driver.
        # Only the two conditions belonging to ne_grad_bc_loc are looked up.
        nebcs = bcig.resolve_ne_bcs(
            self, ne_grad_bc_loc, bc_origin=bc_origin,
            dne_dx=(dne_dx_outer if ne_grad_bc_loc == "outer" else dne_dx_inner),
            ne_inner=ne_inner, scale_ne_inner=scale_ne_inner,
        )
        self.ne_bcs = nebcs

        # Compute reference scales now that x_inner / ne_x0 / etc. are known.
        self._set_nondim_scales(verbose=v)

        # Mesh + frozen non-dim coefficient Functions on hat_x in [-1, 0].
        mesh, V, W, hat_x_dofs, hat_g_fd, hat_Si_fd, hat_Scx_fd, hat_Vcx_fd = (
            self._ensure_firedrake_discretization_nondim(
                x_res, fe_degree, force=force_setup,
            )
        )
        x_dofs_si = self._from_hat_x(hat_x_dofs)  # m
        self.x_dofs_si = x_dofs_si
        x_left_si = -self._L_nd
        x_right_si = 0.0

        # Constants (App. A.8 Eq. (eq:hat-V-A8))
        hat_VFC_const = Constant(abs(self.V_FC) / self._V0_nd)
        # fFC, fCX are spatial profiles on x_init; represent them as scalar
        # Functions on V (not Constant(array), which would have wrong UFL shape).
        fFC_arr = np.interp(x_dofs_si, self.x_init, self.fFC)
        fCX_arr = np.interp(x_dofs_si, self.x_init, self.fCX)
        fFC_fd = Function(V, name="fFC")
        fFC_fd.dat.data[:] = fFC_arr
        fCX_fd = Function(V, name="fCX")
        fCX_fd.dat.data[:] = fCX_arr
        half          = Constant(0.5)

        u, u_prev = self._get_or_create_mixed_solution_nondim(W, force=force_setup)

        # ------------------------------------------------------------------
        # Initial guess (in SI, then rescaled to hat-units).
        # ------------------------------------------------------------------
        ne_init = bcig.build_ne_initial_guess(
            self, x_dofs_si, initial_guess, nebcs,
            tanh_width=tanh_width, tanh_center=tanh_center,
        )

        # ------------------------------------------------------------------
        # KBM coefficients evaluated at the initial guess.  For
        # "picard" they seed the first Picard iteration; for "inline"
        # they are only used for diagnostics (alpha_bar of the guess).
        # ------------------------------------------------------------------
        kbm_treatment = str(kbm_treatment).lower()
        if kbm_treatment not in ("inline", "picard"):
            raise ValueError(
                "kbm_treatment must be 'inline' or 'picard' "
                f"got {kbm_treatment!r}."
            )
        picard_gate_mode = str(picard_gate_mode).lower()
        if kbm_treatment == "picard" and picard_gate_mode not in ("average", "majority"):
            raise ValueError(
                "picard_gate_mode must be 'average' or 'majority', "
                f"got {picard_gate_mode!r}."
            )
        picard_relax = float(picard_relax)
        if kbm_treatment == "picard" and not (0.0 < picard_relax <= 1.0):
            raise ValueError(
                f"picard_relax must be in (0, 1], got {picard_relax}."
            )
        neutrals_treatment = str(neutrals_treatment).lower()
        if neutrals_treatment not in ("fem", "analytic"):
            raise ValueError(
                "neutrals_treatment must be 'fem' or 'analytic', "
                f"got {neutrals_treatment!r}."
            )
        if neutrals_treatment == "analytic" and kbm_treatment != "picard":
            raise ValueError(
                "neutrals_treatment='analytic' requires "
                "kbm_treatment='picard' (the neutrals are refrozen from "
                "the latest n_e inside the Picard loop)."
            )

        if kbm_treatment == "picard":
            gate_mode_input = picard_gate_mode
        elif kbm_treatment == "inline":
            gate_mode_input = "local"

        self.calc_pressure_quantities_nondim(
            ne_init / self._n0_nd,
            gate_mode=gate_mode_input,
        )

        # Neutral initial guesses -- built by the shared helper in
        # src/bc_ig_helpers.py so the firedrake and scipy paths cannot
        # drift apart (same contract as build_ne_initial_guess above).
        nFC_init, nCX_init = bcig.build_neutral_initial_guess(
            self, x_dofs_si, ne_init, nFC_ic=nFC_ic, nCX_ic=nCX_ic,
        )

        # diagnostics
        self.nCX_init = nCX_init
        self.nFC_init = nFC_init
        self.ne_init = ne_init

        u.subfunctions[0].dat.data[:] = ne_init  / self._n0_nd
        u.subfunctions[1].dat.data[:] = nFC_init / self._n0_nd
        u.subfunctions[2].dat.data[:] = nCX_init / self._n0_nd
        u_prev.assign(u)

        # Rescale BC values into hat-units (App. A.8 Eq. (eq:hat-flux-A8) etc.) using initial guesses to accommodate Dirichlet or Neumann boundary condition choice
        # BC conditions for nFC, nCX, ne will not change throughout EPEDNN loop
        hat_ne_x0        = nebcs.ne_outer / self._n0_nd   # = 1 by construction
        hat_nFC_x0       = self.nFC_x0  / self._n0_nd
        hat_nCX_x0       = self.nCX_x0  / self._n0_nd
        # The single Neumann value.  In "inner" mode it is imposed on ds(1);
        # in "outer" mode ds(1) carries the *unknown* inner flux and this is
        # the separatrix-gradient target the secant drives hat_n_e'(0) to, so
        # it also seeds that search -- the pedestal-top gradient is never
        # looked up in that pathway.
        hat_dne_dx_bc    = self._L_nd * nebcs.dne_dx / self._n0_nd

        if v:
            print(nebcs.describe(prefix="[nondim] "))
            print(f"[nondim] hat_ne(0)        = {hat_ne_x0}")
            print(f"[nondim] hat_nFC(0)       = {hat_nFC_x0:.3e}")
            print(f"[nondim] hat_nCX(0)       = {hat_nCX_x0:.3e}")
            print(f"[nondim] hat_dne/dxhat    = {hat_dne_dx_bc:.3e}"
                  f"  ({'ds(1) Neumann' if ne_grad_bc_loc == 'inner' else 'separatrix target + secant seed'})")

        # Frozen ETG and NEO contributions in hat-units.
        hat_C_ETG_arr = (
            self._interp_si_to_hat(hat_x_dofs, self.C_ETG) / self._ETG_scale_nd
        )
        hat_D_NEO_arr = (
            self._interp_si_to_hat(hat_x_dofs, self.D_NEO) / self._D0_nd
        )

        f = Function(V, name="hat_C_ETG"); f.dat.data[:] = hat_C_ETG_arr
        self._fd_cache["hat_C_ETG_fd"] = f
        f = Function(V, name="hat_D_NEO"); f.dat.data[:] = hat_D_NEO_arr
        self._fd_cache["hat_D_NEO_fd"] = f
        f = Function(V, name="hat_A_KBM"); f.dat.data[:] = self._hat_A_KBM
        self._fd_cache["hat_A_KBM_fd"] = f
        f = Function(V, name="hat_B_KBM"); f.dat.data[:] = self._hat_B_KBM
        self._fd_cache["hat_B_KBM_fd"] = f
        f = Function(V, name="hat_D_KBM"); f.dat.data[:] = self._hat_D_KBM
        self._fd_cache["hat_D_KBM_fd"] = f

        # hat_T(hat_x) and d hat_T / d hat_x at mesh DOFs.
        T_e_J = self.T_e_pres * _EV2J # J
        T_i_J = self.T_i_pres * _EV2J # J
        hat_T_arr = (
            self._interp_si_to_hat(hat_x_dofs, T_e_J + T_i_J) / self._T0_nd
        )
        # d hat_T / d hat_x = (L / T0) * d (T_e + T_i) / dx
        dT_dx_J_arr = np.gradient(T_e_J + T_i_J, self.x_init) # gradient just on pedestal grid
        hat_dT_dx_arr = (
            self._interp_si_to_hat(hat_x_dofs, dT_dx_J_arr)
            * self._L_nd / self._T0_nd
        )

        f = Function(V, name="hat_T");        f.dat.data[:] = hat_T_arr
        self._fd_cache["hat_T_fd"] = f
        f = Function(V, name="hat_dT_dxhat"); f.dat.data[:] = hat_dT_dx_arr
        self._fd_cache["hat_dT_dx_fd"] = f

        # ------------------------------------------------------------------
        # KBM equilibrium fields used by the *inline* (trial-dependent)
        # KBM treatment.  These are independent of the trial hat_n_e but
        # depend on the dimensionless scales, so they are built here
        # (after _set_nondim_scales) rather than inside
        # _ensure_firedrake_discretization_nondim.
        #
        #   hat_G_KBM(hat_x)
        #       = G_KBM_si(x) / [D]_0
        #       = ( C_KBM * c_s(x) * rho_s(x)^2 / a ) / (L^2 n_0 S_0).
        #
        #   hat_alpha_nodp(hat_x)
        #       = alpha_nodp_si(x) * n_0 * T_0 / L
        #         (dimensionless: alpha = hat_alpha_nodp * d(hat_n_e hat_T)/dhat_x).
        # ------------------------------------------------------------------
        G_KBM_grid = self.C_KBM * (self.c_s * self.rho_s ** 2) / self.a  # m^2/s
        hat_G_KBM_arr = (
            self._interp_si_to_hat(hat_x_dofs, G_KBM_grid) / self._D0_nd
        )
        hat_alpha_nodp_arr = (
            self._interp_si_to_hat(hat_x_dofs, self._alpha_nodp_xinit)
            * (self._n0_nd * self._T0_nd / self._L_nd)
        )
        f = Function(V, name="hat_G_KBM");       f.dat.data[:] = hat_G_KBM_arr
        self._fd_cache["hat_G_KBM_fd"] = f
        f = Function(V, name="hat_alpha_nodp"); f.dat.data[:] = hat_alpha_nodp_arr
        self._fd_cache["hat_alpha_nodp_fd"] = f

        if v:
            self._plot_profiles(
                x_dofs=x_dofs_si,
                ne=ne_init,
                nFC=nFC_init,
                nCX=nCX_init,
                title=f"Initial guess (SI, from non-dim solver): '{initial_guess}'",
            )

        # ------------------------------------------------------------------
        # Boundary conditions in non-dim units.
        # IntervalMesh boundary IDs: 1 = left (hat_x = -1), 2 = right (0).
        # ------------------------------------------------------------------
        hat_ne_x0_c    = Constant(hat_ne_x0)
        hat_nFC_x0_c   = Constant(hat_nFC_x0)
        hat_nCX_x0_c   = Constant(hat_nCX_x0)
        # ds(1) slope: the prescribed Neumann value in "inner" mode, the
        # free unknown the secant drives in "outer" mode.
        # Seed for the "outer"-mode secant.  Zero inner flux is the
        # natural boundary condition a Galerkin form defaults to, so it is
        # the neutral starting point and -- unlike seeding from a
        # pedestal-top gradient -- uses no information the "outer"
        # pathway forbids.  It is also markedly more robust: r(s) is
        # discontinuous where the profile switches solution branch, and
        # approaching from zero avoids straddling that jump.
        hat_slope_seed = (hat_dne_dx_bc if ne_grad_bc_loc == "inner"
                          else (0.0 if grad_bc_seed is None
                                else self._L_nd * float(grad_bc_seed)
                                     / self._n0_nd))
        hat_dne_dx_inner_c = Constant(hat_slope_seed)

        bcs = [
            DirichletBC(W.sub(0), hat_ne_x0_c,  2),
            DirichletBC(W.sub(1), hat_nFC_x0_c, 2),
            DirichletBC(W.sub(2), hat_nCX_x0_c, 2),
        ]

        # ------------------------------------------------------------------
        # KBM treatment dispatch
        # ------------------------------------------------------------------
        # Smoothing width for the Heaviside in the inline gate.  Default
        # is 5% of alpha_crit (>= 1e-3 to avoid division by zero at
        # alpha_crit == 0).
        if kbm_gate_eps is None:
            kbm_gate_eps_val = max(1e-3, 0.05 * float(self.alpha_crit))
        else:
            kbm_gate_eps_val = float(kbm_gate_eps)
            if kbm_gate_eps_val <= 0.0:
                raise ValueError(
                    f"kbm_gate_eps must be > 0, got {kbm_gate_eps_val}."
                )
        alpha_crit_c = Constant(float(self.alpha_crit))
        gate_eps_c   = Constant(kbm_gate_eps_val)

        # Diagnostic record for callers / notebooks.
        self.kbm_info = {
            "treatment":  kbm_treatment,
            "gate_eps":   kbm_gate_eps_val,
            "alpha_crit": float(self.alpha_crit),
            "alpha_bar_initial_guess": float(self.alpha_bar_ped),
            "neutrals_treatment": neutrals_treatment,
        }
        if kbm_treatment == "picard":
            self.kbm_info.update({
                "picard_gate_mode": picard_gate_mode,
                "picard_max_it":    int(picard_max_it),
                "picard_rtol":      float(picard_rtol),
                "picard_relax":     float(picard_relax),
            })

        # ------------------------------------------------------------------
        # Assemble residuals and solve via SNES.
        # ------------------------------------------------------------------
        v_e, v_F, v_C = TestFunctions(W)
        hat_ne_curr, hat_nFC_curr, hat_nCX_curr = split(u)

        if kbm_treatment == "picard":
            # Per-iteration-frozen KBM coefficients, updated between
            # Picard iterations from the latest hat_n_e.
            #
            # average: freeze hat_D_KBM = (alpha_bar - alpha_crit)*hat_G
            #          into the A-slot with B = 0 (Saarelma Eq. 25).
            # majority: freeze the local A/B structure everywhere when
            #          the majority vote says the gate is on.
            if picard_gate_mode == "average":
                hat_A_KBM_picard_fd = Function(V, name="hat_D_KBM_picard")
                hat_A_KBM_picard_fd.dat.data[:] = self._hat_D_KBM
                self._fd_cache["hat_D_KBM_picard_fd"] = hat_A_KBM_picard_fd
                hat_B_KBM_picard_fd = None
                hat_A_KBM_term = hat_A_KBM_picard_fd
                hat_B_KBM_term = Constant(0.0)
            else:  # majority
                hat_A_KBM_picard_fd = Function(V, name="hat_A_KBM_picard")
                hat_B_KBM_picard_fd = Function(V, name="hat_B_KBM_picard")
                hat_A_KBM_picard_fd.dat.data[:] = self._hat_A_KBM
                hat_B_KBM_picard_fd.dat.data[:] = self._hat_B_KBM
                self._fd_cache["hat_A_KBM_picard_fd"] = hat_A_KBM_picard_fd
                self._fd_cache["hat_B_KBM_picard_fd"] = hat_B_KBM_picard_fd
                hat_A_KBM_term = hat_A_KBM_picard_fd
                hat_B_KBM_term = hat_B_KBM_picard_fd
            # Reuse volume coefficients on ds(1) (static Functions).
            hat_A_KBM_bc_term = None
            hat_B_KBM_bc_term = None
        elif kbm_treatment == "inline":  # inline
            # Volume terms: depend on the trial via hat_ne.dx(0).
            (_alpha_ufl, _gate_ufl,
             hat_A_KBM_term, hat_B_KBM_term) = self._kbm_inline_terms(
                hat_ne_curr,
                self._fd_cache["hat_T_fd"],
                self._fd_cache["hat_dT_dx_fd"],
                self._fd_cache["hat_alpha_nodp_fd"],
                self._fd_cache["hat_G_KBM_fd"],
                alpha_crit_c, gate_eps_c,
                boundary=False,
            )
            # Boundary terms (Neumann ds(1)): the prescribed slope is
            # substituted for hat_n_e' inside alpha so the gate at the
            # inner boundary stays consistent with the imposed flux.
            (_, _,
             hat_A_KBM_bc_term, hat_B_KBM_bc_term) = self._kbm_inline_terms(
                hat_ne_curr,
                self._fd_cache["hat_T_fd"],
                self._fd_cache["hat_dT_dx_fd"],
                self._fd_cache["hat_alpha_nodp_fd"],
                self._fd_cache["hat_G_KBM_fd"],
                alpha_crit_c, gate_eps_c,
                boundary=True,
                hat_dne_dx_inner_c=hat_dne_dx_inner_c,
            )

        F1 = self._build_f1_weak_form_nondim(
            hat_ne_curr,
            self._fd_cache["hat_g_fd"],
            self._fd_cache["hat_C_ETG_fd"],
            self._fd_cache["hat_D_NEO_fd"],
            hat_A_KBM_term, hat_B_KBM_term,
            self._fd_cache["hat_T_fd"],
            self._fd_cache["hat_dT_dx_fd"],
            self._fd_cache["hat_Si_fd"],
            hat_nFC_curr, hat_nCX_curr,
            v_e, hat_dne_dx_inner_c,
            hat_A_KBM_bc_term=hat_A_KBM_bc_term,
            hat_B_KBM_bc_term=hat_B_KBM_bc_term,
        )
        if neutrals_treatment == "analytic":
            # Neutrals are solved exactly (integrating factor) for the
            # current n_e and frozen into Functions.  The mixed-system
            # residuals for n_FC / n_CX reduce to pinning forms so the
            # Newton solve only really works on n_e; the frozen
            # Functions are refreshed each Picard iteration below.
            hat_nFC_a, hat_nCX_a = self._solve_neutrals_analytic_nondim(
                u.subfunctions[0].dat.data, n_sub=n_neutral_sub,
            )
            hat_nFC_frozen_fd = Function(V, name="hat_nFC_frozen")
            hat_nCX_frozen_fd = Function(V, name="hat_nCX_frozen")
            hat_nFC_frozen_fd.dat.data[:] = hat_nFC_a
            hat_nCX_frozen_fd.dat.data[:] = hat_nCX_a
            self._fd_cache["hat_nFC_frozen_fd"] = hat_nFC_frozen_fd
            self._fd_cache["hat_nCX_frozen_fd"] = hat_nCX_frozen_fd
            # Warm-start the mixed vector with the analytic profiles.
            u.subfunctions[1].dat.data[:] = hat_nFC_a
            u.subfunctions[2].dat.data[:] = hat_nCX_a

            F2 = (hat_nFC_curr - hat_nFC_frozen_fd) * v_F * dx
            F3 = (hat_nCX_curr - hat_nCX_frozen_fd) * v_C * dx
        else:  # fem (legacy): neutrals solved in the mixed system
            F2 = self._build_f2_weak_form_nondim(
                hat_ne_curr, hat_nFC_curr,
                self._fd_cache["hat_g_fd"],
                self._fd_cache["hat_Si_fd"],
                self._fd_cache["hat_Scx_fd"],
                hat_VFC_const, fFC_fd, v_F,
            )
            F3 = self._build_f3_weak_form_nondim(
                hat_ne_curr, hat_nFC_curr, hat_nCX_curr,
                self._fd_cache["hat_g_fd"],
                self._fd_cache["hat_Si_fd"],
                self._fd_cache["hat_Scx_fd"],
                self._fd_cache["hat_Vcx_fd"],
                fCX_fd, half, v_C,
            )

        F = F1 + F2 + F3
        snes_params = self._build_petsc_solver_parameters(
            linear_solver=linear_solver,
            ksp_rtol=ksp_rtol,
            ksp_max_it=ksp_max_it,
        )

        # One nonlinear solve.  In "inner" mode that is the bare SNES
        # solve; in "outer" mode it is the secant iteration that pins
        # hat_n_e'(0) by treating the (unconstrained) inner flux as the
        # unknown.  Inside the Picard loop the slope carries over between
        # iterations, so later shootings start already close.
        self.grad_bc_info = None

        def _solve_once():
            if ne_grad_bc_loc == "inner":
                solve(F == 0, u, bcs=bcs, solver_parameters=snes_params)
            else:
                self.grad_bc_info = self._shoot_outer_grad_nondim(
                    F, u, bcs, snes_params,
                    hat_ne_curr, hat_dne_dx_inner_c, hat_dne_dx_bc,
                    float(grad_bc_tol), int(grad_bc_max_it), v,
                )

        if kbm_treatment == "picard":
            # ----------------------------------------------------------
            # Outer Picard loop (Saarelma et al. 2023): freeze the KBM
            # gate/coefficients from the current density profile, solve
            # the three-field system with SNES, recompute the gate from
            # the new profile, and repeat until the profile and gate
            # stop changing.
            # ----------------------------------------------------------
            picard_history = []
            prev_hat_ne = u.subfunctions[0].dat.data.copy()
            prev_gate = bool(self.kbm_gate_on)
            picard_converged = False
            n_picard = 0
            for it in range(1, int(picard_max_it) + 1):
                n_picard = it
                # Solve with the currently frozen KBM coefficients
                # (warm start from the previous iterate stored in u).
                _solve_once()
                hat_ne_new = u.subfunctions[0].dat.data.copy()

                dn_rel = (
                    np.linalg.norm(hat_ne_new - prev_hat_ne)
                    / max(np.linalg.norm(prev_hat_ne), 1e-300)
                )

                # Recompute alpha / gate / KBM coeffs from the newly
                # solved profile and refreeze (optional under-relaxation).
                self.calc_pressure_quantities_nondim(
                    hat_ne_new, gate_mode=picard_gate_mode,
                )
                gate_now = bool(self.kbm_gate_on)

                # Picard relax and update the frozen KBM coefficients
                # (Functions referenced by F; no form rebuild needed).
                if picard_gate_mode == "average":
                    hat_A_KBM_picard_fd.dat.data[:] = ( # = self._hat_D_KBM if picard_relax == 1.0. hat_A_KBM_picard_fd is frozen from the previous iterate.
                        picard_relax * self._hat_D_KBM
                        + (1.0 - picard_relax) * hat_A_KBM_picard_fd.dat.data
                    )
                else:  # majority: freeze A/B
                    hat_A_KBM_picard_fd.dat.data[:] = (
                        picard_relax * self._hat_A_KBM
                        + (1.0 - picard_relax) * hat_A_KBM_picard_fd.dat.data
                    )
                    hat_B_KBM_picard_fd.dat.data[:] = (
                        picard_relax * self._hat_B_KBM
                        + (1.0 - picard_relax) * hat_B_KBM_picard_fd.dat.data
                    )

                # Refreeze the analytic neutrals from the newly solved
                # n_e (same under-relaxation as the KBM coefficients).
                if neutrals_treatment == "analytic":
                    hat_nFC_a, hat_nCX_a = (
                        self._solve_neutrals_analytic_nondim(
                            hat_ne_new, n_sub=n_neutral_sub,
                        )
                    )
                    hat_nFC_frozen_fd.dat.data[:] = (
                        picard_relax * hat_nFC_a
                        + (1.0 - picard_relax) * hat_nFC_frozen_fd.dat.data
                    )
                    hat_nCX_frozen_fd.dat.data[:] = (
                        picard_relax * hat_nCX_a
                        + (1.0 - picard_relax) * hat_nCX_frozen_fd.dat.data
                    )

                picard_history.append({
                    "iteration":        it,
                    "alpha_bar":        float(self.alpha_bar_ped),
                    "alpha_frac_above": float(self.alpha_frac_above),
                    "kbm_gate_on":      gate_now,
                    "dne_rel":          float(dn_rel),
                })
                if v:
                    print(
                        f"[picard] it {it:3d}: |dne|_rel = {dn_rel:.3e}, "
                        f"alpha_bar = {self.alpha_bar_ped:.4f}, "
                        f"frac(alpha>crit) = {self.alpha_frac_above:.2f}, "
                        f"KBM {'ON' if gate_now else 'OFF'}"
                    )

                if dn_rel < float(picard_rtol) and gate_now == prev_gate:
                    picard_converged = True
                    break
                prev_hat_ne = hat_ne_new
                prev_gate = gate_now

            self.picard_info = {
                "converged":  picard_converged,
                "iterations": n_picard,
                "gate_mode":  picard_gate_mode,
                "history":    picard_history,
            }
            self.kbm_info["picard_converged"] = picard_converged
            self.kbm_info["picard_iterations"] = n_picard
            if not picard_converged:
                raise RuntimeError(
                    f"[picard] Picard loop did not converge in "
                    f"{picard_max_it} iterations "
                    f"(last |dne|_rel = {picard_history[-1]['dne_rel']:.3e}); "
                    "consider increasing picard_max_it or setting "
                    "picard_relax < 1."
                )
        else:
            _solve_once()

        # ------------------------------------------------------------------
        # Extract converged hat profiles, then recover SI profiles.
        # ------------------------------------------------------------------
        hat_ne_fd, hat_nFC_fd, hat_nCX_fd = u.subfunctions
        sort_idx = np.argsort(hat_x_dofs)
        self.hat_x_sol   = hat_x_dofs[sort_idx]
        self.hat_ne_sol  = hat_ne_fd .dat.data[sort_idx]
        self.hat_nFC_sol = hat_nFC_fd.dat.data[sort_idx]
        self.hat_nCX_sol = hat_nCX_fd.dat.data[sort_idx]

        # SI profiles -- same attribute names as parent solver so plotting
        # / EPED feed / etc. all keep working unchanged.
        self.x_sol   = self.hat_x_sol  * self._L_nd
        self.ne_sol  = self.hat_ne_sol  * self._n0_nd
        self.nFC_sol = self.hat_nFC_sol * self._n0_nd
        self.nCX_sol = self.hat_nCX_sol * self._n0_nd

        self.u_fd = u
        self.W_fd = W
        self.V_fd = V

        if ne_grad_bc_loc == "outer":
            # The inner slope was the unknown in this mode; report the
            # value the separatrix-gradient condition selected for it.
            self.dne_dx_inner_solved = (
                float(hat_dne_dx_inner_c) * self._n0_nd / self._L_nd
            )
            if v:
                print(f"[grad-bc] converged in "
                      f"{self.grad_bc_info['iterations']} iterations; free "
                      f"dne/dx(x_inner) = "
                      f"{self.dne_dx_inner_solved:.3e} m^-4")

        if v:
            print(
                f"[nondim] hat residuals solved.\n"
                f"  hat_n_e  in [{self.hat_ne_sol.min():.3e}, {self.hat_ne_sol.max():.3e}]\n"
                f"  hat_n_FC in [{self.hat_nFC_sol.min():.3e}, {self.hat_nFC_sol.max():.3e}]\n"
                f"  hat_n_CX in [{self.hat_nCX_sol.min():.3e}, {self.hat_nCX_sol.max():.3e}]"
            )

        return self._build_result_dict('3D', 'firedrake')


def __getattr__(name):
    """Resolve the pre-refactor class name to the assembled class.

    Deferred rather than a top-level import: ``solver_api`` imports this
    module, so binding the name at import time would be a cycle.
    """
    if name == 'saarelma_connor_nondim':
        from src.solver_api import saarelma_connor
        return saarelma_connor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
