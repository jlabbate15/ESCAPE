"""Non-dimensional solvers for the ORIGINAL (implicit) Saarelma-Connor
pedestal model, in a Firedrake and a scipy implementation.

Model
=====
This module solves the *original* Saarelma-Connor model: the single
implicit second-order ODE for the electron density that is obtained
after the two neutral fluids have been eliminated analytically
(S. Saarelma et al 2023 Nucl. Fusion 63 052002; Eqs. (6)-(7) of the
Labbate APAM E9301 report; rigorous derivations in
docs/derivation_eq16.tex and docs/derivation_eq15.tex).  Only the
electron density is solved for -- the neutrals never appear as unknowns,
they enter through the closed-form closures already substituted into
Eqs. (6)-(7).

Step 1 -- the no-CX "first step" (report Eq. 6 == Saarelma Eq. 16):

    d/dx [ <|grad r|^2> D_ped dn_e/dx ]
        = n_e D_ped (S_i + S_CX) / (|V_FC| f_FC)
              * ( dn_e/dx - dn_e/dx|_in )                    (eq6_form="paper")

    d/dx [ <|grad r|^2> D_ped dn_e/dx ]
        = n_e (S_i + S_CX) / (1 + S_CX/(2 S_i))
              * <|grad r|^2> D_ped / (|V_FC| f_FC)
              * ( dn_e/dx - dn_e/dx|_in )               (eq6_form="complete")

``eq6_form`` selects between the equation exactly as printed and the
complete form of docs/derivation_eq16.tex Eq. (16-full), from which the
printed one follows by dropping the shape factor 1 + S_CX/(2 S_i)
(assumption A5) and by cancelling the <|grad r|^2> of the transport
operator against the D_ped of the neutral closure (a *symbolic*
cancellation that is only legitimate when <|grad r|^2> ~ 1).  The two
differ by the factor (1 + S_CX/(2 S_i)) / <|grad r|^2>, which is not a
detail: on the DIII-D 158091 case shipped with this repository
<|grad r|^2> ~ 0.05, so the printed form is ~35x stiffer than the
complete one -- its solution amplifies by exp(176) across the pedestal
and neither implementation can solve it in double precision, whereas
the complete form amplifies by ~70 and converges.  ``"complete"`` is
therefore the default; it is also the form consistent with Eq. (7)
below, which keeps <|grad r|^2> D_ped in the same place.

Step 2 -- the full implicit equation (report Eq. 7 == the corrected
Saarelma Eq. 15):

    d/dx [ <|grad r|^2> D_ped dn_e/dx ]
        = - n_e S_i { - <|grad r|^2> D_ped / (|V_CX| f_CX)
                          * ( dn_e/dx - dn_e/dx|_in )
                      + C_CX(x) <n_FC>(0) E(x) }

with the CX-flux coefficient and the FC attenuation ("optical depth")
kernel

    C_CX(x) = 1 - ( |V_FC| f_FC / (|V_CX| f_CX) )
                  * (S_i + S_CX/2) / (S_i + S_CX)
    E(x)    = exp[ int_0^x n_e(x') (S_i + S_CX) / (f_FC |V_FC|) dx' ].

(The form factors f_FC and f_CX are carried explicitly.  The report
writes Eq. 6 with f_FC = 1 -- assumption A4 of docs/derivation_eq16.tex,
which is what ``saarelma_connor.form_factor`` sets -- so the two forms
agree identically for the current form-factor model.)

The pedestal diffusivity is

    D_ped(x, n_e) = D_NEO(x) + D_KBM(x) + C_ETG(x) / n_e,

with C_ETG = De_chie_etg * P_tot_e / (S_plasma |dT_e/dx|) and D_KBM from
the pedestal-averaged Connor-Hastie alpha (Saarelma Eqs. 24-25); see
:meth:`saarelma_connor_sc.calc_D_KBM_average_sc`.

Implicit (Picard) solve
=======================
Following Saarelma et al. (2023), the non-local kernel E(x) makes Eq. (7)
an integro-differential equation, so it is solved implicitly:

    1. Solve Eq. (6) (no CX neutrals) for a first n_e profile.
    2. Iterate Eq. (7): on each Picard iteration E(x) is evaluated with
       the *previous* n_e and frozen, and D_KBM is refrozen from the
       latest n_e using the pedestal-averaged ("average" gate mode)
       alpha -- exactly the treatment of
       ``solver_nondim.solve_coupled_nondim`` with
       ``kbm_treatment="picard"`` / ``picard_gate_mode="average"``
       (frozen scalar-gated diffusivity, optional under-relaxation).
       Repeat until the profile and the gate state stop changing.

Two practical notes on this loop, both observed on the DIII-D 158091
case and both controllable from the solver signature:

  * Step 1 can have *no solution*.  Eq. (6) is a positive-feedback
    amplifier (more density gradient -> larger implied <n_FC> -> more
    ionisation -> more gradient), and for a steep enough inner-boundary
    slope every profile collapses to n_e ~ 0 before reaching the
    separatrix, so no member of the family satisfies n_e(0) = ne_x0.
    Saarelma et al. only use Eq. (16) as "a convenient starting point",
    so ``first_step`` selects "eq6" (always run it, raise on failure),
    "skip" (go straight to the Eq. 7 Picard loop from ``initial_guess``)
    or "auto" (default: try Eq. 6, warn and fall back to "skip").
  * The KBM gate can limit-cycle.  When the converged alpha_bar sits
    close to alpha_crit the whole-pedestal gate flips every iteration
    and the profile alternates between two states forever.  Damp it with
    ``picard_relax`` < 1 (0.5 works on the case above); the
    non-convergence error message diagnoses this explicitly.

Non-dimensionalisation
======================
    xi = x / L,    L  = |x_inner|,                 xi in [-1, 0]
    N  = n_e / n0, n0 = ne_x0,                     N(0) = 1
    S0 = S_i(0)   (separatrix ionisation rate coefficient)
    [V]_0 = L n0 S0,        [D]_0 = L^2 n0 S0

Writing f(x, n_e) = <|grad r|^2> D_ped for the total conductance and
splitting off its n_e dependence, f = f0(x) + f1(x)/n_e with

    f0 = <|grad r|^2> (D_NEO + D_KBM),   f1 = <|grad r|^2> C_ETG,

the expanded non-dimensional ODEs solved by the scipy implementation are

    Eq. 6:  N'' = C_A6 N (N' - N'_in) - C_K N' + C_N (N')^2
    Eq. 7:  N'' = C_A  N (N' - N'_in) - C_E N - C_K N' + C_N (N')^2

    C_A6 = n0 L (S_i + S_CX) / (<|grad r|^2> |V_FC| f_FC)      ("paper")
         = n0 L (S_i + S_CX) / ((1 + S_CX/(2 S_i)) |V_FC| f_FC)
                                                            ("complete")
    C_A  = n0 L S_i / (|V_CX| f_CX)
    C_E  = L^2 S_i C_CX <n_FC>(0) E(x) / f
    C_K  = (L / f) ( df0/dx + (1/n_e) df1/dx )
    C_N  = f1 / (n0 N^2 f)

where N'_in = (L/n0) dn_e/dx|_in is the non-dimensional Neumann value
(cf. docs/derivation_eq16.tex Sec. "Non-dimensionalization").

The Firedrake implementation keeps the conservative form.  With
hat_f = f/[D]_0, hat_S = S/S0, hat_V = V/[V]_0 and hat_nFC0 = nFC_x0/n0
the strong forms are

    Eq. 6:  d/dxi[hat_f N'] = N (hat_f / <|grad r|^2>)
                (hat_S_i + hat_S_CX) (N' - N'_in) / (hat_V_FC f_FC)
                                                            ("paper")
            d/dxi[hat_f N'] = N hat_f (hat_S_i + hat_S_CX)
                / (1 + hat_S_CX/(2 hat_S_i))
                * (N' - N'_in) / (hat_V_FC f_FC)         ("complete")
    Eq. 7:  d/dxi[hat_f N'] = N hat_S_i [ hat_f (N' - N'_in)
                / (hat_V_CX f_CX) - C_CX hat_nFC0 E(xi) ]

which are multiplied by a test function w and integrated by parts in the
usual way (the residuals are assembled in
:meth:`saarelma_connor_sc.solve_sc_firedrake`).

Agreement between the two implementations
-----------------------------------------
On the DIII-D 158091 case the converged profiles agree to ~3e-3 in
max-norm, and each satisfies its own discrete Eq. (7) far better than
that (the conservative residual of the Firedrake solution falls as the
mesh is refined, ~1e-3 at x_res = 400; the scipy solution sits at ~2e-4
of its own collocation problem).  The residual difference between them
is dominated not by the discretisation but by how each reconstructs the
equilibrium coefficients between the points of the (coarse) p-file
grid: the Firedrake path samples them with ``np.interp`` into the finite
element space, as the rest of this repository does, while the scipy path
needs df/dx and therefore uses a shape-preserving PCHIP interpolant.

Boundary conditions (the only supported combination):

    Neumann   at xi = -1 (inner):  N'(-1) = (L/n0) dn_e/dx|_in
    Dirichlet at xi =  0 (outer):  N(0)   = 1  (i.e. n_e(0) = ne_x0)

Free parameters: alpha_crit, De_chie_etg, C_KBM, nFC_x0 -- set through
the parent constructor, through ``update_free_params``, or per solve
through the ``free_params`` argument of either entry point.

This module subclasses :class:`src.solver.saarelma_connor`, so all
equilibrium / kinetic / cross-section initialisation is inherited
unchanged; ``src/solver.py`` is left untouched.

Entry points
============
    saarelma_connor_sc.solve_sc_scipy(...)      -- scipy solve_bvp
    saarelma_connor_sc.solve_sc_firedrake(...)  -- Firedrake / SNES
    saarelma_connor_sc.solve_sc(...)            -- thin dispatcher
"""

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.integrate import cumulative_trapezoid, solve_bvp

try:
    from firedrake import (
        IntervalMesh, FunctionSpace, Function, TestFunction,
        Constant, DirichletBC, dx, ds, solve, SpatialCoordinate,
    )
    _FIREDRAKE_AVAILABLE = True
    _FIREDRAKE_IMPORT_ERR = None
except Exception as _firedrake_import_err:
    _FIREDRAKE_AVAILABLE = False
    _FIREDRAKE_IMPORT_ERR = _firedrake_import_err

from src.solver import saarelma_connor


# Conversion constant (eV -> J)
_EV2J = 1.60218e-19

# Free parameters of the model (all four must be set before solving).
_FREE_PARAM_NAMES = ("alpha_crit", "De_chie_etg", "C_KBM", "nFC_x0")


class saarelma_connor_sc(saarelma_connor):
    """Original (implicit) Saarelma-Connor model, non-dimensionalised.

    Solves the single second-order ODE for n_e only (report Eqs. (6)-(7)
    / Saarelma Eq. 16 and corrected Eq. 15) with the two-step implicit
    scheme of the original paper, in two interchangeable implementations:

        solve_sc_scipy      -- scipy.integrate.solve_bvp
        solve_sc_firedrake  -- Firedrake (CG finite elements, SNES Newton)

    Both share the initialisation machinery of the parent class
    (``setup_solver_grids``, ``form_factor``, ``find_inner_boundary``,
    ``construct_C_ETG``) and the same Picard "average"-gate pedestal
    pressure feature as ``solver_nondim`` (KBM diffusivity refrozen every
    Picard iteration from the pedestal-averaged alpha of the latest n_e;
    cf. ``solver_nondim.py`` lines 1248-1262).

    Free parameters: alpha_crit, De_chie_etg, C_KBM, nFC_x0.
    Boundary conditions: Neumann at the inner boundary, Dirichlet
    (n_e = ne_x0) at the separatrix.  No other combination is supported.

    After a successful solve the following attributes are set:

        self.x_sol, self.ne_sol, self.dne_dx_sol   : SI profiles
        self.hat_x_sol, self.hat_ne_sol            : non-dim profiles
        self.E_sol                                 : converged FC kernel
                                                     E(x) on x_sol
        self.alpha_bar_ped, self.kbm_gate_on       : final KBM gate state
        self.picard_info, self.kbm_info            : iteration diagnostics
        self._L_sc, self._n0_sc, self._S0_sc,
        self._V0_sc, self._D0_sc                   : reference scales
    """

    # ------------------------------------------------------------------
    # Shared setup
    # ------------------------------------------------------------------

    def _check_free_params_sc(self):
        """Raise if any of the four free parameters is unset."""
        missing = [n for n in _FREE_PARAM_NAMES
                   if getattr(self, n, None) is None]
        if missing:
            raise ValueError(
                "The original Saarelma-Connor model needs all four free "
                f"parameters; missing: {', '.join(missing)}.  Set them on "
                "the constructor, with update_free_params(), or pass "
                "free_params={'alpha_crit': ..., 'De_chie_etg': ..., "
                "'C_KBM': ..., 'nFC_x0': ...} to the solver."
            )

    def _ensure_sc_setup(self, x_res, free_params=None, force=False):
        """Equilibrium-only setup shared by both implementations.

        Optionally updates the free parameters, then runs the parent
        initialisation methods (solver grids, form factors, ETG
        coefficient), locates the inner boundary, and computes the
        non-dimensional reference scales.

        Parameters
        ----------
        x_res : int
            Resolution handed to ``setup_solver_grids``.
        free_params : dict or None
            Optional ``{'alpha_crit', 'De_chie_etg', 'C_KBM', 'nFC_x0'}``
            (plus any keyword accepted by ``update_free_params``, e.g.
            ``psi_N_inner_boundary``) applied before the setup.
        force : bool
            Re-run ``setup_solver_grids`` / ``form_factor`` even if they
            were already built for this ``x_res``.
        """
        if not hasattr(self, "_fd_cache"):
            self._fd_cache = {}

        if free_params is not None:
            fp = dict(free_params)
            self.update_free_params(
                fp.pop("alpha_crit", self.alpha_crit),
                fp.pop("C_KBM", self.C_KBM),
                fp.pop("De_chie_etg", self.De_chie_etg),
                fp.pop("nFC_x0", self.nFC_x0),
                **fp,
            )
        self._check_free_params_sc()

        # form_factor + setup_solver_grids (cached per x_res), exactly as
        # the parent's coupled solver does, then the ETG coefficient
        # (which depends on the free parameter De_chie_etg).
        self._ensure_firedrake_coefficient_grids(x_res, force=force)
        self.construct_C_ETG()

        # Inner boundary location (same logic as solver_nondim).
        if self.psi_N_inner_boundary is None:
            if not hasattr(self, "_D_KBM"):
                # find_inner_boundary needs a D_KBM estimate; a zeroed
                # array (KBM off) is the conservative pre-solve choice.
                self._D_KBM = np.zeros_like(self.x_init)
            self.find_inner_boundary()
        else: # should mostly be using this branch
            self.x_inner = float(np.interp(
                self.psi_N_inner_boundary, self.psi_N_pres, self.x_init
            ))

        if float(self.x_inner) >= 0.0:
            raise ValueError(
                f"x_inner = {self.x_inner} must be strictly less than 0 "
                "(separatrix)."
            )

        # Reference scales.
        self._L_sc = float(abs(self.x_inner))
        self._n0_sc = float(self.ne_x0)
        S0 = float(self.S_i_pres[-1])
        if not np.isfinite(S0) or S0 <= 0.0:
            raise RuntimeError(
                f"S_i(0) = {S0!r} m^3/s is not a usable separatrix "
                "ionisation rate; cannot non-dimensionalise."
            )
        self._S0_sc = S0
        self._V0_sc = self._L_sc * self._n0_sc * S0          # m/s
        self._D0_sc = (self._L_sc ** 2) * self._n0_sc * S0   # m^2/s

    def _sc_neumann_inner_value(self, bc_origin, dne_dx_inner):
        """Return the SI inner-boundary slope dn_e/dx|_in (m^-4).

        Only the Neumann inner / Dirichlet outer combination is
        supported, so this is the single boundary-condition degree of
        freedom.  Also stored as ``self.dne_dx_inner`` and (for
        compatibility with the parent class's post-processing helpers)
        ``self.dne_dx_neginf``.
        """
        bc_origin = str(bc_origin).lower()
        if bc_origin == "p-file":
            dne_dx_pres = np.gradient(self.n_e_pres, self.x_init)
            val = float(np.interp(self.x_inner, self.x_init, dne_dx_pres))
        elif bc_origin == "user":
            if dne_dx_inner is None:
                raise ValueError(
                    "bc_origin='user' requires dne_dx_inner to be given."
                )
            val = float(dne_dx_inner)
        else:
            raise ValueError(
                f"bc_origin must be 'p-file' or 'user', got {bc_origin!r}."
            )
        if val >= 0.0: # this is a likely but possible unnecessary hard stop, will keep in for now
            raise ValueError(
                f"dne/dx(x_inner) = {val:.3e} m^-4 must be strictly "
                "negative (density decreasing outward) for the Neumann "
                "inner boundary condition."
            )
        self.dne_dx_inner = val
        self.dne_dx_neginf = val
        return val

    def _sc_initial_guess(self, x_grid, initial_guess,
                          tanh_width, tanh_center, ne_inner_guess):
        """SI electron-density initial guess on ``x_grid`` (ascending).

        Same options as ``solver_nondim``: 'pfile', 'linear', 'tanh'.
        ``ne_inner_guess`` is the pedestal-top density used by the
        'linear' / 'tanh' shapes (derived from the Neumann slope).
        """
        x_left = float(x_grid[0])
        if initial_guess == "pfile":
            return np.interp(x_grid, self.x_init, self.n_e_pres)
        if initial_guess == "linear":
            s = (x_grid - x_left) / (0.0 - x_left)
            return ne_inner_guess + (self.ne_x0 - ne_inner_guess) * s
        if initial_guess == "tanh":
            width = (float(tanh_width) if tanh_width is not None
                     else 0.1 * abs(x_left))
            if width <= 0:
                raise ValueError(f"tanh_width must be positive, got {width}.")
            center = float(tanh_center) if tanh_center is not None else -width
            s_ne = 0.5 * (1.0 - np.tanh((x_grid - center) / (0.5 * width)))
            return self.ne_x0 + (ne_inner_guess - self.ne_x0) * s_ne
        raise ValueError(
            f"Unknown initial_guess={initial_guess!r}; expected "
            "'pfile', 'linear', or 'tanh'."
        )

    # ------------------------------------------------------------------
    # Picard averaged-pressure KBM diffusivity (Saarelma Eqs. 24-25)
    # ------------------------------------------------------------------

    def calc_D_KBM_average_sc(self, n_e_ped, x_ped):
        """Pedestal-averaged-alpha KBM diffusivity on the x_init grid.

        This is the "average" Picard gate mode of
        :meth:`solver_nondim.saarelma_connor_nondim.calc_pressure_quantities_nondim`
        specialised to the original model: the Connor-Hastie alpha is
        evaluated locally from the given pedestal density profile,

            alpha(x) = alpha_nodp(x) * d/dx [ n_e (T_e + T_i) e ],

        averaged over the pedestal, and gated as a whole:

            gate  = alpha_bar > alpha_crit
            D_KBM = (alpha_bar - alpha_crit) * C_KBM c_s rho_s^2 / a
                    (everywhere, when the gate is on; zero otherwise).

        Parameters
        ----------
        n_e_ped : array_like
            Electron density (m^-3) on ``x_ped``.
        x_ped : array_like
            Ascending pedestal grid (m), x in [x_inner, 0].

        Returns
        -------
        D_KBM_xinit : ndarray
            KBM diffusivity (m^2/s) on ``self.x_init`` (used to build
            the coefficient interpolators / Functions).

        Sets
        ----
        self.alpha_bar_ped : float
        self.alpha_frac_above : float
        self.kbm_gate_on   : bool
        self.alpha_local_ped : ndarray  (local alpha on x_ped)
        self._D_KBM        : ndarray    (same as the return value; also
                                         keeps find_inner_boundary
                                         re-entrant)
        """
        x_ped = np.asarray(x_ped, dtype=float)
        n_e_ped = np.asarray(n_e_ped, dtype=float)

        T_e = np.interp(x_ped, self.x_init, self.T_e_pres)   # eV
        T_i = np.interp(x_ped, self.x_init, self.T_i_pres)   # eV
        alpha_nodp = np.interp(x_ped, self.x_init, self._alpha_nodp_xinit)

        # Pa; neutral pressures neglected, quasi-neutral plasma assumed.
        _pres = n_e_ped * (T_e + T_i) * _EV2J
        dpdx = np.gradient(_pres, x_ped)                     # Pa/m
        _alpha = alpha_nodp * dpdx                           # dimensionless

        alpha_bar = float(np.mean(_alpha))
        gate = alpha_bar > self.alpha_crit
        self.alpha_bar_ped = alpha_bar
        self.alpha_frac_above = float(np.mean(_alpha > self.alpha_crit))
        self.kbm_gate_on = bool(gate)
        self.alpha_local_ped = _alpha
        self.pres_ped = _pres

        G_KBM_xinit = self.C_KBM * (self.c_s * self.rho_s ** 2) / self.a
        D_KBM_xinit = np.where(
            gate, (alpha_bar - self.alpha_crit) * G_KBM_xinit, 0.0
        )
        self._D_KBM = D_KBM_xinit
        return D_KBM_xinit

    # ------------------------------------------------------------------
    # FC attenuation kernel  E(x)  (exponent of Saarelma Eq. 11)
    # ------------------------------------------------------------------

    def _exp_kernel_sc(self, n_e, x):
        """E(x) = exp[ int_0^x n_e (S_i + S_CX) / (f_FC |V_FC|) dx' ].

        ``x`` is an ascending grid (x_inner ... 0).  Returns E on the
        same grid; E(0) = 1 and E decays monotonically inward.  This is
        the same closure the parent class uses for <n_FC> in
        ``find_inner_boundary`` / ``compute_post_solve_SC_neutrals``,
        i.e. <n_FC>(x) = nFC_x0 * E(x).
        """
        x = np.asarray(x, dtype=float)
        n_e = np.asarray(n_e, dtype=float)
        Si = np.interp(x, self.x_init, self.S_i_pres)
        Scx = np.interp(x, self.x_init, self.S_cx_pres)
        fFC = np.interp(x, self.x_init, self.fFC)
        integrand = n_e * (Si + Scx) / (fFC * abs(self.V_FC))

        # int_0^x with x < 0: integrate on a descending ordering
        # (separatrix -> core) so the cumulative integral starts at 0.
        order_desc = np.argsort(x)[::-1]
        cum_desc = cumulative_trapezoid(
            integrand[order_desc], x[order_desc], initial=0.0
        ) # results in a negative value for the sum, since integrand should be positive and integrating from 0 to x < 0
        integral_from_0 = np.empty_like(cum_desc)
        integral_from_0[order_desc] = cum_desc
        return np.exp(integral_from_0)

    def _C_cx_sc(self, x):
        """CX-flux coefficient C_CX(x) of report Eq. (7), on ``x`` (m).

            C_CX = 1 - ( |V_FC| f_FC / (|V_CX| f_CX) )
                       * (S_i + S_CX/2) / (S_i + S_CX)

        Purely a ratio of velocities and rate coefficients, so it is
        scale-invariant and built once in SI.
        """
        x = np.asarray(x, dtype=float)
        Si = np.interp(x, self.x_init, self.S_i_pres)
        Scx = np.interp(x, self.x_init, self.S_cx_pres)
        Vcx = np.interp(x, self.x_init, np.abs(self.V_cx_pres))
        fFC = np.interp(x, self.x_init, self.fFC)
        fCX = np.interp(x, self.x_init, self.fCX)
        return 1.0 - (abs(self.V_FC) * fFC / (Vcx * fCX)) * (
            (Si + 0.5 * Scx) / (Si + Scx)
        )

    # ------------------------------------------------------------------
    # Picard bookkeeping shared by both implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _check_eq6_form_sc(eq6_form):
        """Validate and normalise the ``eq6_form`` switch."""
        eq6_form = str(eq6_form).lower()
        if eq6_form not in ("complete", "paper"):
            raise ValueError(
                "eq6_form must be 'complete' (docs/derivation_eq16.tex "
                "Eq. 16-full) or 'paper' (report Eq. 6 / Saarelma Eq. 16 "
                f"as printed), got {eq6_form!r}."
            )
        if eq6_form == "paper":
            import warnings
            warnings.warn(
                "eq6_form='paper' is the report Eq. 6 / Saarelma Eq. 16 "
                "as printed, which drops the shape factor "
                "1 + S_CX/(2 S_i) and cancels <|grad r|^2> against D_ped.  "
                "On the DIII-D 158091 case this makes it ~35x stiffer than "
                "the complete form (docs/derivation_eq16.tex Eq. 16-full) "
                "and no implementation can solve it in double precision.  "
                "Use eq6_form='complete' instead.",
                RuntimeWarning, stacklevel=3
            )
        return eq6_form

    @staticmethod
    def _check_first_step_sc(first_step):
        """Validate and normalise the ``first_step`` switch."""
        first_step = str(first_step).lower()
        if first_step not in ("auto", "eq6", "skip"):
            raise ValueError(
                "first_step must be 'auto', 'eq6', or 'skip', got "
                f"{first_step!r}."
            )
        return first_step

    def _first_step_failed_sc(self, tag, first_step, err):
        """Handle a failed step 1: re-raise, or warn and fall back.

        Eq. (6) is a positive-feedback amplifier -- every bit of extra
        density gradient increases the implied <n_FC>, which steepens the
        gradient further -- so for a steep enough inner-boundary slope it
        has no solution that still reaches n_e(0) = ne_x0 (the profile
        collapses first).  Saarelma et al. (2023) only use it as "a
        convenient starting point" for the Eq. (7) iteration, so when it
        fails the Picard loop can equally well start from the initial
        guess.
        """
        msg = (
            f"[{tag}] the first step (Eq. 6, eq6_form="
            f"{getattr(self, 'eq6_form', '?')!r}) has no usable solution "
            f"for this case: {err}.  Eq. (6) is a positive-feedback "
            "amplifier and admits no profile reaching n_e(0) = ne_x0 when "
            "the inner-boundary gradient, the pedestal width, or the "
            "D_ped shape make the amplification exp(int C_A6 N dxi) too "
            "large.  Remedies: first_step='skip' (start the Eq. 7 Picard "
            "loop from the initial guess, which is all Saarelma et al. "
            "use Eq. 16 for), a shallower dne_dx_inner, or an inner "
            "boundary closer to the separatrix."
        )
        if first_step == "eq6":
            raise RuntimeError(msg) from err
        import warnings
        warnings.warn(msg + "  Falling back to first_step='skip'.",
                      RuntimeWarning, stacklevel=3)
        self.first_step_used = "skip (Eq. 6 failed)"
        return False

    def _record_picard_sc(self, tag, converged, n_it, history,
                          picard_max_it, picard_relax, picard_rtol):
        """Store ``picard_info`` / ``kbm_info`` and raise if diverged."""
        self.picard_info = {
            "converged": converged,
            "iterations": n_it,
            "gate_mode": "average",
            "history": history,
        }
        self.kbm_info = {
            "treatment": "picard",
            "picard_gate_mode": "average",
            "eq6_form": getattr(self, "eq6_form", None),
            "first_step": getattr(self, "first_step_used", None),
            "alpha_crit": float(self.alpha_crit),
            "picard_max_it": int(picard_max_it),
            "picard_rtol": float(picard_rtol),
            "picard_relax": float(picard_relax),
            "picard_converged": converged,
            "picard_iterations": n_it,
        }
        if not converged:
            last = history[-1]["dne_rel"] if history else float("nan")
            msg = (
                f"[{tag}] Picard loop did not converge in {picard_max_it} "
                f"iterations (last |dne|_rel = {last:.3e})"
            )
            # The usual culprit is a KBM gate limit cycle: alpha_bar
            # straddles alpha_crit, so the gate (and with it D_KBM) flips
            # every iteration and the profile alternates between two
            # states.  Diagnose it explicitly -- the fix is different
            # from "just iterate longer".
            gates = [h["kbm_gate_on"] for h in history[-6:]]
            if len(set(gates)) > 1:
                a_lo = min(h["alpha_bar"] for h in history[-6:])
                a_hi = max(h["alpha_bar"] for h in history[-6:])
                msg += (
                    f".  The KBM gate is flip-flopping: alpha_bar cycles "
                    f"over [{a_lo:.4f}, {a_hi:.4f}] across "
                    f"alpha_crit = {float(self.alpha_crit):.4f}, so the "
                    "frozen D_KBM alternates between its on and off "
                    "branches.  Damp it with picard_relax < 1 (e.g. 0.5), "
                    "or move alpha_crit away from alpha_bar"
                )
            else:
                msg += (
                    "; consider increasing picard_max_it or setting "
                    "picard_relax < 1"
                )
            raise RuntimeError(msg + ".")

    # ------------------------------------------------------------------
    # scipy implementation
    # ------------------------------------------------------------------

    def solve_sc_scipy(self,
                       x_res=200,
                       free_params=None,
                       bc_origin="p-file",
                       dne_dx_inner=None,
                       initial_guess="pfile",
                       tanh_width=None,
                       tanh_center=None,
                       eq6_form="complete",
                       first_step="auto",
                       picard_max_it=50,
                       picard_rtol=1e-6,
                       picard_relax=1.0,
                       bvp_tol=1e-6,
                       bvp_max_nodes=5000,
                       reuse_setup=True,
                       verbose=None):
        """Implicit scipy (``solve_bvp``) solver for the original model.

        Step 1 solves the non-dimensional no-CX equation (report Eq. 6)
        for a first n_e; step 2 Picard-iterates the full equation
        (report Eq. 7) with the exponential kernel E(x) and the
        pedestal-averaged KBM diffusivity frozen from the previous
        iterate, until the profile and the KBM gate stop changing.

        Parameters
        ----------
        x_res : int
            Number of points of the (uniform) non-dimensional solver
            grid handed to ``solve_bvp`` as its initial mesh.
        free_params : dict or None
            Optional free parameters applied before solving; see
            :meth:`_ensure_sc_setup`.
        bc_origin : {"p-file", "user"}
            Where the Neumann inner slope comes from: the p-file density
            gradient at x_inner, or the user-supplied ``dne_dx_inner``.
        dne_dx_inner : float or None
            SI inner-boundary slope (m^-4, must be negative) when
            ``bc_origin="user"``.
        initial_guess : {"pfile", "linear", "tanh"}
            Shape of the initial n_e profile (SI), as in solver_nondim.
        tanh_width, tanh_center : float or None
            Parameters of the "tanh" initial guess (SI metres).
        eq6_form : {"complete", "paper"}
            Which form of the first step to solve; see the module
            docstring.  "paper" is Eq. 6 exactly as printed and is only
            usable when <|grad r|^2> is close to 1: otherwise it either
            fails outright or converges to a collapsed profile that the
            Eq. (7) iteration cannot then start from.
        first_step : {"auto", "eq6", "skip"}
            Whether to run step 1 at all.  "eq6" always runs it and
            raises if it fails; "skip" starts the Eq. (7) Picard loop
            directly from ``initial_guess``; "auto" (default) tries
            Eq. (6) and falls back to "skip" with a ``RuntimeWarning``
            if it has no solution (see :meth:`_first_step_failed_sc`).
        picard_max_it : int
            Maximum number of Picard iterations of the full Eq. (7).
        picard_rtol : float
            Relative max-norm tolerance on the change in n_e between
            Picard iterations.
        picard_relax : float in (0, 1]
            Under-relaxation of the frozen D_KBM and E(x) updates.
        bvp_tol : float
            ``solve_bvp`` residual tolerance.
        bvp_max_nodes : int
            ``solve_bvp`` maximum number of mesh nodes.
        reuse_setup : bool
            Reuse cached solver grids / form factors when the resolution
            is unchanged (set False after changing the equilibrium).
        verbose : bool or None
            Override ``self.verbose``.

        Returns
        -------
        x_sol, ne_sol, dne_dx_sol : ndarray
            SI radial grid (m), electron density (m^-3), and its
            gradient (m^-4) of the converged solution.
        """
        v = self.verbose if verbose is None else bool(verbose)
        picard_relax = float(picard_relax)
        if not (0.0 < picard_relax <= 1.0):
            raise ValueError(
                f"picard_relax must be in (0, 1], got {picard_relax}."
            )
        eq6_form = self._check_eq6_form_sc(eq6_form)
        self.eq6_form = eq6_form
        first_step = self._check_first_step_sc(first_step)
        self.first_step_used = first_step

        self._ensure_sc_setup(x_res, free_params=free_params,
                              force=not reuse_setup)
        L = self._L_sc
        n0 = self._n0_sc
        dne_dx_inner_val = self._sc_neumann_inner_value(bc_origin, dne_dx_inner)
        dN_in = (L / n0) * dne_dx_inner_val   # non-dim Neumann value

        # Frozen equilibrium interpolators (SI, functions of x).
        def _mk(arr):
            return interp1d(self.x_init, arr, kind='linear',
                            bounds_error=False, fill_value='extrapolate')

        g_x = _mk(self.gradr2_fsa)
        Si_x = _mk(self.S_i_pres)
        Scx_x = _mk(self.S_cx_pres)
        Vcx_x = _mk(np.abs(self.V_cx_pres))
        fFC_x = _mk(self.fFC)
        fCX_x = _mk(self.fCX)
        Vfc = abs(self.V_FC)

        # Non-dimensional solver grid and initial guess.
        xi_grid = np.linspace(-1.0, 0.0, int(x_res))
        x_grid = L * xi_grid
        ne_inner_guess = self.ne_x0 + dne_dx_inner_val * self.x_inner
        ne_init = self._sc_initial_guess(
            x_grid, initial_guess, tanh_width, tanh_center, ne_inner_guess,
        )
        self.ne_init = ne_init

        if v:
            print(f"[sc scipy] x_inner         = {self.x_inner:.4e} m")
            print(f"[sc scipy] dne/dx(x_inner) = {dne_dx_inner_val:.3e} m^-4"
                  f"  ({bc_origin})")
            print(f"[sc scipy] N'(-1)          = {dN_in:.3e}")
            print(f"[sc scipy] scales: L = {L:.4e} m, n0 = {n0:.4e} m^-3, "
                  f"[D]_0 = {self._D0_sc:.4e} m^2/s")

        def _build_f_interp(D_KBM_xinit):
            """Interpolators for f0, f1 and their x-derivatives.

            f0 = <|grad r|^2>(D_NEO + D_KBM) (m^2/s),
            f1 = <|grad r|^2> C_ETG          (m^2/s * m^-3).

            The expanded ODE needs df0/dx and df1/dx (the C_K term), and
            the accuracy of the whole solve is limited by them: x_init is
            the p-file pressure grid, which typically carries only a few
            tens of points across the pedestal.  A shape-preserving
            (PCHIP) interpolant of f0 / f1 with its analytic derivative
            is markedly better there than np.gradient followed by linear
            interpolation, and unlike a natural cubic spline it cannot
            overshoot on C_ETG, which spans several decades over x_init.
            The conservative Firedrake form never differentiates f, so it
            has no equivalent of this term (see the note on
            implementation agreement in the module docstring).
            """
            f0_arr = self.gradr2_fsa * (self.D_NEO + D_KBM_xinit)
            f1_arr = self.gradr2_fsa * self.C_ETG
            f0_spl = PchipInterpolator(self.x_init, f0_arr, extrapolate=True)
            f1_spl = PchipInterpolator(self.x_init, f1_arr, extrapolate=True)
            return (f0_spl, f1_spl, f0_spl.derivative(),
                    f1_spl.derivative())

        def _conductance(xi, N, f0_x, f1_x, df0_x, df1_x):
            """Conductance pieces f, C_K, C_N at (xi, N), plus x.

            ``N`` is floored while it appears in a denominator, so that a
            solver excursion towards N <= 0 cannot produce a division by
            zero (the same guard as the parent class's first_step).
            """
            x = L * xi
            N_safe = np.maximum(N, 1e-6)
            f1 = f1_x(x)
            F = f0_x(x) + f1 / (n0 * N_safe)  # = <|grad r|^2> D_ped, m^2/s
            C_K = (L / F) * (df0_x(x) + df1_x(x) / (n0 * N_safe))
            C_N = f1 / (n0 * (N_safe ** 2) * F)
            return x, F, C_K, C_N

        def bc(Ya, Yb):
            return np.array([
                Ya[1] - dN_in,   # Neumann at xi = -1 (inner)
                Yb[0] - 1.0,     # Dirichlet at xi = 0: N = ne_x0/n0 = 1
            ])

        # --------------------------------------------------------------
        # Step 1: no-CX first step (report Eq. 6 / Saarelma Eq. 16),
        # with D_KBM frozen from the initial guess.
        # --------------------------------------------------------------
        self.calc_D_KBM_average_sc(ne_init, x_grid)
        D_KBM_frozen = self._D_KBM.copy()
        f0_x, f1_x, df0_x, df1_x = _build_f_interp(D_KBM_frozen)

        def ode_first(xi, Y):
            N, dN = Y
            x, _F, C_K, C_N = _conductance(
                xi, N, f0_x, f1_x, df0_x, df1_x,
            )
            Si = Si_x(x)
            Scx = Scx_x(x)
            if eq6_form == "paper":
                # Eq. 6 as printed: the RHS D_ped cancels the <|grad r|^2>
                # D_ped of the transport operator, leaving 1/<|grad r|^2>.
                C_A6 = n0 * L * (Si + Scx) / (g_x(x) * Vfc * fFC_x(x))
            else:
                # Eq. (16-full): <|grad r|^2> D_ped kept on both sides
                # (cancels exactly), shape factor 1 + S_CX/(2 S_i) kept.
                C_A6 = (n0 * L * (Si + Scx)
                        / ((1.0 + 0.5 * Scx / Si) * Vfc * fFC_x(x)))
            d2N = C_A6 * N * (dN - dN_in) - C_K * dN + C_N * dN ** 2
            return np.vstack([dN, d2N])

        dne_guess = np.gradient(ne_init, x_grid)
        Y_guess = np.vstack([ne_init / n0, dne_guess * (L / n0)])
        sol = None
        if first_step != "skip":
            sol = solve_bvp(ode_first, bc, xi_grid, Y_guess,
                            tol=bvp_tol, max_nodes=int(bvp_max_nodes),
                            verbose=self.bvp_verbose)
            err = None
            if not sol.success:
                err = RuntimeError(sol.message)
            elif sol.y[0].min() <= 0.0:
                # A "converged" collapsed profile is not a usable start:
                # Eq. 6 can be satisfied by a density that crosses zero
                # inside the pedestal, which no Eq. 7 iterate can recover
                # from (and which is unphysical anyway).
                err = RuntimeError(
                    f"the solution is not positive (min n_e = "
                    f"{n0 * sol.y[0].min():.3e} m^-3)"
                )
            if err is not None:
                self._first_step_failed_sc("sc scipy", first_step, err)
                sol = None
            else:
                self.sol_first = sol
                if v:
                    print(f"[sc scipy] first step (Eq. 6) converged, "
                          f"alpha_bar(guess) = {self.alpha_bar_ped:.4f}, "
                          f"KBM {'ON' if self.kbm_gate_on else 'OFF'}")

        # State carried between Picard iterations: the previous iterate on
        # its own (possibly adapted) grid.  Without step 1 that is simply
        # the initial guess on the uniform grid.
        if sol is None:
            x_prev = x_grid
            N_prev, dN_prev = Y_guess
            if v:
                print("[sc scipy] first step skipped; the Eq. 7 Picard loop "
                      f"starts from initial_guess={initial_guess!r}")
        else:
            x_prev = L * sol.x
            N_prev, dN_prev = sol.y[0], sol.y[1]

        # --------------------------------------------------------------
        # Step 2: Picard iterations of the full Eq. (7).
        # --------------------------------------------------------------
        picard_history = []
        picard_converged = False
        n_picard = 0
        E_frozen = None
        prev_gate = bool(self.kbm_gate_on)

        for it in range(1, int(picard_max_it) + 1):
            n_picard = it

            # Previous iterate in SI on its (possibly adapted) grid.
            ne_prev = n0 * N_prev

            # Refreeze E(x) and D_KBM (averaged-alpha gate) from the
            # previous iterate, with optional under-relaxation -- mirrors
            # solver_nondim's picard_gate_mode="average" branch.
            E_new = np.interp(
                x_grid, x_prev, self._exp_kernel_sc(ne_prev, x_prev)
            )
            if E_frozen is None:
                E_frozen = E_new
            else:
                E_frozen = (picard_relax * E_new
                            + (1.0 - picard_relax) * E_frozen)
            E_x = interp1d(x_grid, E_frozen, kind='linear',
                           bounds_error=False, fill_value='extrapolate')

            ne_prev_on_grid = np.interp(x_grid, x_prev, ne_prev)
            D_KBM_new = self.calc_D_KBM_average_sc(ne_prev_on_grid, x_grid)
            gate_now = bool(self.kbm_gate_on)
            D_KBM_frozen = (picard_relax * D_KBM_new
                            + (1.0 - picard_relax) * D_KBM_frozen)
            self._D_KBM = D_KBM_frozen
            f0_x, f1_x, df0_x, df1_x = _build_f_interp(D_KBM_frozen)

            def ode_full(xi, Y):
                N, dN = Y
                x, F, C_K, C_N = _conductance(
                    xi, N, f0_x, f1_x, df0_x, df1_x,
                )
                Si = Si_x(x)
                Scx = Scx_x(x)
                Vcx = Vcx_x(x)
                fCX = fCX_x(x)
                C_cx = 1.0 - (Vfc * fFC_x(x) / (Vcx * fCX)) * (
                    (Si + 0.5 * Scx) / (Si + Scx)
                )
                C_A = n0 * L * Si / (Vcx * fCX)
                C_E = (L ** 2) * Si * C_cx * self.nFC_x0 * E_x(x) / F
                d2N = (C_A * N * (dN - dN_in) - C_E * N
                       - C_K * dN + C_N * dN ** 2)
                return np.vstack([dN, d2N])

            # Warm start from the previous iterate on the uniform grid.
            N_guess = np.interp(x_grid, x_prev, N_prev)
            dN_guess = np.interp(x_grid, x_prev, dN_prev)
            sol = solve_bvp(ode_full, bc, xi_grid,
                            np.vstack([N_guess, dN_guess]),
                            tol=bvp_tol, max_nodes=int(bvp_max_nodes),
                            verbose=self.bvp_verbose)
            if not sol.success:
                raise RuntimeError(
                    f"[sc scipy] Picard iteration {it} BVP failed: "
                    f"{sol.message}"
                )
            x_prev = L * sol.x
            N_prev, dN_prev = sol.y[0], sol.y[1]

            ne_new_on_grid = n0 * np.interp(x_grid, x_prev, N_prev)
            dn_rel = (np.max(np.abs(ne_new_on_grid - ne_prev_on_grid))
                      / max(np.max(np.abs(ne_new_on_grid)), 1e-300))

            picard_history.append({
                "iteration": it,
                "alpha_bar": float(self.alpha_bar_ped),
                "alpha_frac_above": float(self.alpha_frac_above),
                "kbm_gate_on": gate_now,
                "dne_rel": float(dn_rel),
            })
            if v:
                print(
                    f"[sc scipy] picard it {it:3d}: |dne|_rel = {dn_rel:.3e}, "
                    f"alpha_bar = {self.alpha_bar_ped:.4f}, "
                    f"KBM {'ON' if gate_now else 'OFF'}"
                )
            if dn_rel < float(picard_rtol) and gate_now == prev_gate:
                picard_converged = True
                break
            prev_gate = gate_now

        self._record_picard_sc("sc scipy", picard_converged, n_picard,
                               picard_history, picard_max_it, picard_relax,
                               picard_rtol)

        # De-normalise and store.
        self.hat_x_sol = sol.x.copy()
        self.hat_ne_sol = sol.y[0].copy()
        self.x_sol = L * sol.x
        self.ne_sol = n0 * sol.y[0]
        self.dne_dx_sol = (n0 / L) * sol.y[1]
        self.E_sol = np.interp(self.x_sol, x_grid, E_frozen)
        self.sol = sol

        if v:
            print(f"[sc scipy] solved.  n_e in "
                  f"[{self.ne_sol.min():.3e}, {self.ne_sol.max():.3e}] m^-3")
        return self.x_sol, self.ne_sol, self.dne_dx_sol

    # ------------------------------------------------------------------
    # Firedrake implementation
    # ------------------------------------------------------------------

    def _ensure_firedrake_mesh_sc(self, mesh_n, fe_degree, force=False):
        """Build (or reuse) the non-dim mesh on xi in [-1, 0].

        Only the mesh, function space and DOF coordinates are cached:
        the coefficient Functions depend on the free parameters and the
        reference scales, so they are rebuilt on every solve by
        :meth:`_build_sc_coefficients`.

        Returns
        -------
        mesh, V, xi_dofs
        """
        if not _FIREDRAKE_AVAILABLE:
            raise ImportError(
                "Firedrake is not available in this environment.  "
                "Install Firedrake (https://www.firedrakeproject.org/) "
                "to use this solver.  Original import error:\n"
                f"  {_FIREDRAKE_IMPORT_ERR}"
            )

        mesh_key = ("sc", int(mesh_n), int(fe_degree))
        if (not force
                and self._fd_cache.get("mesh_key_sc") == mesh_key
                and "mesh_sc" in self._fd_cache):
            return (
                self._fd_cache["mesh_sc"],
                self._fd_cache["V_sc"],
                self._fd_cache["xi_dofs_sc"],
            )

        mesh = IntervalMesh(int(mesh_n), -1.0, 0.0)
        V = FunctionSpace(mesh, "CG", int(fe_degree))
        xi_dofs = Function(V).interpolate(
            SpatialCoordinate(mesh)[0]
        ).dat.data.copy()

        self._fd_cache.pop("N_sc", None)
        self._fd_cache.update({
            "mesh_key_sc": mesh_key,
            "mesh_sc": mesh,
            "V_sc": V,
            "xi_dofs_sc": xi_dofs,
        })
        return mesh, V, xi_dofs

    def _build_sc_coefficients(self, V, x_dofs_si):
        """Frozen equilibrium coefficient Functions in non-dim units.

        All entries are ``Function``s on ``V`` in DOF order:

            g          <|grad r|^2>                     (dimensionless)
            hat_Si     S_i / S0,   hat_Scx  S_CX / S0
            hat_Vcx    |V_CX| / [V]_0
            fFC, fCX   form factors                     (dimensionless)
            hat_D_NEO  D_NEO / [D]_0
            hat_C_ETG  C_ETG / (n0 [D]_0)   (so hat_C_ETG / N is the
                                             non-dim ETG diffusivity)
            C_cx       report Eq. (7) CX-flux coefficient
        """
        def _mk(arr_on_xinit, name, scale=1.0):
            f = Function(V, name=name)
            f.dat.data[:] = np.interp(
                x_dofs_si, self.x_init, arr_on_xinit
            ) * scale
            return f

        # make non-dim coefficients
        coeffs = {
            "g":         _mk(self.gradr2_fsa, "gradr2_fsa"),
            "hat_Si":    _mk(self.S_i_pres, "hat_S_i", 1.0 / self._S0_sc),
            "hat_Scx":   _mk(self.S_cx_pres, "hat_S_cx", 1.0 / self._S0_sc),
            "hat_Vcx":   _mk(np.abs(self.V_cx_pres), "hat_V_cx",
                             1.0 / self._V0_sc),
            "fFC":       _mk(self.fFC, "fFC"),
            "fCX":       _mk(self.fCX, "fCX"),
            "hat_D_NEO": _mk(self.D_NEO, "hat_D_NEO", 1.0 / self._D0_sc),
            "hat_C_ETG": _mk(self.C_ETG, "hat_C_ETG",
                             1.0 / (self._n0_sc * self._D0_sc)),
        }
        C_cx_fd = Function(V, name="C_cx")
        C_cx_fd.dat.data[:] = self._C_cx_sc(x_dofs_si)
        coeffs["C_cx"] = C_cx_fd # already is non-dimensional

        self._fd_cache["coeffs_sc"] = coeffs
        return coeffs

    def solve_sc_firedrake(self,
                           x_res=200,
                           fe_degree=2,
                           free_params=None,
                           bc_origin="p-file",
                           dne_dx_inner=None,
                           initial_guess="pfile",
                           tanh_width=None,
                           tanh_center=None,
                           eq6_form="complete",
                           first_step="auto",
                           picard_max_it=50,
                           picard_rtol=1e-8,
                           picard_relax=1.0,
                           linear_solver="lu",
                           ksp_rtol=1e-8,
                           ksp_max_it=200,
                           reuse_setup=True,
                           verbose=None):
        """Implicit Firedrake solver for the original model.

        Solves the same non-dimensional two-step problem as
        :meth:`solve_sc_scipy` but in conservative (weak) form with CG
        finite elements and SNES Newton for each frozen-coefficient
        nonlinear solve:

            Step 1 (report Eq. 6, no CX; ``eq6_form="paper"``):
                F6 = int hat_f N' w' dxi
                     + int N (hat_f/g)(hat_Si + hat_Scx)(N' - N'_in)
                           / (hat_V_FC f_FC) w dxi
                     + hat_f N'_in w ds(1) = 0
            (``eq6_form="complete"`` replaces hat_f/g by
             hat_f / (1 + hat_Scx/(2 hat_Si)); see the module docstring)

            Step 2 (report Eq. 7, Picard on E(x) and D_KBM):
                F7 = int hat_f N' w' dxi
                     + int N hat_Si [ hat_f (N' - N'_in)/(hat_V_CX f_CX)
                                      - C_CX hat_nFC0 E ] w dxi
                     + hat_f N'_in w ds(1) = 0

        with hat_f = g (hat_D_NEO + hat_D_KBM) + g hat_C_ETG / N (the ETG
        piece keeps its 1/N dependence inline, so Newton differentiates
        it exactly), Dirichlet N(0) = 1 (boundary id 2), and the Neumann
        inner flux imposed weakly through ds(1) (boundary id 1).

        The frozen KBM diffusivity ``hat_D_KBM`` and kernel ``E`` are
        Functions refreshed between Picard iterations from the latest
        density -- the same "average"-gate Picard pedestal-pressure
        treatment as ``solver_nondim`` (Saarelma Eqs. 24-25 diffusivity
        frozen into the D-slot, optional under-relaxation).

        Parameters are as in :meth:`solve_sc_scipy`, plus the finite
        element / PETSc controls of the parent coupled solver
        (``fe_degree``, ``linear_solver``, ``ksp_rtol``, ``ksp_max_it``).

        Returns
        -------
        x_sol, ne_sol, dne_dx_sol : ndarray
            SI profiles on the sorted DOF grid.
        """
        if not _FIREDRAKE_AVAILABLE:
            raise ImportError(
                "Firedrake is not available in this environment.  "
                "Install Firedrake to use this solver.  Original import "
                f"error:\n  {_FIREDRAKE_IMPORT_ERR}"
            )

        v = self.verbose if verbose is None else bool(verbose)
        force_setup = not reuse_setup
        picard_relax = float(picard_relax)
        if not (0.0 < picard_relax <= 1.0):
            raise ValueError(
                f"picard_relax must be in (0, 1], got {picard_relax}."
            )
        eq6_form = self._check_eq6_form_sc(eq6_form)
        self.eq6_form = eq6_form
        first_step = self._check_first_step_sc(first_step)
        self.first_step_used = first_step

        self._ensure_sc_setup(x_res, free_params=free_params,
                              force=force_setup)
        L = self._L_sc
        n0 = self._n0_sc
        dne_dx_inner_val = self._sc_neumann_inner_value(bc_origin, dne_dx_inner)
        dN_in_val = (L / n0) * dne_dx_inner_val # non-dim
        hat_nFC0 = self.nFC_x0 / n0 # non-dim

        _mesh, V, xi_dofs = self._ensure_firedrake_mesh_sc(
            x_res, fe_degree, force=force_setup,
        )
        x_dofs_si = L * xi_dofs # dimensionalized
        sort_idx = np.argsort(xi_dofs)
        unsort_idx = np.argsort(sort_idx)
        x_sorted = x_dofs_si[sort_idx] # dimensionalized 
        coeffs = self._build_sc_coefficients(V, x_dofs_si)

        # Initial guess (SI on the sorted grid, then back to DOF order).
        ne_inner_guess = self.ne_x0 + dne_dx_inner_val * self.x_inner
        ne_init_sorted = self._sc_initial_guess(
            x_sorted, initial_guess, tanh_width, tanh_center, ne_inner_guess,
        )
        ne_init = ne_init_sorted[unsort_idx]
        self.ne_init = ne_init

        N = Function(V, name="hat_n_e")
        N.dat.data[:] = ne_init / n0
        self._fd_cache["N_sc"] = N
        w = TestFunction(V)

        if v:
            print(f"[sc firedrake] x_inner         = {self.x_inner:.4e} m")
            print(f"[sc firedrake] dne/dx(x_inner) = {dne_dx_inner_val:.3e}"
                  f" m^-4  ({bc_origin})")
            print(f"[sc firedrake] N'(-1)          = {dN_in_val:.3e}")
            print(f"[sc firedrake] hat_nFC(0)      = {hat_nFC0:.3e}")
            print(f"[sc firedrake] scales: L = {L:.4e} m, n0 = {n0:.4e} m^-3,"
                  f" [D]_0 = {self._D0_sc:.4e} m^2/s")

        # KBM diffusivity frozen from the initial guess (average gate).
        D_KBM_xinit = self.calc_D_KBM_average_sc(ne_init_sorted, x_sorted)
        hat_D_KBM_fd = Function(V, name="hat_D_KBM_picard")
        hat_D_KBM_fd.dat.data[:] = np.interp(
            x_dofs_si, self.x_init, D_KBM_xinit
        ) / self._D0_sc
        self._fd_cache["hat_D_KBM_picard_sc"] = hat_D_KBM_fd

        # Frozen FC attenuation kernel E(xi) (unused by step 1, seeded to
        # the initial guess so the form is well defined from the start).
        E_fd = Function(V, name="E_kernel")
        E_fd.dat.data[:] = self._exp_kernel_sc(
            ne_init_sorted, x_sorted
        )[unsort_idx]
        self._fd_cache["E_kernel_sc"] = E_fd

        # Constants (non-dimensional)
        dN_in_c = Constant(dN_in_val)
        hat_nFC0_c = Constant(hat_nFC0)
        hat_Vfc_c = Constant(abs(self.V_FC) / self._V0_sc)

        g_fd = coeffs["g"]
        hat_Si_fd = coeffs["hat_Si"]
        hat_Scx_fd = coeffs["hat_Scx"]
        hat_Vcx_fd = coeffs["hat_Vcx"]
        fFC_fd = coeffs["fFC"]
        fCX_fd = coeffs["fCX"]
        hat_D_NEO_fd = coeffs["hat_D_NEO"]
        hat_C_ETG_fd = coeffs["hat_C_ETG"]
        C_cx_fd = coeffs["C_cx"]

        # hat_f = g (D_NEO + D_KBM)/[D]_0 + g C_ETG/(n0 [D]_0 N):
        # the inline 1/N keeps the ETG nonlinearity visible to Newton.
        hat_f = g_fd * (hat_D_NEO_fd + hat_D_KBM_fd) + g_fd * hat_C_ETG_fd / N
        N_dx = N.dx(0)

        # Weak residuals.  Boundary id 1 = xi = -1 (inner, Neumann flux
        # imposed with the prescribed slope), id 2 = xi = 0 (Dirichlet).
        bnd_term = hat_f * dN_in_c * w * ds(1)

        # Step 1 (Eq. 6): RHS6 = N D6 (hat_Si + hat_Scx)(N' - N'_in)
        #                        / (hat_V_FC f_FC), where the "D6" slot is
        # hat_f/g (the D_ped of Eq. 6 as printed) or
        # hat_f / (1 + hat_Scx/(2 hat_Si)) (Eq. 16-full).
        if eq6_form == "paper":
            D6 = hat_f / g_fd
        else:
            D6 = hat_f / (Constant(1.0)
                          + Constant(0.5) * hat_Scx_fd / hat_Si_fd)
        rhs6 = (
            N * D6 * (hat_Si_fd + hat_Scx_fd)
            * (N_dx - dN_in_c) / (hat_Vfc_c * fFC_fd)
        )
        F6 = hat_f * N_dx * w.dx(0) * dx + rhs6 * w * dx + bnd_term

        # Step 2 (Eq. 7): RHS7 = N hat_Si [ hat_f (N' - N'_in)
        #                        / (hat_V_CX f_CX) - C_CX hat_nFC0 E ]
        rhs7 = N * hat_Si_fd * (
            hat_f * (N_dx - dN_in_c) / (hat_Vcx_fd * fCX_fd)
            - C_cx_fd * hat_nFC0_c * E_fd
        )
        F7 = hat_f * N_dx * w.dx(0) * dx + rhs7 * w * dx + bnd_term

        bcs = [DirichletBC(V, Constant(1.0), 2)]   # N(0) = 1
        snes_params = self._build_petsc_solver_parameters(
            linear_solver=linear_solver,
            ksp_rtol=ksp_rtol,
            ksp_max_it=ksp_max_it,
        )

        # --------------------------------------------------------------
        # Step 1: solve the no-CX equation.
        # --------------------------------------------------------------
        if first_step != "skip":
            err = None
            try:
                solve(F6 == 0, N, bcs=bcs, solver_parameters=snes_params)
            except Exception as exc:
                err = exc
            else:
                if N.dat.data.min() <= 0.0:
                    # See the note in solve_sc_scipy: a converged but
                    # collapsed profile is not a usable starting point.
                    err = RuntimeError(
                        f"the solution is not positive (min n_e = "
                        f"{n0 * N.dat.data.min():.3e} m^-3)"
                    )
            if err is not None:
                # Restore the initial guess: a diverged SNES leaves N in
                # whatever state the failed line search reached.
                N.dat.data[:] = ne_init / n0
                self._first_step_failed_sc("sc firedrake", first_step, err)
            else:
                self.hat_ne_first = N.dat.data[sort_idx].copy()
                if v:
                    print(f"[sc firedrake] first step (Eq. 6) solved, "
                          f"alpha_bar(guess) = {self.alpha_bar_ped:.4f}, "
                          f"KBM {'ON' if self.kbm_gate_on else 'OFF'}")
        elif v:
            print("[sc firedrake] first step skipped; the Eq. 7 Picard loop "
                  f"starts from initial_guess={initial_guess!r}")

        # --------------------------------------------------------------
        # Step 2: Picard loop on the full equation.
        #
        # Per-iteration-frozen coefficients, updated between Picard
        # iterations from the latest hat_n_e (cf. solver_nondim
        # picard_gate_mode="average"):
        #   - freeze hat_D_KBM = (alpha_bar - alpha_crit) * hat_G into
        #     the diffusivity (Saarelma Eq. 25);
        #   - freeze the FC kernel E(xi) built from the latest n_e.
        # --------------------------------------------------------------
        picard_history = []
        picard_converged = False
        n_picard = 0
        prev_gate = bool(self.kbm_gate_on)
        prev_hat_ne = N.dat.data.copy()
        E_initialised = False

        for it in range(1, int(picard_max_it) + 1):
            n_picard = it

            # Refreeze E and D_KBM from the current iterate.
            ne_curr_sorted = n0 * N.dat.data[sort_idx]
            E_new = self._exp_kernel_sc(ne_curr_sorted, x_sorted)[unsort_idx]
            if not E_initialised: # first step in loop
                E_fd.dat.data[:] = E_new
                E_initialised = True
            else: # all other steps in the loop
                E_fd.dat.data[:] = (
                    picard_relax * E_new
                    + (1.0 - picard_relax) * E_fd.dat.data
                )

            D_KBM_xinit = self.calc_D_KBM_average_sc(ne_curr_sorted, x_sorted)
            gate_now = bool(self.kbm_gate_on)
            hat_D_KBM_new = np.interp(
                x_dofs_si, self.x_init, D_KBM_xinit
            ) / self._D0_sc
            hat_D_KBM_fd.dat.data[:] = (
                picard_relax * hat_D_KBM_new
                + (1.0 - picard_relax) * hat_D_KBM_fd.dat.data
            )

            # Solve the full equation with frozen E / D_KBM (warm start
            # from the previous iterate stored in N).
            solve(F7 == 0, N, bcs=bcs, solver_parameters=snes_params)

            hat_ne_new = N.dat.data.copy()
            dn_rel = (
                np.linalg.norm(hat_ne_new - prev_hat_ne)
                / max(np.linalg.norm(prev_hat_ne), 1e-300)
            )

            picard_history.append({
                "iteration": it,
                "alpha_bar": float(self.alpha_bar_ped),
                "alpha_frac_above": float(self.alpha_frac_above),
                "kbm_gate_on": gate_now,
                "dne_rel": float(dn_rel),
            })
            if v:
                print(
                    f"[sc firedrake] picard it {it:3d}: "
                    f"|dne|_rel = {dn_rel:.3e}, "
                    f"alpha_bar = {self.alpha_bar_ped:.4f}, "
                    f"KBM {'ON' if gate_now else 'OFF'}"
                )
            if abs(dn_rel) < float(picard_rtol) and gate_now == prev_gate:
                picard_converged = True
                break
            prev_hat_ne = hat_ne_new
            prev_gate = gate_now

        self._record_picard_sc("sc firedrake", picard_converged, n_picard,
                               picard_history, picard_max_it, picard_relax,
                               picard_rtol)

        # --------------------------------------------------------------
        # Extract converged profiles (non-dim and SI).
        # --------------------------------------------------------------
        self.hat_x_sol = xi_dofs[sort_idx]
        self.hat_ne_sol = N.dat.data[sort_idx]
        self.x_sol = L * self.hat_x_sol
        self.ne_sol = n0 * self.hat_ne_sol
        self.dne_dx_sol = np.gradient(self.ne_sol, self.x_sol)
        self.E_sol = E_fd.dat.data[sort_idx].copy()
        self.N_fd = N
        self.V_fd = V

        if v:
            print(f"[sc firedrake] solved.  n_e in "
                  f"[{self.ne_sol.min():.3e}, {self.ne_sol.max():.3e}] m^-3")
        return self.x_sol, self.ne_sol, self.dne_dx_sol

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def solve_sc(self, implementation="firedrake", **kwargs):
        """Solve the original model with the chosen implementation.

        Parameters
        ----------
        implementation : {"firedrake", "scipy"}
            Which of the two solvers to call.
        **kwargs
            Passed straight through to :meth:`solve_sc_firedrake` or
            :meth:`solve_sc_scipy`.

        Returns
        -------
        x_sol, ne_sol, dne_dx_sol : ndarray
        """
        implementation = str(implementation).lower()
        if implementation == "firedrake":
            return self.solve_sc_firedrake(**kwargs)
        if implementation == "scipy":
            return self.solve_sc_scipy(**kwargs)
        raise ValueError(
            f"implementation must be 'firedrake' or 'scipy', got "
            f"{implementation!r}."
        )
