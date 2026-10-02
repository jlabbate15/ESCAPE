"""Master electron-density boundary-condition and initial-guess helpers.

Every Saarelma--Connor solver in this repository -- the original
one-equation state (``solver_sc.saarelma_connor_sc.solve_sc_scipy`` and
``.solve_sc_firedrake``) and the coupled three-equation state
(``solver_nondim.saarelma_connor_nondim.solve_coupled_nondim`` and
``.solve_coupled_nondim_scipy``) -- resolves its n_e boundary conditions
and builds its n_e initial guess *here*, so the four cannot drift apart.

What a solver receives
======================
Three things, all in SI units:

``NeBCs``
    the boundary-condition values and where they sit,

``ne_init``
    the initial n_e profile on whatever grid the caller asked for (each
    solver is free to re-interpolate it onto its own mesh), and

``(nFC_init, nCX_init)``
    the matching neutral initial guesses -- coupled three-equation
    state only (:func:`build_neutral_initial_guess`).

Nothing else about the boundary is assumed or looked up downstream.

The two pathways
================
There are exactly **two** n_e boundary-condition pathways, selected by
``ne_bc_loc``.  Anything else raises:

``"inner"``
    Neumann  dn_e/dx at x = x_inner   (the pedestal top)
    Dirichlet n_e     at x = 0        (the separatrix)

``"outer"``
    Neumann  dn_e/dx at x = 0         (the separatrix)
    Dirichlet n_e     at x = 0        (the separatrix)

    Both conditions sit at the separatrix and the inner boundary is left
    free -- the boundary-condition set of Saarelma et al. 2023 Sec. 2.3,
    where the separatrix density and its gradient (their Eq. (20)) are
    the specified data and nothing is imposed at the pedestal top.

The Dirichlet value is ``state.ne_x0`` in both pathways; only the
*location of the Neumann condition* differs.

Where the values come from
==========================
The Neumann value is selected by ``dne_method``:

``"state"`` (default)
    read off the kinetic profile the state holds, ``state.n_e`` (on
    ``state.psi_ne_eval``), interpolated onto the solver's ``x_init``
    grid at the moment the boundary conditions are resolved;

``"user"``
    ``dne_dx_bc`` (m^-4), used as given at the location ``ne_bc_loc``
    selects;

``"Saarelma2023"``  (``ne_bc_loc="outer"`` only)
    Saarelma et al. 2023 Eq. (20), the separatrix gradient of an
    exponential SOL with decay length sqrt(D_SOL tau_par)::

        dn_e/dx|_0 = -n_e(0) / sqrt(D_SOL tau_par)

    with ``tau_par`` (s) supplied by the caller and D_SOL = D_ped(0) =
    D_NEO(0) + D_KBM(0) + C_ETG(0)/n_e(0).  D_KBM needs the n_e initial
    guess, which needs these boundary conditions, so
    :func:`resolve_ne_bcs` leaves this value pending and the solver fills
    it in with :func:`resolve_saarelma2023_neumann` once its
    initial-guess D_KBM exists.  It is computed once, before the KBM
    Picard loop, and stays frozen for the whole solve.

``ne_inner`` (the pedestal-top density for the initial guess) and
``dne_dx_neginf`` (the integration constant below) are used as given
when supplied and otherwise fall back to ``state.n_e``.  The Dirichlet
value is always ``state.ne_x0``.

Only what was asked for is looked up
====================================
:func:`resolve_ne_bcs` reads the gradient at the Neumann location the
caller selected and **nowhere else**.  In ``"inner"`` mode the separatrix
gradient is never evaluated; in ``"outer"`` mode the pedestal-top
gradient is never evaluated.  ``NeBCs.dne_dx_inner`` /
``NeBCs.dne_dx_outer`` return None for whichever end carries no
condition, so a solver that reaches for the wrong one fails loudly
instead of silently using a number nobody asked for.

Not a boundary condition
========================
The original state's Eq. (6)/(7) source term carries Saarelma's constant
of integration ``C = dn_e/dx|_{x=-inf}``, which appears in the
combination ``(dn_e/dx - dn_e/dx|_in)``.  That is a *state constant*
expressing "the edge neutrals are extinguished by the pedestal top", not
a boundary condition, so it is resolved separately by
:func:`resolve_integration_constant` and is unaffected by
``ne_bc_loc``.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import OptimizeResult

__all__ = [
    "NE_BC_LOCS",
    "NE_DNE_METHODS",
    "NeBCs",
    "check_ne_bc_loc",
    "check_dne_method",
    "resolve_ne_bcs",
    "resolve_saarelma2023_neumann",
    "build_ne_initial_guess",
    "resolve_integration_constant",
    "integrate_from_separatrix",
]

#: The only two supported n_e boundary-condition pathways.
NE_BC_LOCS = ("inner", "outer")

#: The supported ways of setting the Neumann value (``dne_method``).
NE_DNE_METHODS = ("user", "state", "Saarelma2023")

#: Labels recorded in ``NeBCs.origin`` / ``NeBCs.ne_inner_origin``.
_PROFILE_ORIGIN = "state.n_e profile"
_USER_ORIGIN = "user-specified"
_SAARELMA_ORIGIN = "Saarelma 2023 Eq. (20)"


@dataclass
class NeBCs:
    """Resolved n_e boundary conditions, in SI units.

    Attributes
    ----------
    loc : {"inner", "outer"}
        Which pathway this is, i.e. where the Neumann condition sits.
    x_inner : float
        Inner-boundary position (m, < 0; the separatrix is x = 0).
    ne_outer : float
        Dirichlet value n_e(0) (m^-3).  Always at the separatrix.
    dne_dx : float or None
        Neumann value dn_e/dx (m^-4) at :attr:`x_neumann`.  None while a
        ``"Saarelma2023"`` value is still pending (see :attr:`pending`).
    ne_inner_guess : float
        Pedestal-top density (m^-3) used *only* to shape the initial
        guess.  This is **not** a boundary condition: in the ``"inner"``
        pathway nothing pins n_e(x_inner), and in the ``"outer"``
        pathway the inner boundary is free entirely.
    origin : str
        Where ``dne_dx`` came from, for logging.
    ne_inner_origin : str
        Where ``ne_inner_guess`` came from, for logging.
    method : {"user", "state", "Saarelma2023"}
        How ``dne_dx`` is set; see the module docstring.
    tau_par : float or None
        Parallel loss time (s) of the ``"Saarelma2023"`` method.
    D_sol : float or None
        D_SOL = D_ped(0) (m^2/s) used by the ``"Saarelma2023"`` method,
        set by :func:`resolve_saarelma2023_neumann`.
    """

    loc: str
    x_inner: float
    ne_outer: float
    dne_dx: float
    ne_inner_guess: float
    origin: str = ""
    ne_inner_origin: str = ""
    method: str = "state"
    tau_par: float = None
    D_sol: float = None

    @property
    def pending(self):
        """True until a ``"Saarelma2023"`` Neumann value has been filled in."""
        return self.dne_dx is None

    @property
    def x_neumann(self):
        """Position (m) at which ``dne_dx`` is imposed."""
        return self.x_inner if self.loc == "inner" else 0.0

    @property
    def dne_dx_inner(self):
        """Neumann value at the pedestal top, or None in ``"outer"`` mode."""
        return self.dne_dx if self.loc == "inner" else None

    @property
    def dne_dx_outer(self):
        """Neumann value at the separatrix, or None in ``"inner"`` mode."""
        return self.dne_dx if self.loc == "outer" else None

    def describe(self, prefix=""):
        """Multi-line summary for a solver's verbose block."""
        where = "x_inner" if self.loc == "inner" else "x = 0"
        dne_dx = ("pending" if self.pending
                  else f"{self.dne_dx:.3e} m^-4")
        out = (
            f"{prefix}ne_bc_loc        = {self.loc!r}\n"
            f"{prefix}x_inner          = {self.x_inner:.4e} m\n"
            f"{prefix}n_e(0)           = {self.ne_outer:.3e} m^-3  (Dirichlet)\n"
            f"{prefix}dne/dx({where:>7}) = {dne_dx}  "
            f"(Neumann, {self.origin})\n"
        )
        if self.method == "Saarelma2023":
            D_sol = ("pending" if self.D_sol is None
                     else f"{self.D_sol:.3e} m^2/s")
            out += (f"{prefix}tau_par          = {self.tau_par:.3e} s\n"
                    f"{prefix}D_SOL = D_ped(0) = {D_sol}  "
                    f"(frozen from the initial guess)\n")
        return out + (
            f"{prefix}ne(x_inner)      = {self.ne_inner_guess:.3e} m^-3  "
            f"(initial guess only, {self.ne_inner_origin})"
        )


def check_ne_bc_loc(ne_bc_loc):
    """Validate and normalise the boundary-condition pathway flag.

    Raises
    ------
    ValueError
        For anything other than ``"inner"`` or ``"outer"`` -- including
        the legacy ``ne_inner_bc="dirichlet"`` pathway (Dirichlet at both
        ends), which is no longer supported.
    """
    loc = str(ne_bc_loc).lower()
    if loc not in NE_BC_LOCS:
        raise ValueError(
            f"ne_bc_loc must be one of {NE_BC_LOCS}, got {ne_bc_loc!r}.  "
            "The supported pathways are 'inner' (Neumann dn_e/dx at "
            "x_inner, Dirichlet n_e at the separatrix) and 'outer' "
            "(Neumann dn_e/dx and Dirichlet n_e both at the separatrix); "
            "no other combination is available."
        )
    return loc


def check_dne_method(dne_method, ne_bc_loc, dne_dx_bc=None, tau_par=None):
    """Validate and normalise the Neumann-value flag ``dne_method``.

    Checks that the method is compatible with the pathway and that exactly
    the inputs it uses were supplied, so an argument that would otherwise
    be silently ignored raises instead.

    Raises
    ------
    ValueError
        For an unknown method; ``"Saarelma2023"`` with
        ``ne_bc_loc="inner"``; ``"user"`` without ``dne_dx_bc``;
        ``"Saarelma2023"`` without a positive ``tau_par``; or
        ``dne_dx_bc`` / ``tau_par`` given to a method that does not use it.
    """
    lookup = {m.lower(): m for m in NE_DNE_METHODS}
    method = lookup.get(str(dne_method).lower())
    if method is None:
        raise ValueError(
            f"dne_method must be one of {NE_DNE_METHODS}, got "
            f"{dne_method!r}."
        )
    loc = check_ne_bc_loc(ne_bc_loc)

    if method == "Saarelma2023" and loc != "outer":
        raise ValueError(
            "dne_method='Saarelma2023' sets the separatrix gradient, so it "
            "requires ne_bc_loc='outer' (got 'inner')."
        )
    if method == "user" and dne_dx_bc is None:
        raise ValueError("dne_method='user' requires dne_dx_bc.")
    if method != "user" and dne_dx_bc is not None:
        raise ValueError(
            f"dne_dx_bc is only used by dne_method='user' (got "
            f"dne_method={method!r}); drop it or set dne_method='user'."
        )
    if method == "Saarelma2023":
        if tau_par is None or not float(tau_par) > 0.0:
            raise ValueError(
                "dne_method='Saarelma2023' requires a positive tau_par (s), "
                f"got {tau_par!r}."
            )
    elif tau_par is not None:
        raise ValueError(
            f"tau_par is only used by dne_method='Saarelma2023' (got "
            f"dne_method={method!r})."
        )
    return method


def _check_negative_slope(loc, dne_dx_val):
    """Raise unless the Neumann slope is strictly negative."""
    if not dne_dx_val < 0.0:
        where = "x_inner" if loc == "inner" else "0"
        raise ValueError(
            f"dne/dx({where}) = {dne_dx_val:.3e} m^-4 must be strictly "
            "negative (density decreasing outward) for the Neumann "
            "boundary condition."
        )


def _ne_on_x_init(state):
    """``state.n_e`` (m^-3, on ``state.psi_ne_eval``) interpolated onto the
    solver grid ``state.x_init`` (via ``state.psi_N_pres``).

    Re-interpolated on every call rather than reusing ``state.n_e_pres``,
    so the boundary conditions always follow the current ``state.n_e``.
    """
    return interp1d(state.psi_ne_eval, state.n_e, kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate")(state.psi_N_pres)


def _profile_gradient_at(state, x_at):
    """dn_e/dx (m^-4) read off ``state.n_e`` at SI ``x_at``."""
    dne_dx = np.gradient(_ne_on_x_init(state), state.x_init)
    return float(np.interp(x_at, state.x_init, dne_dx))


def _profile_ne_inner(state, x_inner):
    """n_e(x_inner) (m^-3) from ``state.n_e``, with the parent class's
    manual-ne_x0 offset applied."""
    ne_prof = _ne_on_x_init(state)
    ne_inner_val = float(np.interp(x_inner, state.x_init, ne_prof))
    if getattr(state, "ne_x0_manual", False):
        # Shift the whole profile so it meets the manually set separatrix
        # density (same convention as the original inline block).
        ne_inner_val += (state.ne_x0
                         - float(np.interp(0.0, state.x_init, ne_prof)))
    return ne_inner_val


def resolve_ne_bcs(state, ne_bc_loc, dne_method="state", dne_dx_bc=None,
                   ne_inner=None, tau_par=None,
                   require_negative_slope=False):
    """Resolve the n_e boundary conditions for one solve.

    The Neumann value is set as ``dne_method`` says; ``ne_inner`` is used
    when given and otherwise read off ``state.n_e``.  Only the two
    conditions belonging to ``ne_bc_loc`` are looked up; the gradient at
    the other end of the domain is never evaluated.

    With ``dne_method="Saarelma2023"`` the returned ``NeBCs`` is
    :attr:`~NeBCs.pending`: the solver must call
    :func:`resolve_saarelma2023_neumann` once its initial-guess D_KBM is
    available, before it uses ``dne_dx``.

    Parameters
    ----------
    state : saarelma_connor
        Solver instance.  ``state.n_e``, ``state.psi_ne_eval``,
        ``state.psi_N_pres``, ``state.x_init``, ``state.ne_x0`` and
        ``state.x_inner`` must already be set (i.e. call this after the
        equilibrium/profile setup and after ``find_inner_boundary``).
    ne_bc_loc : {"inner", "outer"}
        Boundary-condition pathway; see the module docstring.
    dne_method : {"state", "user", "Saarelma2023"}
        How the Neumann value is set; see the module docstring.
        ``"Saarelma2023"`` is valid only with ``ne_bc_loc="outer"``.
    dne_dx_bc : float, optional
        Neumann value dn_e/dx (m^-4) for ``dne_method="user"``, imposed at
        x_inner for ``"inner"`` and at the separatrix for ``"outer"``.
    ne_inner : float, optional
        Pedestal-top density (m^-3) for the initial guess only.  None reads
        it off ``state.n_e`` at x_inner.
    tau_par : float, optional
        Parallel loss time (s) for ``dne_method="Saarelma2023"``.
    require_negative_slope : bool
        Raise if the resolved slope is not strictly negative.  The
        original-state solvers set this; the coupled solvers do not.

    Returns
    -------
    NeBCs
    """
    loc = check_ne_bc_loc(ne_bc_loc)
    method = check_dne_method(dne_method, loc, dne_dx_bc=dne_dx_bc,
                              tau_par=tau_par)

    x_inner = float(state.x_inner)
    if x_inner >= 0.0:
        raise ValueError(
            f"x_inner = {x_inner} must be strictly less than 0 (the "
            "separatrix)."
        )
    ne_outer = float(state.ne_x0)          # Dirichlet, always at x = 0

    # ---- the Neumann value, read only where the pathway puts it -------
    if method == "user":
        dne_dx_val = float(dne_dx_bc)
        origin = _USER_ORIGIN
    elif method == "state":
        x_neumann = x_inner if loc == "inner" else 0.0
        dne_dx_val = _profile_gradient_at(state, x_neumann)
        origin = _PROFILE_ORIGIN
    else:  # "Saarelma2023"
        # Needs D_SOL = D_ped(0), hence D_KBM, hence the n_e initial guess
        # built from these very BCs -- so it is left pending here and set
        # by resolve_saarelma2023_neumann once the solver has D_KBM.
        dne_dx_val = None
        origin = _SAARELMA_ORIGIN

    if require_negative_slope and dne_dx_val is not None:
        _check_negative_slope(loc, dne_dx_val)

    # ---- pedestal-top density for the initial guess only --------------
    if ne_inner is not None:
        ne_inner_guess = float(ne_inner)
        ne_inner_origin = _USER_ORIGIN
    else:
        ne_inner_guess = _profile_ne_inner(state, x_inner)
        ne_inner_origin = _PROFILE_ORIGIN

    # Kept for backwards compatibility with callers/notebooks that read
    # these attributes off the state after a solve.
    state.ne_inner = ne_inner_guess
    if loc == "inner":
        state.dne_dx_inner = dne_dx_val
        state.dne_dx_neginf = dne_dx_val
    else:
        state.dne_dx_outer = dne_dx_val

    return NeBCs(loc=loc, x_inner=x_inner, ne_outer=ne_outer,
                 dne_dx=dne_dx_val, ne_inner_guess=ne_inner_guess,
                 origin=origin, ne_inner_origin=ne_inner_origin,
                 method=method,
                 tau_par=None if tau_par is None else float(tau_par))


def resolve_saarelma2023_neumann(state, bcs, x_D_KBM, D_KBM,
                                 require_negative_slope=False):
    """Fill in a pending ``dne_method="Saarelma2023"`` Neumann value.

    Saarelma et al. 2023 Eq. (20), the separatrix gradient of an
    exponential SOL with decay length sqrt(D_SOL tau_par)::

        dn_e/dx|_0 = -n_e(0) / sqrt(D_SOL tau_par),
        D_SOL = D_ped(0) = D_NEO(0) + D_KBM(0) + C_ETG(0) / n_e(0).

    Solvers call this once, with the D_KBM built from the n_e initial
    guess and before the KBM Picard loop, so the condition is frozen for
    the whole solve.  A no-op for the other methods.

    Parameters
    ----------
    state : saarelma_connor
        Supplies ``D_NEO`` and ``C_ETG`` on ``state.x_init``.
    bcs : NeBCs
        From :func:`resolve_ne_bcs`; updated in place and returned.
    x_D_KBM : array_like
        SI grid (m) on which ``D_KBM`` is given; need not be sorted.
    D_KBM : array_like
        KBM diffusivity (m^2/s) on ``x_D_KBM``.
    require_negative_slope : bool
        As in :func:`resolve_ne_bcs`.
    """
    if bcs.method != "Saarelma2023":
        return bcs

    x_D_KBM = np.asarray(x_D_KBM, dtype=float)
    D_KBM = np.asarray(D_KBM, dtype=float)
    order = np.argsort(x_D_KBM)
    D_KBM_0 = float(np.interp(0.0, x_D_KBM[order], D_KBM[order]))
    x_init = np.asarray(state.x_init, dtype=float)
    D_NEO = np.broadcast_to(np.asarray(state.D_NEO, dtype=float),
                            x_init.shape)
    D_NEO_0 = float(np.interp(0.0, x_init, D_NEO))
    C_ETG_0 = float(np.interp(0.0, x_init, state.C_ETG))

    D_sol = D_NEO_0 + D_KBM_0 + C_ETG_0 / bcs.ne_outer
    if not (np.isfinite(D_sol) and D_sol > 0.0):
        raise ValueError(
            f"D_SOL = D_ped(0) = {D_sol!r} m^2/s (D_NEO = {D_NEO_0:.3e}, "
            f"D_KBM = {D_KBM_0:.3e}, C_ETG/n_e = "
            f"{C_ETG_0 / bcs.ne_outer:.3e}) is not positive; cannot form "
            "the Saarelma 2023 separatrix gradient."
        )

    dne_dx_val = -bcs.ne_outer / np.sqrt(D_sol * bcs.tau_par)
    if require_negative_slope:
        _check_negative_slope(bcs.loc, dne_dx_val)

    bcs.dne_dx = float(dne_dx_val)
    bcs.D_sol = float(D_sol)
    state.dne_dx_outer = bcs.dne_dx
    state.D_sol = bcs.D_sol
    return bcs


def build_ne_initial_guess(state, x_grid, initial_guess, bcs,
                           tanh_width=None, tanh_center=None):
    """SI electron-density initial guess (m^-3) on ``x_grid`` (m).

    Shapes are anchored on the resolved boundary conditions: every option
    meets the outer Dirichlet value ``bcs.ne_outer`` at the separatrix
    and rises to ``bcs.ne_inner_guess`` at the pedestal top.

    Parameters
    ----------
    state : class
        Used only by the profile-reading options.
    x_grid : array_like
        SI radial grid (m), 0 at the separatrix.  Need not be sorted.
    initial_guess : {"pfile", "linear", "tanh"}
    bcs : NeBCs
        From :func:`resolve_ne_bcs`.
    tanh_width, tanh_center : float or None
        ``"tanh"`` shape parameters in SI metres.  Defaults: a width of
        10% of the domain and a centre one width inside the separatrix.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    x_left = float(bcs.x_inner)
    x_right = 0.0
    ne_inner_val = float(bcs.ne_inner_guess)
    ne_outer_val = float(bcs.ne_outer)

    if initial_guess == "linear":
        xi = (x_grid - x_left) / (x_right - x_left)
        return ne_inner_val + (ne_outer_val - ne_inner_val) * xi

    if initial_guess == "state":
        psi_to_x = interp1d(state.psi_N_pres, state.x_init, kind='linear',bounds_error=False, fill_value='extrapolate')
        return np.interp(x_grid, psi_to_x(state.psi_ne_eval), state.n_e)

    if initial_guess == "tanh":
        width = (float(tanh_width) if tanh_width is not None
                 else 0.1 * abs(x_left))
        if width <= 0:
            raise ValueError(f"tanh_width must be positive, got {width}.")
        center = float(tanh_center) if tanh_center is not None else -width
        s_ne = 0.5 * (1.0 - np.tanh((x_grid - center) / (0.5 * width)))
        return ne_outer_val + (ne_inner_val - ne_outer_val) * s_ne

    raise ValueError(
        f"Unknown initial_guess={initial_guess!r}."
    )



def build_neutral_initial_guess(state, x_grid, ne_init,
                                nFC_ic="solve", nCX_ic="solve"):
    """SI neutral initial guesses ``(nFC_init, nCX_init)`` in m^-3 on
    ``x_grid`` (m), for a given electron-density guess ``ne_init``.

    Shared by both discretisations of the coupled three-equation state
    (``solve_coupled_nondim`` and ``solve_coupled_nondim_scipy``) so the
    two cannot drift apart, exactly as
    :func:`build_ne_initial_guess` is shared for n_e.

    Parameters
    ----------
    state : saarelma_connor
        Supplies the frozen coefficient arrays on ``state.x_init`` and
        the separatrix values ``nFC_x0`` / ``nCX_x0``.
    x_grid : array_like
        SI radial grid (m), 0 at the separatrix.  Need not be sorted;
        the returned arrays follow the order of ``x_grid``.
    ne_init : array_like
        SI electron-density guess (m^-3) on ``x_grid``, e.g. from
        :func:`build_ne_initial_guess`.
    nFC_ic : {"solve", "state"}
        ``"solve"``
            Integrate the FC neutral equation (Eq. (14) of Saarelma et
            al. 2023) analytically at the frozen ``ne_init``.  With
            n_e fixed the equation is exactly homogeneous and linear in
            u = f_FC n_FC, so an integrating factor gives it in closed
            form.
        ``"state"``
            Interpolate the previous Saarelma-Connor solution
            ``state.nFC`` (given on ``state.psi_nFC_eval``, set by
            ``ESCAPE_solve``) onto ``x_grid``.
    nCX_ic : {"solve", "scale nFC", "state"}
        ``"solve"``
            Integrate the CX neutral equation (Eq. (10)) analytically at
            the frozen ``ne_init`` *and* the FC guess just built,
            keeping the FC source term.
        ``"scale nFC"``
            ``nCX_init = nFC_init * nCX_x0 / nFC_x0``.
        ``"state"``
            As above, from ``state.nCX`` on ``state.psi_nCX_eval``.

    Notes
    -----
    Every branch is evaluated on the grid sorted in *descending* x, i.e.
    integrating inward from the separatrix, where the Dirichlet data
    ``nFC_x0`` / ``nCX_x0`` live; the results are scattered back into
    the caller's ordering before being returned.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    ne_init = np.asarray(ne_init, dtype=float)

    # Everything is built on the descending-x ordering (separatrix first),
    # because that is the end that carries the neutral Dirichlet data.
    order_desc = np.argsort(x_grid)[::-1]
    x_desc = x_grid[order_desc]
    ne_desc = ne_init[order_desc]                                   # m^-3
    Si_desc = np.interp(x_desc, state.x_init, state.S_i_pres)
    Scx_desc = np.interp(x_desc, state.x_init, state.S_cx_pres)
    fFC_desc = np.interp(x_desc, state.x_init, state.fFC)
    fCX_desc = np.interp(x_desc, state.x_init, state.fCX)
    g_desc = np.interp(x_desc, state.x_init, state.gradr2_fsa)
    Vcx_desc = np.interp(x_desc, state.x_init, np.abs(state.V_cx_pres))  # m/s

    def _state_on_desc(name, psi_name):
        """Interpolate the psi_N-tabulated state profile ``state.<name>``
        (on ``state.<psi_name>``) onto x_desc."""
        vals = np.asarray(getattr(state, name, []), dtype=float)
        psi_n = np.asarray(getattr(state, psi_name, []), dtype=float)
        if vals.size == 0 or vals.size != psi_n.size:
            raise ValueError(
                f"state.{name} (size {vals.size}) / state.{psi_name} "
                f"(size {psi_n.size}) are empty or mismatched; the 'state' "
                "neutral initial guess needs a previous Saarelma-Connor "
                "solve (see ESCAPE_solve)."
            )
        psi_to_x = interp1d(state.psi_N_pres, state.x_init, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        x_n = psi_to_x(psi_n)
        order = np.argsort(x_n)   # np.interp needs ascending abscissae
        return np.interp(x_desc, x_n[order], vals[order])

    def _scatter(vals_desc):
        out = np.empty_like(x_grid)
        out[order_desc] = vals_desc
        return out

    # ---- n_FC --------------------------------------------------------
    if nFC_ic == "solve":
        # Eq. (14) of Saarelma et al. (2023) at frozen n_e:
        #   |V_FC| d/dx[f_FC n_FC] = n_e (S_i + S_CX) n_FC,
        # homogeneous in u = f_FC n_FC, so the integrating factor from
        # the separatrix gives n_FC in closed form.
        integrand_init = (
            ne_desc * (Si_desc + Scx_desc) / (fFC_desc * abs(state.V_FC))
        )
        cumint_desc = cumulative_trapezoid(integrand_init, x_desc, initial=0.0)
        nFC_on_desc = state.nFC_x0 * np.exp(cumint_desc)
    elif nFC_ic == "state":
        # Previous-iteration FE solutions undershoot to tiny negatives in
        # the core where n_FC has decayed away; floor them at a small
        # positive fraction of the separatrix value rather than reject.
        nFC_on_desc = np.clip(_state_on_desc('nFC', 'psi_nFC_eval'),
                              1e-15 * state.nFC_x0, None)
    else:
        raise ValueError(
            f"Unknown nFC_ic={nFC_ic!r}; expected 'solve' or "
            "'state'."
        )

    nFC_init = _scatter(nFC_on_desc)
    if np.any(nFC_init < 0):
        raise ValueError(
            f"nFC_init = {nFC_init} is negative, which is not allowed."
        )

    # ---- n_CX --------------------------------------------------------
    if nCX_ic == "solve":
        # Initial guess for nCX from the n_CX fluid governing equation
        # (Eq. (10) of Saarelma-Connor):
        #
        #   |V_CX| d/dx[ f_CX g n_CX ] = n_e (n_CX S_i - (S_CX/2) n_FC)
        #
        # Let u = f_CX g n_CX and tau = -x (inward distance, >= 0). Then
        #   du/dtau = -P(tau) u + Q(tau)
        # with
        #   P = n_e S_i / (|V_CX| f_CX g)
        #   Q = n_e S_CX n_FC / (2 |V_CX|)   <-- Note: f_CX is no longer here!
        # Integrating factor nu(tau) = exp(int_0^tau P dtau') gives
        #   u(tau) = (u(0) + int_0^tau nu Q dtau') / nu(tau)
        # with u(0) = f_CX(0) * g(0) * nCX_x0.
        # Finally, n_CX = u / (f_CX g).
        tau_desc = -x_desc                              # >= 0, ascending from 0

        P_desc = ne_desc * Si_desc / (Vcx_desc * fCX_desc * g_desc)
        Q_desc = ne_desc * Scx_desc * nFC_on_desc / (2.0 * Vcx_desc)

        int_P = cumulative_trapezoid(P_desc, tau_desc, initial=0.0)
        nu_desc = np.exp(int_P)
        nu_Q_int = cumulative_trapezoid(nu_desc * Q_desc, tau_desc, initial=0.0)

        u_desc_0 = fCX_desc[0] * g_desc[0] * state.nCX_x0
        u_desc = (u_desc_0 + nu_Q_int) / nu_desc

        nCX_init = _scatter(u_desc / (fCX_desc * g_desc))
    elif nCX_ic == "scale nFC":
        nCX_init = nFC_init * state.nCX_x0 / state.nFC_x0
    elif nCX_ic == "state":
        # Same floor as the manual n_FC branch, relative to nCX_x0.
        nCX_init = _scatter(np.clip(_state_on_desc('nCX', 'psi_nCX_eval'),
                                    1e-15 * state.nCX_x0, None))
    else:
        raise ValueError(
            f"Unknown nCX_ic={nCX_ic!r}."
        )

    return nFC_init, nCX_init


def resolve_integration_constant(state, dne_dx_neginf=None):
    """Saarelma's constant of integration C = dn_e/dx|_{x=-inf} (m^-4).

    ``dne_dx_neginf`` is used as given when not None; otherwise C is read
    off ``state.n_e`` at the pedestal top.

    **Not a boundary condition.**  In the original state this is the
    particle flux left over where the edge-neutral ionisation source has
    died out; it enters Eqs. (6)/(7) only through the combination
    ``(dn_e/dx - dn_e/dx|_in)``, which is the accumulated source between
    the deep interior and x.  Saarelma et al. take it from the measured
    gradient at the pedestal top, which is legitimate exactly when the
    neutrals really are extinguished there.

    Because it is a state constant rather than a boundary condition it is
    resolved independently of ``ne_bc_loc`` -- an ``"outer"`` solve still
    needs it, and it still comes from the pedestal top of ``state.n_e``.
    """
    if dne_dx_neginf is not None:
        val = float(dne_dx_neginf)
    else:
        val = _profile_gradient_at(state, float(state.x_inner))
    state.dne_dx_neginf = val
    return val


def reject_legacy_ne_inner_bc(ne_inner_bc):
    """Reject the removed Dirichlet-at-the-inner-boundary pathway.

    ``ne_inner_bc`` used to choose between a Dirichlet and a Neumann
    condition at x_inner.  Only the Neumann pathway survives (it is the
    ``ne_bc_loc="inner"`` half of the two supported layouts), so the
    argument is kept solely to give callers a clear error rather than
    silently changing the problem they posed.
    """
    val = str(ne_inner_bc).lower()
    if val != "neumann":
        raise ValueError(
            f"ne_inner_bc={ne_inner_bc!r} is no longer supported.  The only "
            f"two n_e boundary-condition pathways are ne_grad_bc_loc="
            f"'inner' (Neumann dn_e/dx at x_inner + Dirichlet n_e at the "
            f"separatrix) and 'outer' (Neumann dn_e/dx + Dirichlet n_e both "
            f"at the separatrix).  A Dirichlet condition at the inner "
            f"boundary is not one of them; drop ne_inner_bc and select the "
            f"pathway with ne_grad_bc_loc."
        )
    return val


def integrate_from_separatrix(fun, y0, x_grid, method="Radau",
                              rtol=1e-8, atol=1e-10, positive_index=0):
    """Solve the ``"outer"`` pathway as an initial value problem.

    In ``"outer"`` mode every condition sits at the separatrix, so the
    problem is an IVP, not a BVP: integrate ``Y' = fun(x, Y)`` with
    ``scipy.integrate.solve_ivp`` from ``Y(x_grid[-1]) = y0`` (the
    separatrix) inward to ``x_grid[0]`` (the inner boundary).

    ``fun`` uses the ``solve_bvp`` calling convention (vectorised: ``Y``
    of shape (n, k), returns (n, k)), so the same right-hand side serves
    both pathways.  Integration stops, and the solve is reported as
    failed, if component ``positive_index`` (n_e) reaches zero.

    Returns an ``OptimizeResult`` carrying the ``solve_bvp`` fields the
    solvers read -- ``x`` (ascending: ``x_grid`` plus the integrator's
    own steps), ``y``, ``sol``, ``success``, ``message``, ``status`` --
    so downstream code is pathway-agnostic.
    """
    x_grid = np.asarray(x_grid, dtype=float)

    def _hit_zero(t, y):
        return y[positive_index]
    _hit_zero.terminal = True
    _hit_zero.direction = -1        # positive -> zero as x decreases

    ivp = solve_ivp(fun, (x_grid[-1], x_grid[0]), np.asarray(y0, dtype=float),
                    method=method, rtol=rtol, atol=atol,
                    dense_output=True, vectorized=True, events=_hit_zero)

    if ivp.status == 0:
        x = np.union1d(x_grid, ivp.t)
        return OptimizeResult(x=x, y=ivp.sol(x), sol=ivp.sol, success=True,
                              status=0, message=ivp.message, nfev=ivp.nfev)

    if ivp.status == 1:
        msg = (f"n_e reached zero at x = {ivp.t_events[0][0]:.4e} "
               f"(solver units) while integrating from the separatrix "
               f"towards the inner boundary at x = {x_grid[0]:.4e}")
    else:
        msg = ivp.message
    return OptimizeResult(x=ivp.t[::-1], y=ivp.y[:, ::-1], sol=ivp.sol,
                          success=False, status=ivp.status, message=msg,
                          nfev=ivp.nfev)
