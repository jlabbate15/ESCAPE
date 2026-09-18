# Loop between EPEDNN and coupled version of Saarelma-Connor model to find self-consistent pedestal

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d

ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(ROOT))

from src.solver_nondim import saarelma_connor_nondim

def profiles_loop_solve_scloop(
    MHD_FP = None,
    KPROF_FP = None,
    ne_success_fp = 'compare_nondim',
    initial_guess = "tanh",
    ne_inner_bc = "neumann",
    success_fp = 'success_PTHmode.txt',
    failure_fp = 'failure_PTHmode.txt',
    error_messages_fp = 'error_messages_PTHmode.txt',
    x_res = 40,
    N = 3,
    P_tot_e = 5e6, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
    psi_N_inner = 0.85,
    alpha_crits_minmax = [-1,1],
    C_KBMs_minmax = [-1,1],
    De_chie_etgs_minmax = [-1,1],
    nFC_x0s_minmax = [14,18],
    ncx_x0_ratios_minmax = [0.1,10],
    eped_tol_max = 1e-3,
    eped_iter_max = 50,
    kbm_gate_eps = 0.01,
    EPEDNN_core = 'pfile',
    kbm_treatment = "inline",
    picard_gate_mode = None,
    picard_max_it = 50,
    picard_rtol = 1e-8,
    picard_relax = 1.0,
    verbose = False,
):
    """Solve the self-consistent pedestal problem using the EPEDNN model and the Saarelma-Connor model.

    Parameters
    ----------
    MHD_FP : str
        Path to MHD equilibrium file.
    KPROF_FP : str
        Path to KPROF profile file.
    ne_success_fp : str
        Path to directory to save successfull solutions.
    initial_guess : str
        Initial guess for the pedestal height and width.
    ne_inner_bc : str
        Boundary condition for the electron density at the inner boundary.
    x_res : int
        Number of grid points in the radial direction.
    P_tot_e : float
        Total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox.
    psi_N_inner : float
        Inner boundary of the pedestal in normalized poloidal flux.
        Per-DOF x coordinates (m), unsorted (as stored in dat.data).
    free_params : dict
        Dictionary of free parameters for the EPEDNN model.
    eped_tol_max : float
        Maximum tolerance for the pedestal height and width.
    eped_iter_max : int
        Maximum number of iterations for the EPEDNN model.
    kbm_treatment : str
        Treatment for the KBM.
    picard_gate_mode : str
        Mode for the Picard gate.
    picard_max_it : int
        Maximum number of iterations for the Picard gate.
    picard_rtol : float
        Relative tolerance for the Picard gate.
    picard_relax : float
        Relaxation factor for the Picard gate.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    psi_N_inner_boundary_new : float
        Inner boundary of the pedestal in normalized poloidal flux.
    T_prof_keV : float
        Temperature profile in keV.
    best_x : ndarray
        Best x coordinates.
    best_ne : ndarray
        Best electron density profile.
    pedestal_height : float
        Final pedestal pressure height in MPa.
    """

    te_plot_profiles = []
    ne_plot_profiles = []

    # Load in free parameters
    alpha_crits = np.logspace(alpha_crits_minmax[0], alpha_crits_minmax[1], N)
    C_KBMs = np.logspace(C_KBMs_minmax[0], C_KBMs_minmax[1], N)
    De_chie_etgs = np.logspace(De_chie_etgs_minmax[0], De_chie_etgs_minmax[1], N)
    nFC_x0s = np.logspace(nFC_x0s_minmax[0], nFC_x0s_minmax[1], N)
    ncx_x0_ratios = np.logspace(ncx_x0_ratios_minmax[0],ncx_x0_ratios_minmax[1],N)

    # Setup solver parameters
    SOLVE_KW = dict(
        x_res=x_res,
        fe_degree=2,
        initial_guess=initial_guess,
        ne_inner_bc=ne_inner_bc,   # Saarelma A7 default; see dirichlet comparison below
        linear_solver="lu",      # or "gamg" for GMRES + algebraic multigrid on J
        nCX_ic="solve",
        kbm_treatment=kbm_treatment,
        kbm_gate_eps=kbm_gate_eps, # 1e-3 minimum
        picard_gate_mode=picard_gate_mode,
        picard_max_it=picard_max_it,
        picard_rtol=picard_rtol,
        picard_relax=picard_relax,
        verbose=False,
    )

    base_model = saarelma_connor_nondim(
            P_tot_e      = P_tot_e,
            alpha_crit   = alpha_crits[0],
            C_KBM        = C_KBMs[0],
            De_chie_etg  = De_chie_etgs[0],
            nFC_x0       = nFC_x0s[0],
            ncx_x0_ratio = ncx_x0_ratios[0],
            psi_N_inner_boundary = psi_N_inner,
            mhd_fp       = MHD_FP,
            kprof_fp     = KPROF_FP,
            verbose      = False,
            # psi_N_inner_boundary = 0.85, # set to None to use adaptive inner boundary method
    )
    psi_N_inner_boundary_new = psi_N_inner
    base_model.setup_epednn()
    print("Base model built.")

    # Parameters to be used in the loop
    prev_best_x = None
    prev_best_ne = None
    tanh_width_new = None
    psi_N_Te_new = None
    T_prof_keV = None
    pedestal_height = None
    pedestal_width = None

    for eped_iter in range(eped_iter_max):

        # Setup output directory; keep scan results from all EPED iterations.
        ne_success_dir = Path(ne_success_fp)
        ne_success_dir.mkdir(parents=True, exist_ok=True)
        iter_tag = f"iter{eped_iter:03d}"
        iter_success_fp = Path(success_fp).with_name(
            f"{Path(success_fp).stem}_{iter_tag}{Path(success_fp).suffix}"
        )
        iter_failure_fp = Path(failure_fp).with_name(
            f"{Path(failure_fp).stem}_{iter_tag}{Path(failure_fp).suffix}"
        )
        iter_error_fp = Path(error_messages_fp).with_name(
            f"{Path(error_messages_fp).stem}_{iter_tag}{Path(error_messages_fp).suffix}"
        )
        with open(iter_success_fp, "w") as f:
            f.write("alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, psi_N_inner\n")
        with open(iter_failure_fp, "w") as f:
            f.write("alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, psi_N_inner\n")
        with open(iter_error_fp, "w") as f:
            f.write("alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, psi_N_inner, message\n")

        # Scan free parameters for SC model
        i=0
        for alpha_crit in alpha_crits:
                for C_KBM in C_KBMs:
                        for De_chie_etg in De_chie_etgs:
                                for nFC_x0 in nFC_x0s:
                                        for ncx_x0_ratio in ncx_x0_ratios:
                                                ac = round(float(alpha_crit), 3)
                                                ck = round(float(C_KBM), 3)
                                                de = round(float(De_chie_etg), 3)
                                                nf = round(float(nFC_x0), 3)
                                                nc = round(float(ncx_x0_ratio),3)
                                                try:
                                                        base_model.update_free_params(
                                                                alpha_crit            = ac,
                                                                C_KBM                 = ck,
                                                                De_chie_etg           = de,
                                                                nFC_x0                = nf,
                                                                ncx_x0_ratio          = nc,
                                                        )
                                                        x_sol, ne_sol, nFC_sol, nCX_sol = base_model.solve_coupled_nondim(**SOLVE_KW)
                                                        sol = {'x': x_sol, 'y': ne_sol, 'nFC': nFC_sol, 'nCX': nCX_sol}
                                                except Exception as e: # run fails
                                                        with open(iter_failure_fp, 'a') as f:
                                                                f.write(f"{ac}, {ck}, {de}, {nf}, {nc}, {psi_N_inner:.4f}\n")
                                                        with open(iter_error_fp, 'a') as f:
                                                                f.write(f"{ac}, {ck}, {de}, {nf}, {nc}, {psi_N_inner:.4f}, {e}\n")
                                                else: # run works
                                                        with open(iter_success_fp, 'a') as f:
                                                                f.write(f"{ac}, {ck}, {de}, {nf}, {nc}, {psi_N_inner:.4f}\n")
                                                        np.save(
                                                            ne_success_dir / (
                                                                f'ne_{iter_tag}_a{ac}_C{ck}_D{de}_n{nf}_nc{nc}'
                                                                f'_b{psi_N_inner:.4f}'
                                                            ),
                                                            sol,
                                                            allow_pickle=True,
                                                        )
                # print(f"Completed {i} of {len(alpha_crits)} alpha_crits") # progress logging

        # --- Pick best ne profile -----------------------------------------
        n_combos = (len(alpha_crits) * len(C_KBMs) * len(De_chie_etgs)
                    * len(nFC_x0s) * len(ncx_x0_ratios))

        # psi_N <-> x map (reuse base_model; no extra solver instantiation)
        psi_N_pres = np.asarray(base_model.psi_N_pres, dtype=float)
        x_grid_full = np.asarray(base_model.r_psi, dtype=float) - float(base_model.r_psi[-1])
        psi_to_x = interp1d(psi_N_pres, x_grid_full, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        psi_ped_grid = np.linspace(psi_N_inner, 1.0, x_res)
        x_ped_grid = psi_to_x(psi_ped_grid)

        if n_combos <= 1:
            # Single-combination run: the only sol from the inner loop IS the best.
            best_x = np.asarray(sol['x'], dtype=float)
            best_ne = np.asarray(sol['y'], dtype=float)
        else:
            if KPROF_FP is None:
                raise ValueError("Must specify pfile for a parameter scan implementation "
                                 "of SC model in EPEDNN loop")

            # Read p-file ne once
            psi_pfile, ne_pfile = [], []
            in_ne = False
            with open(KPROF_FP) as f:
                for line in f:
                    if '3 N Z A' in line:
                        break
                    if line.startswith('201'):
                        in_ne = 'ne(10^20/m^3)' in line
                        continue
                    if in_ne:
                        p, v, _ = line.split()
                        psi_pfile.append(float(p)); ne_pfile.append(float(v))
            psi_pfile = np.asarray(psi_pfile); ne_pfile = np.asarray(ne_pfile) * 1e20
            ne_pfile_on_ped = interp1d(psi_pfile, ne_pfile, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')(psi_ped_grid)
            mean_ne_pfile = float(np.mean(ne_pfile_on_ped))

            # Sweep .npy successes, keep minimum-L2 profile
            best_l2 = np.inf
            best_x = None
            best_ne = None
            for npy_path in Path(ne_success_fp).glob(f"ne_{iter_tag}_*.npy"):
                sol_i = np.load(npy_path, allow_pickle=True).item()
                x_i = np.asarray(sol_i['x'], dtype=float)
                ne_i = np.asarray(sol_i['y'], dtype=float)
                ne_on_ped = interp1d(x_i, ne_i, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')(x_ped_grid)
                l2 = float(np.sqrt(np.mean((ne_on_ped - ne_pfile_on_ped) ** 2)) / mean_ne_pfile)
                if l2 < best_l2:
                    best_l2, best_x, best_ne = l2, x_i, ne_i
            if best_ne is None:
                raise RuntimeError(f"No successful runs found in {ne_success_fp}")
            print(f"Best normalized L2 = {best_l2:.4f}")

        # --- Feed best profile into EPEDNN --------------------------------
        if eped_iter == 0:
            pedestal_height_prev = 0.0
            pedestal_width_prev = 0.0
            pedestal_height, pedestal_width = base_model.feed_epednn(ne_ped=best_ne, x_ne=best_x, EPEDNN_core='pfile')
        else:
            pedestal_height_prev, pedestal_width_prev = pedestal_height, pedestal_width
            pedestal_height, pedestal_width = base_model.feed_epednn(ne_ped=best_ne, x_ne=best_x, psiN_Te=psi_N_Te_new, Te_prev=T_prof_keV * 1e3, EPEDNN_core=EPEDNN_core)
        tanh_width_new = psi_to_x(1-pedestal_width) * -1 # will error if result if negative which should not happen
        print(f"Pedestal height: {pedestal_height} MPa, Pedestal width: {pedestal_width} (psi_N)")

        if eped_iter > 0:
            eped_tol = ((pedestal_height - pedestal_height_prev) / pedestal_height_prev
                        + (pedestal_width - pedestal_width_prev) / pedestal_width_prev)
            print(f"Normalized pedestal pressure height and width tolerance: {eped_tol}")
            if abs(eped_tol) < eped_tol_max:
                break

        # --- New T_e profile (EPED1 tanh form, Eq. 1b without core H term) ---
        #   T(psi) = T_sep + aT0 * { tanh[2(1 - psi_mid)/Delta]
        #                          - tanh[2(psi - psi_mid)/Delta] }
        # On [psi_ped, 1] the tanh shape peaks at psi_ped; aT0 is fixed so
        # T(psi_ped) = Te_ped derived from EPED pedestal pressure (p_ped =
        # 2 * ne_ped * Te_ped, Ti = Te, Zeff ~ 1).
        Delta = float(pedestal_width)
        psi_mid = 1.0 - 0.5 * Delta
        psi_ped = 1.0 - Delta
        ne_ped_val = float(interp1d(best_x, best_ne, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')(
                                        psi_to_x(psi_ped)))
        T_sep_eV = float(np.asarray(base_model.T_e)[-1] * 1e3)  # keV -> eV
        eV_to_J = 1.602176634e-19
        Te_ped_eV = (float(pedestal_height) * 1.0e6 / (2.0 * ne_ped_val)) / eV_to_J
        tanh_peak = 2.0 * np.tanh(1.0)  # shape-function maximum on [psi_ped, 1]
        aT0 = (Te_ped_eV - T_sep_eV) / tanh_peak

        def _eped1_tanh_Te(psi_N):
            return T_sep_eV + aT0 * (
                np.tanh(2.0 * (1.0 - psi_mid) / Delta)
                - np.tanh(2.0 * (psi_N - psi_mid) / Delta)
            )

        # EPED tanh only on the pedestal strip [psi_ped, 1]; splice onto p-file core.
        psi_tanh = np.linspace(psi_ped, 1.0, x_res)
        Te_tanh_eV = _eped1_tanh_Te(psi_tanh)

        psi_prev = np.asarray(base_model.psi_Te_eval, dtype=float)
        Te_prev_keV = np.asarray(base_model.T_e, dtype=float)

        # Add offset to T_e core profile
        Te_prev_keV_ped = interp1d(psi_prev, Te_prev_keV, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_ped)
        Te_tanh_eV_ped = interp1d(psi_tanh, Te_tanh_eV / 1e3, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_ped)
        T_e_offset = Te_tanh_eV_ped - Te_prev_keV_ped # ped - core at psi_ped

        keep = psi_prev < psi_ped
        psi_N_Te_new = np.concatenate([psi_prev[keep], psi_tanh])
        T_prof_keV = np.concatenate([Te_prev_keV[keep] + T_e_offset, Te_tanh_eV / 1e3])

        psi_N_inner_boundary_new = psi_ped

        if verbose:  # collect profile data for post-loop plotting
            Te_spliced_eV = T_prof_keV * 1e3
            print(f"  Te_ped = {Te_ped_eV:.1f} eV, T_sep = {T_sep_eV:.1f} eV "
                  f"(ne_ped = {ne_ped_val:.3e} m^-3, psi_ped = {psi_ped:.4f}, "
                  f"Delta = {Delta:.4f})")
            print(f"  psi_N_inner_boundary_new = {psi_N_inner_boundary_new:.4f}")
            og_Te_peak = float(interp1d(psi_prev, Te_prev_keV * 1e3, kind='linear',
                                        bounds_error=False, fill_value='extrapolate')(psi_ped))
            # print(f"Percent change from previous T_e at psi_ped = "
            #       f"{(100*(Te_ped_eV - og_Te_peak) / og_Te_peak):.4f}")

            if eped_iter == 0:
                te_plot_profiles.append({
                    'psi_N': np.asarray(psi_prev, dtype=float),
                    'y': np.asarray(Te_prev_keV * 1e3, dtype=float),
                    'label': 'p-file $T_e$',
                    'ls': '--',
                })
                ne_plot_profiles.append({
                    'psi_N': np.asarray(base_model.psi_N_pres, dtype=float),
                    'y': np.asarray(base_model.n_e_pres, dtype=float) / 1e19,
                    'label': 'p-file $n_e$',
                    'ls': '--',
                })

            te_plot_profiles.append({
                'psi_N': np.asarray(psi_N_Te_new, dtype=float),
                'y': np.asarray(Te_spliced_eV, dtype=float),
                'label': f'Iter {eped_iter}',
                'ls': '-',
            })

            x_to_psiN = interp1d(x_grid_full, psi_N_pres, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
            psi_N_ne = x_to_psiN(best_x)
            sort_idx = np.argsort(psi_N_ne)
            ne_plot_profiles.append({
                'psi_N': psi_N_ne[sort_idx],
                'y': (best_ne[sort_idx] / 1e19),
                'label': f'Iter {eped_iter}',
                'ls': '-',
            })

            fig,ax = plt.subplots(figsize=(6, 4))
            ax.plot(psi_N_ne[sort_idx], (best_ne[sort_idx] / 1e19), lw=2, ls='-',
                    color='blue', label=f'Iter {eped_iter}')
            ax.set_xlabel(r'$\psi_N$')
            ax.set_ylabel(r'$n_e$ ($10^{19}$ m$^{-3}$)')
            ax.set_title('Solved $n_e$ profiles')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_xlim(0.8, 1.0)

            fig,ax = plt.subplots(figsize=(6, 4))
            ax.plot(psi_N_Te_new, Te_spliced_eV, lw=2, ls='-',
                    color='blue', label=f'Iter {eped_iter}')
            ax.set_xlabel(r'$\psi_N$')
            ax.set_ylabel(r'$T_e$ [eV]')
            ax.set_title('Solved $T_e$ profiles')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_xlim(0.8, 1.0)
            ax.set_ylim(0, 10000)

        # MAKE THIS PART FASTER
        base_model = saarelma_connor_nondim( # reset base_model to the new T_e profile and related quantities
            P_tot_e      = P_tot_e,
            alpha_crit   = alpha_crit,
            C_KBM        = C_KBM,
            De_chie_etg  = De_chie_etg,
            nFC_x0       = nFC_x0,
            ncx_x0_ratio = ncx_x0_ratio,
            mhd_fp       = MHD_FP,
            kprof_fp     = KPROF_FP,
            verbose      = False,
            psi_N_inner_boundary = psi_N_inner,
            T_e_source = 'epednn',
            T_prof = T_prof_keV,
            T_prof_psi_N = psi_N_Te_new,
        )
        base_model.setup_epednn()

    gfile_pres_grid = base_model.psi_N_pres
    gfile_pres = base_model.pres_gfile

    if te_plot_profiles:
        red_blue = LinearSegmentedColormap.from_list('red_blue', ['red', 'blue'])
        te_colors = red_blue(np.linspace(0, 1, len(te_plot_profiles)))
        ne_colors = red_blue(np.linspace(0, 1, len(ne_plot_profiles)))

        fig, ax = plt.subplots(figsize=(6, 4))
        for prof, color in zip(te_plot_profiles, te_colors):
            ax.plot(prof['psi_N'], prof['y'], lw=2, ls=prof['ls'],
                    color=color, label=prof['label'])
        ax.set_xlabel(r'$\psi_N$')
        ax.set_ylabel(r'$T_e$ [eV]')
        ax.set_title('Solved $T_e$ profiles')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0.8, 1.0)
        fig.tight_layout()

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        for prof, color in zip(ne_plot_profiles, ne_colors):
            ax1.plot(prof['psi_N'], prof['y'], lw=2, ls=prof['ls'],
                     color=color, label=prof['label'])
        ax1.set_xlabel(r'$\psi_N$')
        ax1.set_ylabel(r'$n_e$ ($10^{19}$ m$^{-3}$)')
        ax1.set_title('Solved $n_e$ profiles')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0.8, 1.0)
        fig1.tight_layout()
        plt.show()

    return pedestal_width, pedestal_height, gfile_pres, gfile_pres_grid