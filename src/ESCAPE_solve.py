# Loop between EPEDNN and coupled version of Saarelma-Connor model to find self-consistent pedestal
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import shutil
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parent.parent  # ESCAPE root (src/..)
sys.path.insert(0, str(ROOT))
from src.ESCAPE_api import ESCAPE
from src.ped_width_proxy import ped_width

def ESCAPE_solve(
    sc_params,
    epednn_params,
    Z_i = 1, # Z of ions
    P_tot_e = None, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
    ne_x0 = None, # m^-3, electron density at the separatrix (boundary condition, default is to use from profiles)        
    equil_params = None, # dictionary of required equilibrium parameters
    kprof_params = None, # dictionary of required kinetic profile (density, temperature) parameters
    T_rat_flag = True, # True if using a temperature ratio between ions and electrons, False if doing something else
    T_rat = 1,
    pol_norm = False, # True for when the poloidal flux is not normalized by 2pi. COCOS 7 convention is pol_norm=False, so poloidal flux is normalized by 2pi
    species = 'D', # species of ions, currently supporting: D, D-T
    regime_flag = 'PT H-mode', # regime of the plasma, currently supporting: 'PT H-mode', 'NT'
    ESCAPE_iter_max = 500, 
    ESCAPE_tol_max = 1e-5,
    out_dir = None, # directory to output files
    verbose = False,
):
    """Solve the self-consistent pedestal problem using the EPEDNN model and the Saarelma-Connor model.

    Parameters
    ----------
    sc_params : dict
        Dictionary of parameters for the Saarelma-Connor model solve (for options see src/saarelma_connor/saarelma_connor_base.py)
    epednn_params : dict
        Dictionary of parameters for the EPEDNN model solve (for options see src/epednn/epednn_call.py)
    out_dir : str
        Path to directory to save successful solutions from the Saarelma-Connor model.
    """

    te_plot_profiles = []
    ne_plot_profiles = []

    state = ESCAPE(
        Z_i = Z_i, # Z of ions
        P_tot_e = P_tot_e, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
        ne_x0 = ne_x0, # m^-3, electron density at the separatrix (boundary condition, default is to use from profiles)        
        equil_params = equil_params, # dictionary of required equilibrium parameters
        kprof_params = kprof_params, # dictionary of required kinetic profile (density, temperature) parameters
        T_rat_flag = T_rat_flag, # True if using a temperature ratio between ions and electrons, False if doing something else
        T_rat = T_rat,
        pol_norm = pol_norm, # True for when the poloidal flux is not normalized by 2pi. COCOS 7 convention is pol_norm=False, so poloidal flux is normalized by 2pi
        species = species, # species of ions, currently supporting: D, D-T
        regime_flag = regime_flag, # regime of the plasma, currently supporting: 'PT H-mode', 'NT'
        verbose = verbose,
    )
    print("Initial state built.")

    state.setup_epednn(model=epednn_params['model'])
    print("One-time EPEDNN setup was successful.")

    # Clear outputs from any previous scan (including appended failure logs) and setup logging files
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    for path in out_dir_p.iterdir(): # clear output directory
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


    # Parameters to be used in the loop
    tanh_width_new = None
    psi_N_Te_new = None
    T_prof_keV = None
    pedestal_height = None
    pedestal_width = None
    betan = -1 # initialize to -1 to indicate that betan is not yet calculated

    # ----------------------------- ESCAPE LOOP -----------------------------#
    for ESCAPE_iter in range(ESCAPE_iter_max):
        print("------------------------------------------------")
        print(f"ESCAPE Loop Iter {ESCAPE_iter}")

        if ESCAPE_iter == 0:
            SOLVE_KW['initial_guess'] = "tanh"
            neutral_ic = {'nCX_ic': "scale nFC", 'nFC_ic': "solve"}
        elif ESCAPE_iter > 0 and ig=='solve': # bc
            SOLVE_KW['initial_guess'] = "tanh"
            neutral_ic = {'nCX_ic': "solve", 'nFC_ic': "solve"}
        elif ESCAPE_iter > 0 and ig=='manual': # bc+ig
            SOLVE_KW['initial_guess'] = "manual EPEDNN loop" # use previous loop's profiles as initial guess for ne, nFC, nCX
            neutral_ic = {'nCX_ic': "manual EPEDNN loop", 'nFC_ic': "manual EPEDNN loop"}
        elif ESCAPE_iter > 0 and ig=='fix': # fix; only T changes, the densities are fixed from the pfile
            SOLVE_KW['initial_guess'] = "tanh"
            neutral_ic = {'nCX_ic': "solve", 'nFC_ic': "solve"}
        else:
            raise ValueError(f"Invalid initial guess mode: {ig}")
        if model == "3D":
            SOLVE_KW.update(neutral_ic)

        # Both models return the same schema, so the parse below is identical
        # for either one.  tanh_width is only used if initial_guess='tanh'.
        res = base_model.solve(model=model, tanh_width=tanh_width_new, **SOLVE_KW)
        x_sol, ne_sol = res['x'], res['ne']
        nFC_sol, nCX_sol = res['nFC'], res['nCX']
        T_e_pres, psi_N_pres = res['T_e_pres'], res['psi_N_pres']
        # x -> psi_N on the density solution grid (for output only; same map as below)
        psi_N_ne = interp1d(np.asarray(base_model.r_psi, dtype=float) - float(base_model.r_psi[-1]),
                            np.asarray(base_model.psi_N_pres, dtype=float),
                            kind='linear', bounds_error=False, fill_value='extrapolate')(x_sol)
        sol = {'x': x_sol, 'y': ne_sol, 'nFC': nFC_sol, 'nCX': nCX_sol,
               'T_e': T_e_pres, 'psi_N': psi_N_pres, 'psi_N_ne': psi_N_ne,
               'alpha_crit': alpha_crit, 'C_KBM': C_KBM, 'De_chie_etg': De_chie_etg, 'nFC_x0': nFC_x0, 'ncx_x0_ratio': ncx_x0_ratio,
               'betan': betan, 'ig': ig}
        best_x = np.asarray(sol['x'], dtype=float)
        best_ne = np.asarray(sol['y'], dtype=float)

        # psi_N <-> x map (reuse base_model)
        psi_N_pres = np.asarray(base_model.psi_N_pres, dtype=float)
        x_grid_full = np.asarray(base_model.r_psi, dtype=float) - float(base_model.r_psi[-1])
        psi_to_x = interp1d(psi_N_pres, x_grid_full, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        psi_ped_grid = np.linspace(psi_N_inner, 1.0, x_res)

        # --- Feed best profile into EPEDNN --------------------------------
        if ESCAPE_iter == 0:
            pedestal_height_prev = 0.0
            pedestal_width_prev = 0.0

            pw = ped_width(best_x, best_ne)

            pedestal_height, pedestal_width, betan = base_model.feed_epednn(model=epednn_model, ne_ped=best_ne, x_ne=best_x, EPEDNN_core='pfile', Z_eff=Z_eff, neped_x_loc=0-pw)    
        else:
            pedestal_height_prev, pedestal_width_prev = pedestal_height, pedestal_width

            pw = psi_to_x(1 - pedestal_width_prev)

            pedestal_height, pedestal_width, betan = base_model.feed_epednn(model=epednn_model, ne_ped=best_ne, x_ne=best_x, psiN_Te=psi_N_Te_new, Te_prev=T_prof_keV * 1e3, EPEDNN_core=EPEDNN_core, Z_eff=Z_eff, neped_x_loc=pw)
        tanh_width_new = psi_to_x(1-pedestal_width) * -1 # will error if result if negative which should not happen
        print(f"Pedestal height: {pedestal_height} MPa, Pedestal width: {pedestal_width} (psi_N)")

        if ESCAPE_iter > 0:
            eped_tol = abs((pedestal_height - pedestal_height_prev) / pedestal_height_prev) + abs((pedestal_width - pedestal_width_prev) / pedestal_width_prev)
            print(f"Normalized pedestal pressure height and width tolerance: {eped_tol}")
            sol['loop_tol'] = eped_tol


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

        # Store the EPEDNN-generated T_e profile (and the quantities that set it)
        # on `sol` and re-save, so downstream plotting can see the profile this
        # iteration produced rather than only the profile it was fed.
        sol['psi_N_Te_new'] = np.asarray(psi_N_Te_new, dtype=float)
        sol['Te_new_keV'] = np.asarray(T_prof_keV, dtype=float)
        sol['Te_ped_eV'] = float(Te_ped_eV)
        sol['ne_ped_val'] = float(ne_ped_val)
        sol['psi_ped'] = float(psi_ped)
        sol['pedestal_height'] = float(pedestal_height)
        sol['pedestal_width'] = float(pedestal_width)
        # Record which model / backend produced this iteration, so a saved
        # scan can be read back without guessing from the caller's script.
        sol['model'] = res['model']
        sol['solver_structure'] = res['solver_structure']
        sol['diagnostics'] = res['diagnostics']
        np.save(f'{out_dir_p}/ne_and_Te_iter_{ESCAPE_iter}.npy', sol, allow_pickle=True)

        if verbose:  # collect profile data for post-loop plotting
            Te_spliced_eV = T_prof_keV * 1e3

            if ESCAPE_iter == 0:
                te_plot_profiles.append({
                    'psi_N': np.asarray(psi_prev, dtype=float),
                    'y': np.asarray(Te_prev_keV * 1e3, dtype=float),
                    'label': 'Reference $T_e$',
                    'ls': '--',
                })
                ne_plot_profiles.append({
                    'psi_N': np.asarray(base_model.psi_N_pres, dtype=float),
                    'y': np.asarray(base_model.n_e_pres, dtype=float) / 1e19,
                    'label': 'Reference $n_e$',
                    'ls': '--',
                })

            te_plot_profiles.append({
                'psi_N': np.asarray(psi_N_Te_new, dtype=float),
                'y': np.asarray(Te_spliced_eV, dtype=float),
                'label': f'ESCAPE iteration {ESCAPE_iter}',
                'ls': '-',
            })

            x_to_psiN = interp1d(x_grid_full, psi_N_pres, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
            psi_N_ne = x_to_psiN(best_x)
            sort_idx = np.argsort(psi_N_ne)
            ne_plot_profiles.append({
                'psi_N': psi_N_ne[sort_idx],
                'y': (best_ne[sort_idx] / 1e19),
                'label': f'ESCAPE iteration {ESCAPE_iter}',
                'ls': '-',
            })

        if ESCAPE_iter > 0:
            if abs(eped_tol) < ESCAPE_tol_max:
                break
        
        # MAKE THIS PART FASTER
        x_to_psiN = interp1d(x_grid_full, psi_N_pres, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
        psi_N_ne = x_to_psiN(best_x)
        sort_idx = np.argsort(psi_N_ne)
        manual_profs = {
            'Te': T_prof_keV,
            'psi_N_Te': psi_N_Te_new,
            'ne': best_ne[sort_idx],
            'nCX': nCX_sol[sort_idx],
            'nFC': nFC_sol[sort_idx],
            'psi_N_n': psi_N_ne[sort_idx],
        }
        if ig == 'manual' or ig == 'solve':
            base_model = saarelma_connor( # reset base_model to the new T_e profile and define new "p-file" profiles from the manual profiles
                P_tot_e      = P_tot_e,
                species      = species,
                alpha_crit   = alpha_crit,
                C_KBM        = C_KBM,
                De_chie_etg  = De_chie_etg,
                nFC_x0       = nFC_x0,
                ncx_x0_ratio = ncx_x0_ratio,
                mhd_fp       = MHD_FP,
                kprof_loc    = 'manual EPEDNN loop',
                kprof_fp = KPROF_FP,
                manual_profs = manual_profs,
                verbose      = False,
                psi_N_inner_boundary = psi_N_inner,
            )
        elif ig == 'fix':
            base_model = saarelma_connor( # reset base_model to the new T_e profile and use the p-file n_e
                P_tot_e      = P_tot_e,
                species      = species,
                alpha_crit   = alpha_crit,
                C_KBM        = C_KBM,
                De_chie_etg  = De_chie_etg,
                nFC_x0       = nFC_x0,
                ncx_x0_ratio = ncx_x0_ratio,
                mhd_fp       = MHD_FP,
                kprof_loc    = 'pfile',
                kprof_fp     = KPROF_FP,
                manual_profs = manual_profs,
                verbose      = False,
                psi_N_inner_boundary = psi_N_inner,
            )
        base_model.setup_epednn(model=epednn_model)

    if te_plot_profiles:
        red_blue = LinearSegmentedColormap.from_list('red_blue', ['red', 'blue'])
        te_colors = red_blue(np.linspace(0, 1, len(te_plot_profiles)))
        ne_colors = red_blue(np.linspace(0, 1, len(ne_plot_profiles)))

        fig, ax = plt.subplots(figsize=(6, 4))
        for prof, color in zip(te_plot_profiles, te_colors):
            ax.plot(prof['psi_N'], prof['y'] / 1e3, lw=2, ls=prof['ls'],
                    color=color, label=prof['label'])
        ax.set_xlabel(r'$\psi_N$')
        ax.set_ylabel(r'$T_e$ [keV]')
        ax.set_title('Solved $T_e$ profiles')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(psi_N_inner, 1.0)
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
        ax1.set_xlim(psi_N_inner, 1.0)
        fig1.tight_layout()
        plt.show()

    return pedestal_width, pedestal_height, sol