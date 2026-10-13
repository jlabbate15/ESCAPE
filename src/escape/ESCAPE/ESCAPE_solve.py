# Loop between EPEDNN and coupled version of Saarelma-Connor model to find self-consistent pedestal
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import shutil
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[3]  # ESCAPE root (src/escape/ESCAPE/..)
sys.path.insert(0, str(ROOT))
from escape.ESCAPE.ESCAPE_api import ESCAPE
from helpers.ped_width_proxy import ped_width


def ESCAPE_solve(
    sc_params,
    epednn_params,
    Z_eff = 1,
    Z_i = 1, # Z of ions
    P_tot_e = None, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
    ne_x0 = None, # m^-3, electron density at the separatrix (boundary condition, default is to use from profiles)        
    tanh_center_psiN = None, # psi_N, centre of the tanh n_e initial guess (converted to SI x for sc_params['tanh_center']); None keeps sc_params as is, >= 1 uses the tanh_center=None default
    psiN_pwg = None, # psi_N, pedestal top (1 - pedestal width) for the 'tanh_linear' n_e initial guess (converted to SI x for sc_params['x_pwg'])
    equil_params = None, # dictionary of required equilibrium parameters
    kprof_params = None, # dictionary of required kinetic profile (density, temperature) parameters
    quasineutral_flag = True,
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
    tanh_center_psiN : float or None
        Centre of the tanh n_e initial guess in normalized poloidal flux. On the
        first ESCAPE iteration it is mapped to the SI x grid (m, 0 at the
        separatrix), clipped to lie inside x_inner, and stored as
        sc_params['tanh_center']. A value >= 1 (at or past the separatrix)
        is dropped and the solver's tanh_center=None (midpoint) default used.
    psiN_pwg : float or None
        Pedestal top (psi_N = 1 - pedestal width) of the 'tanh_linear' n_e
        initial guess. On the first ESCAPE iteration it is mapped to the SI x
        grid, clipped to lie inside (x_inner, 0), and stored as
        sc_params['x_pwg'].
    out_dir : str
        Path to directory to save successful solutions from the Saarelma-Connor model.
    """

    state = ESCAPE(
        Z_eff = Z_eff,
        Z_i = Z_i, # Z of ions
        P_tot_e = P_tot_e, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
        ne_x0 = ne_x0, # m^-3, electron density at the separatrix (boundary condition, default is to use from profiles)        
        equil_params = equil_params, # dictionary of required equilibrium parameters
        kprof_params = kprof_params, # dictionary of required kinetic profile (density, temperature) parameters
        quasineutral_flag = quasineutral_flag,
        T_rat_flag = T_rat_flag, # True if using a temperature ratio between ions and electrons, False if doing something else
        T_rat = T_rat,
        pol_norm = pol_norm, # True for when the poloidal flux is not normalized by 2pi. COCOS 7 convention is pol_norm=False, so poloidal flux is normalized by 2pi
        species = species, # species of ions, currently supporting: D, D-T
        regime_flag = regime_flag, # regime of the plasma, currently supporting: 'PT H-mode', 'NT'
        verbose = verbose,
    )
    print("Initial state built.")

    state.setup_epednn(model=epednn_params['epednn_model'])
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
    psi_N_Te_new = None
    T_prof_keV = None
    pedestal_height = None
    pedestal_width = None
    betan = -1 # initialize to -1 to indicate that betan is not yet calculated

    # psi_N <-> x map
    psi_N_pres = np.asarray(state.psi_N_pres, dtype=float)
    x_grid_full = np.asarray(state.r_psi, dtype=float) - float(state.r_psi[-1])
    psi_to_x = interp1d(psi_N_pres, x_grid_full, kind='linear',bounds_error=False, fill_value='extrapolate')
    x_to_psi = interp1d(x_grid_full,psi_N_pres, kind='linear',bounds_error=False, fill_value='extrapolate')


    # Requested first-iteration guess, read before the loop overwrites
    # sc_params['initial_guess'] with 'state'
    initial_guess_0 = "tanh_linear" if sc_params.get('initial_guess') == "tanh_linear" else "tanh"

    # ----------------------------- ESCAPE LOOP -----------------------------#
    for ESCAPE_iter in range(ESCAPE_iter_max):
        print("------------------------------------------------")
        print(f"ESCAPE Loop Iter {ESCAPE_iter}")

        # First iteration uses a tanh guess ('tanh_linear' if requested,
        # otherwise 'tanh'); later iterations start from the state
        if ESCAPE_iter == 0:
            sc_params['initial_guess'] = initial_guess_0
        else:
            sc_params['initial_guess'] = "state"

        # SAARELMA-CONNOR SOLVE
        state.init_saarelmaconnor(params=sc_params)

        # tanh initial-guess centre: psi_N -> SI x, kept strictly inside
        # (x_inner, 0) as build_ne_initial_guess requires.  A centre at or
        # past the separatrix is dropped (tanh_center=None, the midpoint
        # default); one inside the inner boundary is clipped.
        if ESCAPE_iter == 0 and tanh_center_psiN is not None and tanh_center_psiN >= 1.0:
            print(f"tanh_center: psi_N = {tanh_center_psiN:.4f} >= 1 (at/past the separatrix); "
                  f"dropped, using the tanh_center=None default")
            sc_params['tanh_center'] = None
            state.tanh_center_used = None
        elif ESCAPE_iter == 0 and tanh_center_psiN is not None:
            xc = float(psi_to_x(tanh_center_psiN))
            x_in = float(state.x_inner)
            margin = 0.02 * abs(x_in)
            xc_clip = float(np.clip(xc, x_in + margin, -margin))
            if xc_clip != xc:
                print(f"tanh_center: psi_N = {tanh_center_psiN:.4f} -> x = {xc:.4e} m lies "
                      f"outside (x_inner, 0) = ({x_in:.4e}, 0) m; clipped to {xc_clip:.4e} m")
            sc_params['tanh_center'] = xc_clip
            state.tanh_center_used = xc_clip

        # 'tanh_linear' pedestal top: psi_N -> SI x, kept inside (x_inner, 0)
        if ESCAPE_iter == 0 and psiN_pwg is not None:
            xp = float(psi_to_x(psiN_pwg))
            x_in = float(state.x_inner)
            margin = 0.02 * abs(x_in)
            xp_clip = float(np.clip(xp, x_in + margin, -margin))
            if xp_clip != xp:
                print(f"x_pwg: psi_N = {psiN_pwg:.4f} -> x = {xp:.4e} m lies "
                      f"outside (x_inner, 0) = ({x_in:.4e}, 0) m; clipped to {xp_clip:.4e} m")
            sc_params['x_pwg'] = xp_clip
            state.x_pwg_used = xp_clip

        res = state.solve_scmodel(**sc_params) # model and implementation are keys of sc_params
        x_sol, ne_sol = res['x'], res['ne']
        nFC_sol, nCX_sol = res['nFC'], res['nCX']
        T_e_pres, psi_N_pres = res['T_e_pres'], res['psi_N_pres']
        sol = {'x': x_sol, 'ne': ne_sol, 'nFC': nFC_sol, 'nCX': nCX_sol,
               'T_e': T_e_pres, 'psi_N': psi_N_pres,'SOLVE_KW': sc_params}

        # Raise density core profile to match new density pedestal
        psi_N_core = np.linspace(0, state.psi_N_inner_boundary, 75)
        _n_e_core = interp1d(state.psi_ne_eval, state.n_e, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_N_core)
        n_e_core_at_ped = interp1d(state.psi_ne_eval, state.n_e, kind='linear', bounds_error=False, fill_value='extrapolate')(state.psi_N_inner_boundary)
        ne_ped_at_top = interp1d(x_sol, ne_sol, kind='linear', bounds_error=False, fill_value='extrapolate')(state.x_inner)
        delta = ne_ped_at_top - n_e_core_at_ped
        n_e_core = _n_e_core + delta # change core stiffly to match pedestal top
        
        # Calculate total n_e
        psi_N_ped = interp1d(state.x_init, state.psi_N_pres, kind='linear', bounds_error=False, fill_value='extrapolate')(x_sol)
        psi_N_plasma = np.concatenate([psi_N_core[:-1], psi_N_ped])
        n_e_plasma = np.concatenate([n_e_core[:-1], ne_sol])

        # Update state (n_i follows n_e when quasineutral_flag, T_i follows T_e when T_rat_flag)
        state.set_kprof_params({
            'n_e': n_e_plasma,   'psin_ne': psi_N_plasma,
            'nFC': nFC_sol,      'psin_nFC': x_to_psi(x_sol),
            'nCX': nCX_sol,      'psin_nCX': x_to_psi(x_sol),
        })

        # EPEDNN SOLVE with updated state
        if ESCAPE_iter == 0:
            pedestal_height_prev = 0.0
            pedestal_width_prev = 0.0
            pw = ped_width(state.psi_ne_eval, state.n_e) # pedestal width from proxy for first ESCAPE iteration

            pedestal_height, pedestal_width, betan = state.feed_epednn(model=epednn_params['epednn_model'], psin_ped=pw, pres_gfile=epednn_params['pres_gfile'])    
        else:
            pedestal_height_prev, pedestal_width_prev = pedestal_height, pedestal_width

            pedestal_height, pedestal_width, betan = state.feed_epednn(model=epednn_params['epednn_model'], psin_ped=pedestal_width_prev, pres_gfile=epednn_params['pres_gfile'])

        if ESCAPE_iter > 0:
            eped_tol = abs((pedestal_height - pedestal_height_prev) / pedestal_height_prev) + abs((pedestal_width - pedestal_width_prev) / pedestal_width_prev)
            print(f"Normalized pedestal pressure height and width tolerance: {eped_tol}")
            sol['loop_tol'] = eped_tol

        # --- Construct new T_e profile (EPED1 tanh form without core H term) ---
        #   T(psi) = T_sep + aT0 * { tanh[2(1 - psi_mid)/Delta]
        #                          - tanh[2(psi - psi_mid)/Delta] }
        # On [psi_ped, 1] the tanh shape peaks at psi_ped; aT0 is fixed so
        # T(psi_ped) = Te_ped derived from EPED pedestal pressure (p_ped =
        # 2 * ne_ped * Te_ped, Ti = Te, Zeff ~ 1).
        Delta = float(pedestal_width)
        psi_mid = 1.0 - 0.5 * Delta
        psi_ped = 1.0 - Delta
        ne_ped_val = float(interp1d(state.psi_ne_eval, state.n_e, kind='linear',bounds_error=False, fill_value='extrapolate')(psi_ped))
        T_sep_eV = float(np.asarray(state.T_e)[-1] * 1e3)  # keV -> eV
        eV_to_J = 1.602176634e-19
        Te_ped_eV = (float(pedestal_height) * 1.0e6 / (2.0 * ne_ped_val)) / eV_to_J
        tanh_peak = 2.0 * np.tanh(1.0)  # shape-function maximum on [psi_ped, 1]
        aT0 = (Te_ped_eV - T_sep_eV) / tanh_peak

        def _eped1_tanh_Te(psi_N):
            return T_sep_eV + aT0 * (
                np.tanh(2.0 * (1.0 - psi_mid) / Delta)
                - np.tanh(2.0 * (psi_N - psi_mid) / Delta)
            )

        # New temperature pedestal profiles
        psi_tanh = np.linspace(psi_ped, 1.0, len(state.psi_Te_eval))
        Te_tanh_eV = _eped1_tanh_Te(psi_tanh)

        # Extract state from previous iteration (or initialized state)
        psi_prev = np.asarray(state.psi_Te_eval, dtype=float)
        Te_prev_keV = np.asarray(state.T_e, dtype=float)

        # Add offset to T_e core profile to match new pedestal temperature profile
        Te_prev_keV_ped = interp1d(psi_prev, Te_prev_keV, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_ped)
        Te_tanh_eV_ped = interp1d(psi_tanh, Te_tanh_eV / 1e3, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_ped)
        T_e_offset = Te_tanh_eV_ped - Te_prev_keV_ped # ped - core at psi_ped

        # Construct full (core+pedestal) temperature profile
        keep = psi_prev < psi_ped
        psi_N_Te_new = np.concatenate([psi_prev[keep], psi_tanh])
        T_prof_keV = np.concatenate([Te_prev_keV[keep] + T_e_offset, Te_tanh_eV / 1e3])

        # -- -- -- -- -- -- ELM-free SCALING -- -- -- -- -- --
        '''
        NOT YET IMPLEMENTED

        # Apply ELM-free regime scaling
        if self.regime_flag == 'PT H-mode':
            pass
        elif self.regime_flag == 'NT':
            self.pedestal_pressure = self.pedestal_pressure * self.NT_scaling
            raise NotImplementedError('NT ELM-free regime scaling is not yet implemented')
        else:
            assert False, 'specified regime_flag not supported'
        '''
        
        # Update temperature state (include core profile); T_i follows T_e when T_rat_flag
        state.set_kprof_params({'T_e': T_prof_keV, 'psin_Te': psi_N_Te_new})

        # Store the new state and resave iteration
        sol['psi_Te_eval'] = state.psi_Te_eval
        sol['T_e'] = state.T_e
        sol['psi_ne_eval'] = state.psi_ne_eval
        sol['n_e'] = state.n_e
        sol['Te_ped_eV'] = float(Te_ped_eV)
        sol['ne_ped_val'] = float(ne_ped_val)
        sol['psi_ped'] = float(psi_ped)
        sol['pedestal_height'] = float(pedestal_height)
        sol['pedestal_width'] = float(pedestal_width)
        sol['model'] = res['model']
        sol['implementation'] = res['implementation']
        sol['diagnostics'] = res['diagnostics']
        np.save(f'{out_dir_p}/ne_and_Te_iter_{ESCAPE_iter}.npy', sol, allow_pickle=True)

        if ESCAPE_iter > 0:
            if abs(eped_tol) < ESCAPE_tol_max:
                break

    return state