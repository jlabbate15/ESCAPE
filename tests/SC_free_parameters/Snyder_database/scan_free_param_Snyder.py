#!/usr/bin/env python3
"""
Free-parameter scan of the Saarelma-Connor pedestal model over the Snyder
HighPerfHMode equilibrium database (same inputs as
epednn_multieq_profilesloop_Snyder.py).

For each matched gEQDSK / OMFITnc pair, builds one base model and scans the
free-parameter space with update_free_params + solve_coupled_nondim
(no profiles_loop_solve / EPEDNN).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent  # saarelma-conner-ped/
sys.path.insert(0, str(ROOT))

tokamaker_python_path = os.getenv('OFT_ROOTPATH')
if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path, 'python'))

from ESCAPE.helpers.load_equil import initialize_inputs
from src.solver_nondim import saarelma_connor_nondim

# ── Scan grid ────────────────────────────────────────────────────────────────
N = 3
alpha_crits = np.logspace(-1, 1, N)
C_KBMs = np.logspace(-1, 1, N)
De_chie_etgs = np.logspace(-1, 1, N)
nFC_x0s = np.logspace(14, 17, N)
ncx_x0_ratios = np.logspace(0.1, 1.25, N)
psi_val = 0.85
x_res = 20
verbose = False

# Static parameters (same defaults as epednn_multieq_profilesloop_Snyder /
# profiles_loop_solve for this database)
P_tot_e = 5e6  # W
species = 'D'
Z_i = 1

# Equil num: integer count, or 'all' for every matched OMFITnc profile
equil_num = 100

# ── Snyder HighPerfHMode inputs (same as epednn_multieq_profilesloop_Snyder) ─
geqdsk_dir = Path('/mnt/homes_global/jal2351/software/sc_inputs/gHighPerfHMode')
pfile_dir = Path('/mnt/homes_global/jal2351/software/sc_inputs/OMFITnc_HighPerfHMode')
kprof_loc = 'OMFITnc'

scan_success_dir = HERE / f'scan_results_Snyder_N{N}'
success_fp = Path('success_Snyder.txt')
failure_fp = Path('failure_Snyder.txt')
error_messages_fp = Path('error_messages_Snyder.txt')

SOLVE_KW = dict(
    x_res=x_res,
    fe_degree=2,
    initial_guess='tanh',
    ne_inner_bc='neumann',
    linear_solver='lu',
    nCX_ic='scale nFC',
    kbm_treatment='picard',
    kbm_gate_eps=0.1,
    picard_gate_mode='average',
    picard_max_it=50,
    picard_rtol=1e-8,
    picard_relax=1.0,
    bc_origin='p-file',
    verbose=verbose,
)


def main():
    t0 = time.perf_counter()

    if equil_num == 'all':
        n_equil = len(list(pfile_dir.glob('*.cdf')))
    else:
        n_equil = int(equil_num)
    print(f'Number of equilibria requested: {n_equil}')
    equilibria = initialize_inputs(
        n_equil, geqdsk_dir, pfile_dir, p_filetype='OMFITnc',
    )
    print(f'Scanning {len(equilibria)} equilibria × '
          f'{len(alpha_crits)*len(C_KBMs)*len(De_chie_etgs)*len(nFC_x0s)*len(ncx_x0_ratios)} '
          f'free-parameter combinations.')

    scan_success_dir.mkdir(parents=True, exist_ok=True)

    n_ok_tot = n_fail_tot = 0
    n_combos = (len(alpha_crits) * len(C_KBMs) * len(De_chie_etgs)
                * len(nFC_x0s) * len(ncx_x0_ratios))

    for eq_idx, (mhd_fp, kprof_fp) in enumerate(equilibria):
        eq_tag = Path(mhd_fp).name[1:]  # e.g. 125729.03589
        eq_dir = scan_success_dir / eq_tag
        eq_dir.mkdir(parents=True, exist_ok=True)

        eq_success_fp = eq_dir / success_fp
        eq_failure_fp = eq_dir / failure_fp
        eq_error_fp = eq_dir / error_messages_fp

        for old in eq_dir.glob('ne_*.npy'):
            old.unlink()
        for fp in (eq_success_fp, eq_failure_fp, eq_error_fp):
            fp.unlink(missing_ok=True)
        with open(eq_success_fp, 'w') as f:
            f.write('eq_tag, alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, '
                    'psi_N_inner, L2_error\n')
        with open(eq_failure_fp, 'w') as f:
            f.write('eq_tag, alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, '
                    'psi_N_inner\n')
        with open(eq_error_fp, 'w') as f:
            f.write('eq_tag, alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, '
                    'psi_N_inner, message\n')

        print(f'\n=== Equilibrium {eq_idx + 1}/{len(equilibria)}: {eq_tag} ===')
        print(f'  g: {mhd_fp}')
        print(f'  p: {kprof_fp}')
        print('  Building Saarelma-Connor base model...')
        base_model = saarelma_connor_nondim(
            P_tot_e=P_tot_e,
            species=species,
            Z_i=Z_i,
            alpha_crit=round(float(alpha_crits[0]), 3),
            C_KBM=round(float(C_KBMs[0]), 3),
            De_chie_etg=round(float(De_chie_etgs[0]), 3),
            nFC_x0=round(float(nFC_x0s[0]), 3),
            ncx_x0_ratio=round(float(ncx_x0_ratios[0]), 3),
            psi_N_inner_boundary=psi_val,
            mhd_fp=mhd_fp,
            kprof_fp=kprof_fp,
            kprof_loc=kprof_loc,
            verbose=verbose,
        )
        print('  Base model ready — starting free-parameter scan.')

        n_ok = n_fail = 0
        i_combo = 0
        for alpha_crit in alpha_crits:
            for C_KBM in C_KBMs:
                for De_chie_etg in De_chie_etgs:
                    for nFC_x0 in nFC_x0s:
                        for ncx_x0_ratio in ncx_x0_ratios:
                            i_combo += 1
                            ac = round(float(alpha_crit), 3)
                            ck = round(float(C_KBM), 3)
                            de = round(float(De_chie_etg), 3)
                            nf = round(float(nFC_x0), 3)
                            nc = round(float(ncx_x0_ratio), 3)
                            try:
                                base_model.update_free_params(
                                    alpha_crit=ac,
                                    C_KBM=ck,
                                    De_chie_etg=de,
                                    nFC_x0=nf,
                                    ncx_x0_ratio=nc,
                                    psi_N_inner_boundary=psi_val,
                                )
                                x_sol, ne_sol, nFC_sol, nCX_sol, T_e_pres, psi_N_pres = (
                                    base_model.solve_coupled_nondim(**SOLVE_KW)
                                )
                                sol = {
                                    'x': x_sol, 'y': ne_sol,
                                    'nFC': nFC_sol, 'nCX': nCX_sol,
                                    'T_e_pres': T_e_pres, 'psi_N_pres': psi_N_pres,
                                }
                            except Exception as e:
                                n_fail += 1
                                with open(eq_failure_fp, 'a') as f:
                                    f.write(
                                        f'{eq_tag}, {ac}, {ck}, {de}, {nf}, {nc}, '
                                        f'{psi_val:.4f}\n'
                                    )
                                with open(eq_error_fp, 'a') as f:
                                    f.write(
                                        f'{eq_tag}, {ac}, {ck}, {de}, {nf}, {nc}, '
                                        f'{psi_val:.4f}, {e}\n'
                                    )
                            else:
                                n_ok += 1
                                ne_ref = interp1d(
                                    base_model.x_init, base_model.n_e_pres,
                                    kind='linear', bounds_error=False,
                                    fill_value='extrapolate',
                                )(x_sol)
                                L2_error = (
                                    np.linalg.norm(ne_sol - ne_ref)
                                    / np.linalg.norm(ne_ref)
                                )
                                with open(eq_success_fp, 'a') as f:
                                    f.write(
                                        f'{eq_tag}, {ac}, {ck}, {de}, {nf}, {nc}, '
                                        f'{psi_val:.4f}, {L2_error}\n'
                                    )
                                np.save(
                                    eq_dir / (
                                        f'ne_a{ac}_C{ck}_D{de}_n{nf}_nc{nc}'
                                        f'_b{psi_val:.4f}'
                                    ),
                                    sol,
                                    allow_pickle=True,
                                )

                            if i_combo % 20 == 0 or i_combo == n_combos:
                                print(f'  {i_combo}/{n_combos}  '
                                      f'(ok={n_ok}, fail={n_fail})')

        print(f'  Done {eq_tag}: ok={n_ok}, fail={n_fail}  → {eq_dir}')
        n_ok_tot += n_ok
        n_fail_tot += n_fail
        del base_model

    elapsed = time.perf_counter() - t0
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'\nAll equilibria done. Saved under {scan_success_dir}')
    print(f'Total successes={n_ok_tot}, failures={n_fail_tot}')
    print(f'Total runtime: {elapsed:.1f} s '
          f'({int(hours):d}h {int(minutes):02d}m {seconds:05.2f}s)')


if __name__ == '__main__':
    main()
