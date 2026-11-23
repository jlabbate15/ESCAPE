#!/bin/bash
# Submit one scan_SCfp_ESCAPE_SPARC.slurm job per parameter combination.
# Job name and .out/.err files match the output_dir built in scan_SCfp_ESCAPE_SPARC.py.
set -euo pipefail
cd "$(dirname "$0")"

# sc_models=('1D' '3D')
# sc_implementations=('firedrake' 'scipy')
# ne_grad_bc_locs=('inner' 'outer')
sc_models=('1D')
sc_implementations=('scipy')
ne_grad_bc_locs=('inner')
run_suffix='biggerscan_corner'
picard_relax="${PICARD_RELAX:-1:1.0;11:0.5;31:0.2}"
eq6_form="${EQ6_FORM:-complete}"
eq7_form="${EQ7_FORM:-complete}"
impurity_dilution="${IMPURITY_DILUTION:-None}"
d_tg_treatment="${D_TG_TREATMENT:-picard}"
picard_rel_tol="${PICARD_REL_TOL:-1e-2}"  # SC Picard loop: converged when |dne|_rel < this
picard_gate_mode="${PICARD_GATE_MODE:-steep_grad}"  # 'average' or 'steep_grad'
bvp_tol="${BVP_TOL:-1e-5}"                # solve_bvp tolerance ('inner', scipy only)
bvp_max_nodes="${BVP_MAX_NODES:-20000}"    # solve_bvp max mesh nodes ('inner', scipy only)
run_suffix="${RUN_SUFFIX:-$run_suffix}"
profile_source="${PROFILE_SOURCE:-cgyro}"  # 'transp' or 'cgyro'
# First-iteration n_e pedestal top (helpers/ped_width_proxy.ped_top_width): 'tanh', 'grad', 'lin',
# 'tanh+grad', 'tanh+lin', 'grad+lin', 'all' (psi_N = 1 - width), or 'corner' (two-line fit corner)
ped_width_method="${PED_WIDTH_METHOD:-tanh+grad}"

for sc_model in "${sc_models[@]}"; do
  for sc_implementation in "${sc_implementations[@]}"; do
    for ne_grad_bc_loc in "${ne_grad_bc_locs[@]}"; do
      # Must match output_dir in scan_SCfp_ESCAPE_SPARC.py
      name="SPARC_ESCAPE_${sc_model}${sc_implementation}${ne_grad_bc_loc}${run_suffix:+_${run_suffix}}"
      sbatch \
        --job-name="${name}" \
        --output="${name}_%j.out" \
        --error="${name}_%j.err" \
        --export=ALL,SC_MODEL="${sc_model}",SC_IMPLEMENTATION="${sc_implementation}",NE_GRAD_BC_LOC="${ne_grad_bc_loc}",RUN_SUFFIX="${run_suffix}",PICARD_RELAX="${picard_relax}",PICARD_REL_TOL="${picard_rel_tol}",EQ6_FORM="${eq6_form}",EQ7_FORM="${eq7_form}",IMPURITY_DILUTION="${impurity_dilution}",D_TG_TREATMENT="${d_tg_treatment}",PICARD_GATE_MODE="${picard_gate_mode}",BVP_TOL="${bvp_tol}",BVP_MAX_NODES="${bvp_max_nodes}",PROFILE_SOURCE="${profile_source}",PED_WIDTH_METHOD="${ped_width_method}" \
        scan_SCfp_ESCAPE_SPARC.slurm
    done
  done
done
