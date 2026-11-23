#!/bin/bash
# Submit one scan_SCfp_ESCAPE.slurm job per parameter combination.
# Job name and .out/.err files match the output_dir built in scan_SCfp_ESCAPE.py.
set -euo pipefail
cd "$(dirname "$0")"

# sc_models=('1D' '3D')
# sc_implementations=('firedrake' 'scipy')
# ne_grad_bc_locs=('inner' 'outer')
sc_models=('1D')
sc_implementations=('scipy')
ne_grad_bc_locs=('inner')
run_suffix='10272026_samulidamp_complete'
picard_relax="${PICARD_RELAX:-1:1.0;11:0.5;31:0.2}"
eq6_form="${EQ6_FORM:-complete}"
eq7_form="${EQ7_FORM:-complete}"
impurity_dilution="${IMPURITY_DILUTION:-carbon}"
d_tg_treatment="${D_TG_TREATMENT:-picard}"
picard_gate_mode="${PICARD_GATE_MODE:-steep_grad}"  # 'average' or 'steep_grad'
run_suffix="${RUN_SUFFIX:-$run_suffix}"

for sc_model in "${sc_models[@]}"; do
  for sc_implementation in "${sc_implementations[@]}"; do
    for ne_grad_bc_loc in "${ne_grad_bc_locs[@]}"; do
      # Must match output_dir in scan_SCfp_ESCAPE.py
      name="DIIIDDunsmore_ESCAPE_${sc_model}${sc_implementation}${ne_grad_bc_loc}_${run_suffix}"
      sbatch \
        --job-name="${name}" \
        --output="${name}_%j.out" \
        --error="${name}_%j.err" \
        --export=ALL,SC_MODEL="${sc_model}",SC_IMPLEMENTATION="${sc_implementation}",NE_GRAD_BC_LOC="${ne_grad_bc_loc}",RUN_SUFFIX="${run_suffix}",PICARD_RELAX="${picard_relax}",EQ6_FORM="${eq6_form}",EQ7_FORM="${eq7_form}",IMPURITY_DILUTION="${impurity_dilution}",D_TG_TREATMENT="${d_tg_treatment}",PICARD_GATE_MODE="${picard_gate_mode}" \
        scan_SCfp_ESCAPE.slurm
    done
  done
done
