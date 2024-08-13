#!/usr/bin/env bash

paramfile='../paramfiles/paramfile_planck.yaml'

echo "Running pipeline with paramfile: ${paramfile}"

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Preprocessing (need not be run if already done)
python misc/get_binning.py --globals ${paramfile} --deltal 10
python get_analysis_mask.py --globals ${paramfile}

# Compute mask-based mode coupling matrix
python get_mode_coupling.py --globals ${paramfile}

# Generate transfer function simulations
srun -n 10 -c 24 python simulations/generate_tf_estimation_sims.py --globals ${paramfile} --verbose --no-plots

# Compute transfer function
srun -n 10 -c 24 python transfer/compute_pseudo_cells_tf_estimation.py --globals ${paramfile}
python transfer/compute_transfer_function.py --globals ${paramfile}

# Compute bandpower window function
python get_full_couplings.py --globals ${paramfile}