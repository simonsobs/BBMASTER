#!/usr/bin/env bash

paramfile='../paramfiles/paramfile_planck.yaml'

echo "Running pipeline with paramfile: ${paramfile}"

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Co-add signal and noise bundle simulations
#srun -n 10 -c 24 
python simulations/coadd_simulated_maps.py --globals ${paramfile} --verbose

# Compute cross-bundle spectra from sims
srun -n 10 -c 24 python compute_sims_pseudo_cells.py --globals ${paramfile} --verbose

# Coadd simulation cross-bundle spectra
python coadd_sims_pseudo_cells.py --globals ${paramfile} --verbose

# Compute covariance matrix
python compute_covariance_from_sims.py --globals ${paramfile}