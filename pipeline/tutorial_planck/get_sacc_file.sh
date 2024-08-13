#!/usr/bin/env bash

paramfile='../paramfiles/paramfile_planck.yaml'

echo "Running pipeline with paramfile: ${paramfile}"

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Package data into SACC file
python create_sacc_file.py --globals ${paramfile} --data

# Package sims into SACC files (optional)
srun -n 10 -c 24 python create_sacc_file.py --globals ${paramfile} --sims

# Plot SACC contents (includes sims if present)
python sacc_plotter.py --globals ${paramfile} --verbose