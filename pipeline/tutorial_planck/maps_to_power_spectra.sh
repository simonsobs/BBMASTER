#!/usr/bin/env bash

paramfile='../paramfiles/paramfile_planck.yaml'

echo "Running pipeline with paramfile: ${paramfile}"

# Compute multipole binning (with constant bin width 10) and analysis mask
python misc/get_binning.py --globals ${paramfile} --deltal 10
python get_analysis_mask.py --globals ${paramfile}

# Load Planck maps and beams and save them to disk
python pre_processer_ext.py --data --planck --globals ${paramfile}

# Compute Planck per-bundle power spectra
python compute_pseudo_cells.py --globals ${paramfile}

# Coadd Planck per-bundle power spectra
python coadd_pseudo_cells.py --globals ${paramfile}
