# paramfile='../paramfiles/paramfile_planck.yaml'
# paramfile='../paramfiles/paramfile_wmap.yaml'
paramfile='../paramfiles/paramfile_planck_wmap.yaml'

echo "Running pipeline with paramfile: ${paramfile}"

echo "------------------------------------------------------------"
echo "|           PREPARING METADATA AND SIMULATIONS             |"
echo "------------------------------------------------------------"

echo "Pre-processing data..."
echo "-------------------"
python pre_processer.py --globals ${paramfile} --sims
python mask_handler.py --globals ${paramfile} --plots

echo "Pre-processing external data..."
echo "-------------------"
echo "Planck data"
python pre_processer_ext.py --globals ${paramfile} --planck --data --plots
echo "Planck sims"
python pre_processer_ext.py --globals ${paramfile} --planck --sims
echo "Planck noise"
python pre_processer_ext.py --globals ${paramfile} --planck --noise

echo "-------------------"
echo "WMAP data"
python pre_processer_ext.py --globals ${paramfile} --wmap --data --plots
echo "WMAP sims"
python pre_processer_ext.py --globals ${paramfile} --wmap --sims
echo "WMAP noise"
python pre_processer_ext.py --globals ${paramfile} --wmap --noise
