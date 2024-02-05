import argparse
import healpy as hp
from soopercool.utils import get_noise_cls, beam_gaussian, generate_noise_map
import numpy as np
import os
import soopercool.SO_Noise_Calculator_Public_v3_1_2 as noise_calc
from soopercool import BBmeta, utils
import warnings


def mocker(args):
    """
    Implement a very basic simulation routine
    to generate mock maps or a set of simulation
    to estimate the covariance (if --sims is set to True).

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from the command line.
    """
    meta = BBmeta(args.globals)

    map_dir = meta.map_directory
    beam_dir = meta.beam_directory

    os.makedirs(map_dir, exist_ok=True)
    os.makedirs(beam_dir, exist_ok=True)

    ps_th = meta.load_fiducial_cl(cl_type="cosmo")

    # Load binary mask
    binary_mask = meta.read_mask("binary")
    fsky = np.mean(binary_mask)
    lmax_sim = 3 * meta.nside - 1

    # Load noise curves
    meta.timer.start("Computing noise cls")
    noise_model = noise_calc.SOSatV3point1(sensitivity_mode='baseline')
    lth, nlth_dict = get_noise_cls(fsky, lmax_sim+1, is_beam_deconvolved=False)
    meta.timer.stop("Computing noise cls")

    # Load hitmap
    hitmap = meta.read_hitmap()

    # Load and save beams
    meta.timer.start("Generating beams")

    beam_arcmin = {freq_band: beam_arcmin
                   for freq_band, beam_arcmin in zip(noise_model.get_bands(),
                                                     noise_model.get_beams())}
    beams = {}
    for map_set in meta.map_sets_list:
        freq_tag = meta.freq_tag_from_map_set(map_set)
        beams[map_set] = beam_gaussian(lth, beam_arcmin[freq_tag])

        # Save beams
        file_root = meta.file_root_from_map_set(map_set)
        if not os.path.exists(file_root):
            np.savetxt(f"{beam_dir}/beam_{file_root}.dat",
                       np.transpose([lth, beams[map_set]]))
    meta.timer.stop("Generating beams")

    hp_ordering = ["TT", "TE", "TB", "EE", "EB", "BB"]

    Nsims = meta.num_sims if args.sims else 1

    for id_sim in range(Nsims):
        alms_T, alms_E, alms_B = hp.synalm([ps_th[k] for k in hp_ordering],
                                           lmax=lmax_sim)
        if meta.null_e_modes:
            cmb_map = hp.alm2map([alms_T, alms_E*0, alms_B],
                                 meta.nside, lmax=lmax_sim)
        else:
            cmb_map = hp.alm2map([alms_T, alms_E, alms_B],
                                 meta.nside, lmax=lmax_sim)

        for map_set in meta.map_sets_list:

            meta.timer.start(f"Generate map set {map_set} split maps")
            freq_tag = meta.freq_tag_from_map_set(map_set)
            cmb_map_beamed = hp.sphtfunc.smoothing(
                cmb_map, fwhm=np.deg2rad(beam_arcmin[freq_tag] / 60))

            n_splits = meta.n_splits_from_map_set(map_set)
            file_root = meta.file_root_from_map_set(map_set)
            for id_split in range(n_splits):
                noise_map = generate_noise_map(
                    nlth_dict["T"][freq_tag],
                    nlth_dict["P"][freq_tag],
                    hitmap,
                    n_splits,
                    is_anisotropic=meta.anisotropic_noise
                )
                split_map = cmb_map_beamed + noise_map

                split_map *= binary_mask

                map_file_name = meta.get_map_filename(
                    map_set, id_split,
                    id_sim if Nsims > 1 else None
                )
                hp.write_map(
                    map_file_name,
                    split_map,
                    overwrite=True,
                    dtype=np.float32
                )

                if args.plots:
                    if Nsims == 1:
                        plot_dir = meta.plot_dir_from_output_dir(
                            meta.map_directory_rel
                        )
                        utils.plot_map(split_map,
                                       f"{plot_dir}/map_{map_set}__{id_split}",
                                       title=map_set,
                                       TQU=True)
            meta.timer.stop(f"Generate map set {map_set} split maps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simplistic simulator')
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--sims", action="store_true",
                        help="Generate a set of sims if True.")
    parser.add_argument("--plots", action="store_true",
                        help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.sims and args.plots:
        warnings.warn("Both --sims and --plot are set to True. "
                      "Too many plots will be generated. "
                      "Set --plot to False")
        args.plots = False

    mocker(args)
