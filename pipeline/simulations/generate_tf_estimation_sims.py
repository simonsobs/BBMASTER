import argparse
from soopercool import BBmeta, utils
from soopercool import mpi_utils as mpi
from soopercool import map_utils as mu
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose
    do_plots = not args.no_plots

    out_dir = meta.output_directory
    sim_dir = f"{out_dir}/tf_est_sims"
    BBmeta.make_dir(sim_dir)

    plot_dir = f"{out_dir}/plots/tf_est_sims"
    if do_plots:
        BBmeta.make_dir(plot_dir)

    lmax_sim = 3 * meta.nside - 1
    lth = np.arange(lmax_sim + 1)

    tf_settings = meta.transfer_settings
    cl_power_law_tf_est = utils.power_law_cl(
        lth, **tf_settings["power_law_pars_tf_est"]
    )
    np.savez(f"{sim_dir}/cl_power_law_tf_est.npz",
             ell=lth, **cl_power_law_tf_est)

    Nsims = tf_settings["tf_est_num_sims"]

    hp_ordering = ["TT", "TE", "TB", "EE", "EB", "BB"]

    if tf_settings["beams_list"] is not None:
        beams = {}
        for beam_label in tf_settings["beams_list"]:
            _, bl = meta.read_beam(beam_label)
            beams[beam_label] = bl
    else:
        beams = {None: None}

    mpi.init(True)

    for id_sim in mpi.taskrange(Nsims - 1):
        almsTEB = hp.synalm(
            [cl_power_law_tf_est[k] for k in hp_ordering],
            lmax=lmax_sim
        )

        for beam_label, bl in beams.items():
            if verbose:
                print(f"# {id_sim} | {beam_label}")

            suffix = ""
            if tf_settings["do_not_beam_est_sims"]:
                bl = None

            elif beam_label is not None:
                suffix = f"_{beam_label}"

            sims = {
                f"pure{f}": utils.generate_map_from_alms(
                    almsTEB * select[:, None],
                    meta.nside,
                    bl=bl
                )
                for f, select in zip("TEB", np.eye(3))
            }

            for f in "TEB":
                fname = f"pure{f}_power_law_tf_est_{id_sim:04d}{suffix}.fits"
                mu.write_map(f"{sim_dir}/{fname}",
                             sims[f"pure{f}"], dtype=np.float32,
                             convert_muK_to_K=True)

    if do_plots:
        ps_hp_order = ["TT", "EE", "BB", "TE", "EB", "TB"]
        for beam_label in beams:
            suffix = "" if beam_label is None else f"_{beam_label}"
            for f in "TEB":

                cls_dict = {fp: [] for fp in hp_ordering}

                for id_sim in range(Nsims):
                    fname = f"pure{f}_power_law_tf_est_{id_sim:04d}{suffix}"
                    alms = hp.map2alm(
                        mu.read_map(
                            f"{sim_dir}/{fname}.fits",
                            field=[0, 1, 2],
                            pix_type=meta.pix_type,
                            convert_K_to_muK=True),
                    )
                    cls = hp.alm2cl(alms)

                    for i, fp in enumerate(ps_hp_order):
                        cls_dict[fp] += [cls[i]]

                for fp in ps_hp_order:
                    cls_dict[fp] = np.mean(cls_dict[fp], axis=0)

                plt.figure(figsize=(10, 8))
                plt.xlabel(r"$\ell$", fontsize=15)
                plt.ylabel(r"$C_\ell$", fontsize=15)
                for fp in ps_hp_order:
                    plt.plot(cls_dict[fp], label=fp, lw=1.7)
                plt.yscale("symlog", linthresh=1e-6)
                plt.xlim(0, lmax_sim)
                plt.legend()
                plt.title(f"Power law pure{f} simulation")
                plt.savefig(f"{plot_dir}/power_law_pure{f}{suffix}.png",
                            bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pureT/E/B simulations \
                     for transfer function estimation")
    parser.add_argument("--globals", type=str,
                        help="Path to the yaml with global parameters")
    parser.add_argument("--no-plots", action="store_true",
                        help="Pass to generate plots")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)
