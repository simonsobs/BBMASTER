import argparse
import numpy as np
from soopercool import BBmeta
import matplotlib.pyplot as plt
from soopercool import utils


def read_transfer(transfer_file):
    """
    """
    tfd = np.load(transfer_file)

    tf_dict = {}
    spin_pairs = {
        "spin0xspin0": ["TT"],
        "spin0xspin2": ["TE", "TB"],
        "spin2xspin2": ["EE", "EB", "BE", "BB"]
    }
    for spin_pair, fields in spin_pairs.items():
        for id1, f1 in enumerate(fields):
            for id2, f2 in enumerate(fields):
                tf_dict[f1, f2] = tfd[f"tf_{spin_pair}"][id1, id2]
                tf_dict[f1, f2, "std"] = tfd[f"tf_std_{spin_pair}"][id1, id2]
    return tf_dict


def transfer_validator(args):
    """
    Compare decoupled spectra (including deconvolved form the TF)
    to the unfiltered spectra (deconvolved from mode coupling only)
    Compare coupled spectra (with TF + MCM)
    """
    meta = BBmeta(args.globals)

    fields = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

    nmt_binning = meta.read_nmt_binning()
    nl = nmt_binning.lmax+1
    lb = nmt_binning.get_effective_ells()

    plot_dir = meta.plot_dir_from_output_dir(meta.coupling_directory)

    filtering_pairs = meta.get_independent_filtering_pairs()

    # First plot the transfer functions
    for ftag1, ftag2 in filtering_pairs:
        tf_dict = np.load(f"{meta.coupling_directory}/transfer_function_{ftag1}x{ftag2}.npz") # noqa

        utils.plot_transfer_function(
            lb, tf_dict, meta.lmin, meta.lmax,
            fields,
            file_name=f"{plot_dir}/transfer_{ftag1}x{ftag2}.pdf"
        )

    # Then we read the decoupled spectra
    # both for the filtered and unfiltered cases
    cl_dir = meta.cell_transfer_directory
    cross_fields = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    val_types = ["tf_val", "cosmo"]
    ftypes = ["filtered", "unfiltered"]

    cls_dict = {
        (val_type, ftype, ftag1, ftag2, k): []
        for val_type in val_types
        for k in cross_fields
        for ftype in ftypes
        for ftag1, ftag2 in filtering_pairs
    }

    for val_type in ["tf_val", "cosmo"]:
        for ftype in ftypes:
            for ftag1, ftag2 in filtering_pairs:
                for id_sim in range(meta.tf_est_num_sims):
                    cls = np.load(f"{cl_dir}/pcls_{val_type}_{ftag1}x{ftag2}_{id_sim:04d}_{ftype}.npz")  # noqa
                    for k in cross_fields:
                        cls_dict[val_type, ftype, ftag1, ftag2, k] += [cls[k]]

    # Compute mean and std
    cls_mean_dict = {
        (val_type, ftype, ftag1, ftag2, k):
        np.mean(cls_dict[val_type, ftype, ftag1, ftag2, k], axis=0)
        for val_type in val_types
        for ftype in ftypes
        for k in cross_fields
        for ftag1, ftag2 in filtering_pairs
    }
    cls_std_dict = {
        (val_type, ftype, ftag1, ftag2, k):
        np.std(cls_dict[val_type, ftype, ftag1, ftag2, k], axis=0)
        for val_type in val_types
        for ftype in ftypes
        for k in cross_fields
        for ftag1, ftag2 in filtering_pairs
    }
    cls_theory = {
        val_type: meta.load_fiducial_cl(val_type)
        for val_type in val_types
    }

    cls_theory_binned = {}
    for ftag1, ftag2 in filtering_pairs:
        ftag_label = f"{ftag1}x{ftag2}"
        bp_win = np.load(
            f"{meta.coupling_directory}/couplings_{ftag_label}_unfiltered.npz"
        )
        bpw_mat = bp_win["bp_win"]
        for val_type in val_types:
            cls_vec = np.array([
                cls_theory[val_type][k][:nl] for k in cross_fields
            ])
            cls_vec_binned = np.einsum("ijkl,kl", bpw_mat, cls_vec)
            for i, fp in enumerate(cross_fields):
                cls_theory_binned[ftag1, ftag2, val_type, fp] = \
                    cls_vec_binned[i]

    for val_type in val_types:
        for ftag1, ftag2 in filtering_pairs:
            plt.figure(figsize=(16, 16))
            grid = plt.GridSpec(9, 3, hspace=0.3, wspace=0.3)
            for id1, f1 in enumerate("TEB"):
                for id2, f2 in enumerate("TEB"):
                    # Define subplots
                    main = plt.subplot(grid[3*id1:3*(id1+1)-1, id2])
                    sub = plt.subplot(grid[3*(id1+1)-1, id2])

                    spec = f2 + f1 if id1 > id2 else f1 + f2

                    # Plot theory
                    ell = cls_theory[val_type]["l"]
                    rescaling = 1 if val_type == "tf_val" \
                        else ell * (ell + 1) / (2*np.pi)
                    main.plot(ell, rescaling*cls_theory[val_type][spec],
                              color="k")

                    offset = 0.5
                    rescaling = 1 if val_type == "tf_val" \
                        else lb * (lb + 1) / (2*np.pi)
                    # Plot filtered & unfiltered (decoupled)
                    main.errorbar(
                        lb-offset,
                        rescaling*cls_mean_dict[val_type,
                                                "unfiltered",
                                                ftag1, ftag2,
                                                spec],
                        rescaling*cls_std_dict[val_type,
                                               "unfiltered",
                                               ftag1, ftag2,
                                               spec],
                        color="navy",
                        marker=".",
                        markerfacecolor="white",
                        label=r"Unfiltered decoupled $C_\ell$",
                        ls="None"
                    )
                    main.errorbar(
                        lb+offset, rescaling*cls_mean_dict[val_type,
                                                           "filtered",
                                                           ftag1, ftag2,
                                                           spec],
                        rescaling*cls_std_dict[val_type,
                                               "filtered",
                                               ftag1, ftag2,
                                               spec],
                        color="darkorange",
                        marker=".",
                        markerfacecolor="white",
                        label=r"Filtered decoupled $C_\ell$",
                        ls="None"
                    )

                    main.plot(lb,
                              rescaling*cls_theory_binned[ftag1, ftag2,
                                                          val_type, spec],
                              color="tab:red", ls="--", label="Theory")

                    if f1 == f2:
                        main.set_yscale("log")

                    # Plot residuals
                    residual_unfiltered = \
                        (cls_mean_dict[val_type, "unfiltered",
                                       ftag1, ftag2, spec] -
                         cls_theory_binned[ftag1, ftag2,
                                           val_type, spec]) / \
                        cls_std_dict[val_type, "unfiltered",
                                     ftag1, ftag2, spec]

                    residual_filtered = \
                        (cls_mean_dict[val_type, "filtered",
                                       ftag1, ftag2, spec] -
                         cls_theory_binned[ftag1, ftag2,
                                           val_type, spec]) / \
                        cls_std_dict[val_type, "filtered",
                                     ftag1, ftag2, spec]

                    sub.axhspan(-2, 2, color="gray", alpha=0.2)
                    sub.axhspan(-1, 1, color="gray", alpha=0.7)

                    sub.axhline(0, color="k")
                    sqrt_nsims = np.sqrt(meta.tf_est_num_sims)
                    sub.plot(lb-offset,
                             residual_unfiltered * sqrt_nsims,
                             color="navy", marker=".", markerfacecolor="white",
                             ls="None")
                    sub.plot(lb+offset,
                             residual_filtered * sqrt_nsims,
                             color="darkorange", marker=".",
                             markerfacecolor="white", ls="None")

                    # Multipole range
                    main.set_xlim(2, meta.lmax)
                    sub.set_xlim(*main.get_xlim())

                    # Suplot y range
                    sub.set_ylim((-5., 5.))

                    # Cosmetix
                    main.set_title(f1+f2, fontsize=14)
                    if spec == "TT":
                        main.legend(fontsize=13)
                    main.set_xticklabels([])
                    if id1 != 2:
                        sub.set_xticklabels([])
                    else:
                        sub.set_xlabel(r"$\ell$", fontsize=13)

                    if id2 == 0:
                        if isinstance(rescaling, float):
                            main.set_ylabel(r"$C_\ell$", fontsize=13)
                        else:
                            main.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$",
                                            fontsize=13)
                        sub.set_ylabel(r"$\Delta C_\ell / (\sigma/\sqrt{N_\mathrm{sims}})$",  # noqa
                                       fontsize=13)

            plt.savefig(f"{plot_dir}/decoupled_{val_type}_{ftag1}x{ftag2}.pdf",
                        bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer function')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')

    args = parser.parse_args()

    transfer_validator(args)
