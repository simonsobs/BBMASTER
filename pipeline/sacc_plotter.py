import matplotlib.pyplot as plt
from soopercool import BBmeta
import argparse
from itertools import product
import sacc
import numpy as np
import healpy as hp
import pymaster as nmt
import os


def load_bpwins(coupling_file):
    """
    """
    bp_win = np.load(coupling_file)
    return bp_win["bp_win"]


def binned_theory_from_unbinned(clth, bpw_mat):
    """
    """
    modes = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    clth_vec = np.concatenate(
        [clth[mode] for mode in modes]
    ).reshape(len(modes), -1)
    clth_binned = np.einsum("ijkl,kl->ij", bpw_mat, clth_vec)
    cl_out = {}
    for i, m in enumerate(modes):
        cl_out[m] = clth_binned[i]
    return cl_out


def multipole_min_from_tf(tf_file, field_pairs, snr_cut=3.):
    """
    """
    tf = np.load(tf_file)
    idx_bad_tf = {}
    for fp in field_pairs:
        name = f"{fp}_to_{fp}"
        snr = tf[name] / tf[f"{name}_std"]
        idx = np.where(snr < snr_cut)[0]
        idx_bad_tf[fp] = idx.max() if idx.size > 0 else 0

    return idx_bad_tf


def plot_spectrum(lb, cb, cb_err, title, ylabel, xlim,
                  cb_data=None, cb_data_err=None, add_theory=False,
                  lth=None, clth=None, cbth=None, save_file=None):
    """
    """
    plt.figure(figsize=(8, 6))
    grid = plt.GridSpec(4, 1, wspace=0, hspace=0)
    if add_theory:
        main = plt.subplot(grid[:-1])
        sub = plt.subplot(grid[-1])
        sub.set_xlabel(r"$\ell$")
        sub.set_ylabel(r"$\Delta C_\ell / (\sigma / \sqrt{N_\mathrm{sims}})$")
    else:
        main = plt.subplot(grid[:])
        main.set_xlabel(r"$\ell$")

    main.set_ylabel(r"$\ell (\ell + 1) C_\ell^{%s} / 2\pi$" % ylabel)
    main.set_title(title)

    fac = lb * (lb + 1) / (2 * np.pi)
    offset = 0 if cb_data is None else 1

    if add_theory:
        fac_th = lth * (lth + 1) / (2 * np.pi)
        main.plot(lth, fac_th * clth, c="darkgray", ls="-.", lw=2.6,
                  label="theory")

    main.errorbar(
        lb-offset, fac * cb, yerr=fac * cb_err, marker="o", ls="None",
        markerfacecolor="white", markeredgecolor="navy", label="sims",
        elinewidth=1.75, ecolor="navy", markeredgewidth=1.75
    )

    if cb_data is not None:
        main.errorbar(
            lb+offset, fac * cb_data, yerr=fac * cb_data_err, marker="o",
            ls="None", markerfacecolor="white", markeredgecolor="darkorange",
            elinewidth=1.75, ecolor="darkorange", markeredgewidth=1.75,
            label="data"
        )

    main.legend(fontsize=13)
    main.set_xlim(*xlim)

    if add_theory:
        sub.axhspan(-3, 3, color="gray", alpha=0.1)
        sub.axhspan(-2, 2, color="gray", alpha=0.4)
        sub.axhspan(-1, 1, color="gray", alpha=0.8)

        sub.plot(
            lb-offset, (cb - cbth) / cb_err, marker="o", ls="None",
            markerfacecolor="white", markeredgecolor="navy",
            markeredgewidth=1.75
        )
        if cb_data is not None:
            sub.plot(
                lb+offset, (cb_data - cbth) / cb_data_err,
                marker="o", ls="None",
                markerfacecolor="white", markeredgecolor="darkorange",
                markeredgewidth=1.75
            )

        sub.set_xlim(*xlim)
        sub.set_ylim(-4.5, 4.5)

    if save_file:
        plt.savefig(save_file, bbox_inches="tight")
    else:
        plt.tight_layout()
        plt.show()
    plt.close()


def main(args):
    """
    This script will read the spectra and covariance
    stored in the `sacc` files and plot the power
    spectra.
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose

    out_dir = meta.output_directory
    sacc_dir = f"{out_dir}/saccs"
    coupling_dir = f"{out_dir}/couplings"

    plot_dir = f"{out_dir}/plots/sacc_spectra"
    BBmeta.make_dir(plot_dir)

    binning = np.load(meta.binning_file)
    nmt_binning = nmt.NmtBin.from_edges(binning["bin_low"],
                                        binning["bin_high"] + 1)
    lb = nmt_binning.get_effective_ells()
    lmax = 3*meta.nside - 1

    field_pairs = [m1+m2 for m1, m2 in product("TEB", repeat=2)]
    ps_names = meta.get_ps_names_list(type="all", coadd=True)

    types = {"T": "0", "E": "e", "B": "b"}

    Nsims = meta.covariance["cov_num_sims"]

    # Transfer function
    idx_bad_tf = {}
    bpw_mats = {}

    transfer_dir = meta.transfer_settings["transfer_directory"]
    for ftag1, ftag2 in meta.get_independent_filtering_pairs():
        idx = multipole_min_from_tf(
            f"{transfer_dir}/transfer_function_{ftag1}_x_{ftag2}.npz",
            field_pairs=field_pairs,
            snr_cut=3
        )
        idx_bad_tf[ftag1, ftag2] = idx

    for ms1, ms2 in ps_names:
        bpw_file = f"couplings_{ms1}_{ms2}.npz"
        bpw_mats[ms1, ms2] = load_bpwins(f"{coupling_dir}/{bpw_file}")

    plot_sims = {
        (ms1, ms2, fp): {
            "x": None,
            "y": [],
            "err": None,
            "x_th": None,
            "y_th": None,
            "th_binned": None,
            "title": None,
            "ylabel": None
        } for ms1, ms2 in ps_names
        for fp in field_pairs
    }

    fiducial_cmb = meta.covariance["fiducial_cmb"]
    fiducial_dust = meta.covariance["fiducial_dust"]
    fiducial_synch = meta.covariance["fiducial_synch"]

    for ms1, ms2 in ps_names:
        nu1 = meta.freq_tag_from_map_set(ms1)
        nu2 = meta.freq_tag_from_map_set(ms2)

        clth = {}

        for i, fp in enumerate(["TT", "EE", "BB", "TE"]):
            cmb_cl = hp.read_cl(fiducial_cmb)[i, :lmax+1]
            dust_cl = hp.read_cl(
                fiducial_dust.format(nu1=nu1, nu2=nu2)
            )[i, :lmax+1]
            synch_cl = hp.read_cl(
                fiducial_synch.format(nu1=nu1, nu2=nu2)
            )[i, :lmax+1]
            clth[fp] = (cmb_cl + dust_cl + synch_cl)
        clth["EB"] = np.zeros(lmax+1)
        clth["TB"] = np.zeros(lmax+1)
        clth["l"] = np.arange(lmax+1)

        cl_th = {}
        for fp in field_pairs:
            if fp in clth:
                cl_th[fp] = clth[fp]
            else:
                # reverse the order of the two fields
                cl_th[fp] = clth[fp[::-1]]

        clth_binned = binned_theory_from_unbinned(cl_th, bpw_mats[(ms1, ms2)])

        ftag1 = meta.filtering_tag_from_map_set(ms1)
        ftag2 = meta.filtering_tag_from_map_set(ms2)

        for fp in field_pairs:
            mask_th = (clth["l"] <= 2 * meta.nside - 1)
            x_th, y_th = clth["l"][mask_th], cl_th[fp][mask_th]

            idx_bad = idx_bad_tf[ftag1, ftag2][fp]
            mask = ((np.arange(len(lb)) > idx_bad) &
                    (lb <= 2 * meta.nside - 1))
            th_binned = clth_binned[fp][mask]

            plot_sims[ms1, ms2, fp]["x_th"] = x_th
            plot_sims[ms1, ms2, fp]["y_th"] = y_th
            plot_sims[ms1, ms2, fp]["th_binned"] = th_binned

    # Load data
    plot_data = {
        (ms1, ms2, fp): {
            "y": None,
        } for ms1, ms2 in ps_names
        for fp in field_pairs
    }
    fname_data = f"{sacc_dir}/cl_and_cov_sacc.fits"

    if not os.path.isfile(fname_data):
        print("No data sacc to print. Skipping...")
    else:
        s = sacc.Sacc.load_fits(fname_data)

        for ms1, ms2 in ps_names:
            ftag1 = meta.filtering_tag_from_map_set(ms1)
            ftag2 = meta.filtering_tag_from_map_set(ms2)

            for fp in field_pairs:
                f1, f2 = fp

                ell, cl, cov = s.get_ell_cl(
                    f"cl_{types[f1]}{types[f2]}",
                    ms1,
                    ms2,
                    return_cov=True)

                idx_bad = idx_bad_tf[ftag1, ftag2][fp]
                mask = (
                    (np.arange(len(ell)) > idx_bad) &
                    (ell <= 2 * meta.nside - 1))
                x, y, err = ell, cl, np.sqrt(cov.diagonal())
                x, y, err = x[mask], y[mask], err[mask]

                plot_data[ms1, ms2, fp]["x"] = x
                plot_data[ms1, ms2, fp]["y"] = y
                plot_data[ms1, ms2, fp]["err"] = err

    # Load simulations
    for id_sim in range(Nsims):
        if verbose:
            print(f"# {id_sim} | {ms1} x {ms2}")
        s = sacc.Sacc.load_fits(
            f"{sacc_dir}/cl_and_cov_sacc_{id_sim:04d}.fits"
        )

        for ms1, ms2 in ps_names:
            ftag1 = meta.filtering_tag_from_map_set(ms1)
            ftag2 = meta.filtering_tag_from_map_set(ms2)

            for fp in field_pairs:
                f1, f2 = fp

                ell, cl, cov = s.get_ell_cl(
                    f"cl_{types[f1]}{types[f2]}",
                    ms1,
                    ms2,
                    return_cov=True)

                idx_bad = idx_bad_tf[ftag1, ftag2][fp]
                mask = (
                    (np.arange(len(ell)) > idx_bad) &
                    (ell <= 2 * meta.nside - 1))
                x, y, err = ell, cl, np.sqrt(cov.diagonal())
                x, y, err = x[mask], y[mask], err[mask]

                plot_sims[ms1, ms2, fp]["x"] = x
                plot_sims[ms1, ms2, fp]["y"].append(y)
                plot_sims[ms1, ms2, fp]["err"] = err
                plot_sims[ms1, ms2, fp]["title"] = f"{ms1} x {ms2} - {fp}"
                plot_sims[ms1, ms2, fp]["ylabel"] = fp

    for ms1, ms2 in ps_names:
        for fp in field_pairs:

            plot_sims[ms1, ms2, fp]["y"] = np.mean(
                plot_sims[ms1, ms2, fp]["y"], axis=0
            )
            plot_sims[ms1, ms2, fp]["err"] /= np.sqrt(Nsims)
            plot_name = f"plot_cells_{ms1}_{ms2}_{fp}.pdf"

            plot_spectrum(
                plot_sims[ms1, ms2, fp]["x"],
                plot_sims[ms1, ms2, fp]["y"],
                plot_sims[ms1, ms2, fp]["err"],
                cb_data=plot_data[ms1, ms2, fp]["y"],
                cb_data_err=plot_data[ms1, ms2, fp]["err"],
                title=plot_sims[ms1, ms2, fp]["title"],
                ylabel=plot_sims[ms1, ms2, fp]["ylabel"],
                xlim=(30, 2 * meta.nside - 1),
                add_theory=True,
                lth=plot_sims[ms1, ms2, fp]["x_th"],
                clth=plot_sims[ms1, ms2, fp]["y_th"],
                cbth=plot_sims[ms1, ms2, fp]["th_binned"],
                save_file=f"{plot_dir}/{plot_name}"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sacc plotter')
    parser.add_argument("--globals", type=str,
                        help="Path to the global configuration file")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    main(args)
