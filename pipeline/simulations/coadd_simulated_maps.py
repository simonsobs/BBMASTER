import argparse
from soopercool import BBmeta
from soopercool import mpi_utils as mpi
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose

    do_plots = not args.no_plots

    out_dir = meta.output_directory
    masks_dir = f"{out_dir}/masks"

    sims_dir = f"{out_dir}/cov_sims"
    BBmeta.make_dir(sims_dir)

    signal_alm_dir = meta.covariance["signal_alm_sims_dir"]
    signal_alm_template = meta.covariance["signal_alm_sims_template"]

    binary = hp.read_map(f"{masks_dir}/binary_mask.fits")

    lmax = 3*meta.nside - 1
    ll = np.arange(lmax + 1)
    cl2dl = ll*(ll+1)/2./np.pi

    beams = {
        ms: meta.read_beam(ms)[1][:lmax+1]
        for ms in meta.map_sets_list
    }

    if do_plots:
        field_pairs = ["TT", "EE", "BB", "TE"]
        fiducial_cmb = meta.covariance["fiducial_cmb"]
        fiducial_dust = meta.covariance["fiducial_dust"]
        fiducial_synch = meta.covariance["fiducial_synch"]
        clth = {}

        for ms in meta.map_sets_list:
            nu = meta.freq_tag_from_map_set(ms)
            for i, fp in enumerate(field_pairs):
                cmb_cl = hp.read_cl(fiducial_cmb)[i, :lmax+1]
                dust_cl = hp.read_cl(
                    fiducial_dust.format(nu1=nu, nu2=nu)
                )[i, :lmax+1]
                synch_cl = hp.read_cl(
                    fiducial_synch.format(nu1=nu, nu2=nu)
                )[i, :lmax+1]
                clth[ms, fp] = (cmb_cl + dust_cl + synch_cl) * beams[ms]**2

    mpi.init(True)

    for id_sim in mpi.taskrange(meta.covariance["cov_num_sims"] - 1):

        base_dir = f"{sims_dir}/{id_sim:04d}"
        BBmeta.make_dir(base_dir)
        if do_plots:
            plots_dir = f"{out_dir}/plots/cov_sims/{id_sim:04d}"
            BBmeta.make_dir(plots_dir)

        for ms in meta.map_sets_list:
            if verbose:
                print(f"# {id_sim} | {ms}")

            noise_map_dir = meta.covariance["noise_map_sims_dir"][ms]
            noise_map_template = meta.covariance["noise_map_sims_template"][ms]

            ft = meta.freq_tag_from_map_set(ms)
            fname = signal_alm_template.format(id_sim=id_sim, freq_tag=ft)

            alms = hp.read_alm(f"{signal_alm_dir}/{fname}", hdu=(1, 2, 3))
            alms_beamed = [hp.almxfl(alm, beams[ms]) for alm in alms]
            signal = hp.alm2map(alms_beamed, nside=meta.nside)

            if do_plots:
                signal_cl = {fp: hp.anafast(signal)[i]
                             for i, fp in enumerate(field_pairs)}

            for id_bundle in range(meta.n_bundles_from_map_set(ms)):

                fname = noise_map_template.format(id_sim=id_sim, map_set=ms,
                                                  id_bundle=id_bundle)
                noise = hp.read_map(
                    f"{noise_map_dir}/{fname}", field=[0, 1, 2]
                )
                if do_plots and id_bundle == 0:
                    noise_cl = {
                        fp: hp.anafast(noise)[i]
                        for i, fp in enumerate(field_pairs)
                    }
                    for i, fp in enumerate(field_pairs):
                        plt.title(f"{ms} | bundle {id_bundle} | {fp}")
                        plt.loglog(ll, cl2dl*signal_cl[fp], c="navy",
                                   label="Signal")
                        plt.loglog(ll, cl2dl*noise_cl[fp], c="grey", alpha=0.5,
                                   label="Noise")
                        plt.loglog(ll, cl2dl*clth[ms, fp], c="darkorange",
                                   label="Theory (beamed)")
                        plt.xlabel(r"$\ell$", fontsize=15)
                        plt.ylabel(r"$D_\ell$", fontsize=15)
                        plt.legend()
                        plt.savefig(f"{plots_dir}/cl_{ms}_bundle{id_bundle}_{fp}.png")  # noqa
                        plt.close()

                split_map = signal + noise

                map_name = f"cov_sims_{ms}_bundle{id_bundle}.fits"
                hp.write_map(f"{base_dir}/{map_name}", split_map*binary,
                             overwrite=True,
                             dtype=np.float32)

                if do_plots:
                    for i, f in enumerate("TQU"):
                        hp.mollview(split_map[i]*binary,
                                    cmap="RdYlBu_r",
                                    title=f"{ms} - {id_sim} - {f}",
                                    min=-300 if f == "T" else -10,
                                    max=300 if f == "T" else 10)
                        plt.savefig(f"{plots_dir}/map_{ms}_bundle{id_bundle}_{f}.png") # noqa
                        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--no_plots", action="store_true", help="Do not plot the maps.") # noqa
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    main(args)
