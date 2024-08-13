import argparse
from soopercool import BBmeta
import healpy as hp
from soopercool import utils
from soopercool import map_utils as mu
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose
    out_dir = meta.output_directory

    sims_dir = f"{out_dir}/cmb_sims"
    plot_dir = f"{out_dir}/plots/cmb_sims"
    BBmeta.make_dir(sims_dir)
    BBmeta.make_dir(plot_dir)

    lmax_sim = 3*meta.nside - 1

    # Create the CMB fiducial cl
    lth, clth = utils.get_theory_cls(
        meta.covariance["cosmo"],
        lmax=lmax_sim  # ensure cl accuracy up to lmax
    )
    np.savez(f"{sims_dir}/cl_theory.npz",
             l=lth, **clth)

    beams = {
        ms: meta.read_beam(ms)[1]
        for ms in meta.map_sets_list
    }

    hp_ordering = ["TT", "TE", "TB", "EE", "EB", "BB"]

    for id_sim in range(meta.covariance["cov_num_sims"]):
        alms = hp.synalm([clth[fp] for fp in hp_ordering])
        for ms in meta.map_sets_list:
            if verbose:
                print(f"# {id_sim+1} | {ms}")

            alms_beamed = [hp.almxfl(alm, beams[ms]) for alm in alms]

            map = hp.alm2map(alms_beamed, nside=meta.nside)
            mu.write_map(f"{sims_dir}/cmb_{ms}_{id_sim:04d}.fits", map,
                         dtype=np.float32, convert_muK_to_K=True)
            hp.write_cl(f"{sims_dir}/cl_{ms}_{id_sim:04d}.fits",
                        hp.anafast(map), overwrite=True, dtype=np.float32)

    # Plotting
    cls = {ms: [] for ms in meta.map_sets_list}
    for ms in meta.map_sets_list:
        for id_sim in range(meta.covariance["cov_num_sims"]):
            cls[ms].append(
                hp.read_cl(f"{sims_dir}/cl_{ms}_{id_sim:04d}.fits")
            )
    cl_mean = {ms: np.mean(np.array(cls[ms]), axis=0)
               for ms in meta.map_sets_list}
    cl_std = {ms: np.std(np.array(cls[ms]), axis=0)
              for ms in meta.map_sets_list}

    ll = np.arange(lmax_sim + 1)
    cl2dl = ll*(ll + 1)/2./np.pi
    for ms in meta.map_sets_list:
        for ip, fp in enumerate(["TT", "EE", "BB", "TE"]):
            plt.title(f"{ms} | {fp}")
            plt.errorbar(ll, cl2dl*cl_mean[ms][ip], cl2dl*cl_std[ms][ip],
                         label="data", color="navy", lw=0.5, zorder=-32)
            plt.plot(ll, cl2dl*clth[fp], label=" theory",
                     color="k", ls="--")
            plt.plot(ll, cl2dl*clth[fp]*beams[ms]**2, label=" beamed theory",
                     color="darkorange", ls="--")
            plt.ylabel(r"$D_\ell$")
            plt.xlabel(r"$\ell$")
            plt.yscale("log")
            plt.legend()
            plt.savefig(f"{plot_dir}/cl_{ms}_{fp}.png")
            plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    main(args)
