import argparse
from soopercool import BBmeta, utils
from soopercool import map_utils as mu
import numpy as np
import os
import healpy as hp
import tarfile
import shutil
import wget
import astropy.io.fits as fits


def pre_processer(args):
    """
    Pre-process external data (Planck and WMAP)
    to be compatible with SOOPERCOOL.
    Main operations:
        - compute alms of original maps (expensive part)
        - rotate alms from galactic to equatorial coords
        - downgrade maps to nside defined in paramfile
    Simulations are read (already downgraded and rotated) from NERSC.
    If not found, they will be processed starting
    from original maps as done with the data.
    ----------
    Runs in three modes: "data", "sims", "noise".
    * "data": operate on real data
    * "sims": operate on simulations
    * "noise": operate on noise simulations
    ----------
    Parameters
    ----------
    args: dictionary
        Global parameters and command line arguments.
    ----------
    outputs:
        - Beams, temperature (both) and polarization (Planck only)
        - Spherical armonics transforms (alm) of data maps
        - Downgraded data maps, rotated in equatorial coords
        - Downgraded sims/noise maps, rotated in equatorial coords
    """

    meta = BBmeta(args.globals)

    output_dir = meta.output_directory
    sims_dir = meta.covariance["signal_alm_sims_dir"]

    directories = [output_dir, sims_dir]
    for dirs in directories:
        os.makedirs(dirs, exist_ok=True)

    if meta.pix_type == "car":
        raise NotImplementedError("Can only accept healpix for now.")

    binary_file = f"{output_dir}/masks/binary_mask.fits"
    if not os.path.isfile(binary_file):
        raise ValueError(f"Binary mask not found at {output_dir}/masks. "
                         "Have you run the mask maker?")

    binary_mask = mu.read_map(f"{output_dir}/masks/binary_mask.fits")
    nside_out = meta.nside
    lmax_out = 3*nside_out - 1
    ells_beam = np.arange(3*nside_out + 1)

    if args.planck:
        npipe_dir = "/global/cfs/cdirs/cmb/data/planck2020/npipe"
        bundles = ['A', 'B']
        bundles_dict = {'A': 0, 'B': 1}
        convert_to_K = 1

        print("Beams")
        bdir = meta.beam_dir_from_map_set(meta.map_sets_list[0])
        os.makedirs(bdir, exist_ok=True)
        beam_windows_dir = f"{bdir}/simulated_maps/npipe_aux/beam_window_functions"  # noqa

        # Download Planck NPIPE RIMO from URL and save temporarily
        url_pref = "https://portal.nersc.gov/cfs/cmb/planck2020/misc"
        url = f"{url_pref}/PLANCK_RIMO_TF_R4.00.tar.gz"
        fname_rimo = f"{bdir}/PLANCK_RIMO_TF_R4.00.tar.gz"
        if not os.path.isfile(fname_rimo):
            print("Downloading Planck RIMO...")
            wget.download(url, fname_rimo)
            print("\n")
        tf = tarfile.open(fname_rimo)
        tf.extractall(bdir)
        tf.close()

        spectra = ["TT", "EE", "BB", "TE"]
        leakage_term = {}
        for spec in spectra:
            leakage_term[spec] = [f"{spec}_2_TT", f"{spec}_2_EE",
                                  f"{spec}_2_BB", f"{spec}_2_TE",
                                  f"{spec}_2_TB", f"{spec}_2_EB",
                                  f"{spec}_2_ET", f"{spec}_2_BT",
                                  f"{spec}_2_BE"]

        beams = {}
        for map_set in meta.map_sets_list:
            if 'planck' not in map_set:
                continue
            print(map_set)
            f = str(meta.freq_tag_from_map_set(map_set)).zfill(3)
            beam_dir = meta.beam_dir_from_map_set(map_set)
            beam_file = f"{beam_dir}/{meta.beam_file_from_map_set(map_set)}"
            beam_file_T = f"{beam_dir}/beam_T_{map_set}.dat"
            beam_file_pol = f"{beam_dir}/beam_pol_{map_set}.dat"

            if os.path.isfile(beam_file):
                print("beams found in", beam_dir)
                _, beams[map_set] = np.loadtxt(beam_file, unpack=True)
                continue

            wl = fits.open(f"{beam_windows_dir}/full_frequency/Wl_npipe6v20_{f}GHzx{f}GHz.fits")  # noqa
            wl_dict = {}
            num = 1
            for spec in spectra:
                for leak in leakage_term[spec]:
                    wl_dict[leak] = wl[num].data[leak]
                num += 1

            # Planck beam: sqrt of XX_2_XX term of the beam leakage matrix
            bl_T = np.sqrt(wl_dict["TT_2_TT"][0])[:(3*nside_out+1)]
            bl_pol = np.sqrt(wl_dict["EE_2_EE"][0])[:(3*nside_out+1)]
            beams[map_set] = bl_pol

            # Save beams
            np.savetxt(beam_file_T, np.transpose([ells_beam, bl_T]))
            np.savetxt(beam_file_pol, np.transpose([ells_beam, bl_pol]))
            # DEBUG
            print(f"Saving beam under {beam_file}")
            np.savetxt(beam_file, np.transpose([ells_beam, bl_pol]))

            # Bundle-dependent beams
            for bundle in bundles:
                print(f"{map_set}_bundle{bundle}")
                beam_file_T = f"{beam_dir}/beam_T_{map_set}_bundle{bundle}.dat"
                beam_file_pol = f"{beam_dir}/beam_pol_{map_set}_bundle{bundle}.dat"  # noqa

                if os.path.isfile(beam_file_pol):
                    print("beams found in", beam_dir)
                    continue
                wl = fits.open(f"{beam_windows_dir}/AB/Wl_npipe6v20_{f}{bundle}x{f}{bundle}.fits")  # noqa
                wl_dict = {}
                num = 1
                for spec in spectra:
                    for leak in leakage_term[spec]:
                        wl_dict[leak] = wl[num].data[leak]
                    num += 1

                # Planck beam: sqrt of XX_2_XX term of the beam leakage matrix
                bl_T = np.sqrt(wl_dict["TT_2_TT"][0])[:(3*nside_out+1)]
                bl_pol = np.sqrt(wl_dict["EE_2_EE"][0])[:(3*nside_out+1)]

                np.savetxt(beam_file_T, np.transpose([ells_beam, bl_T]))
                np.savetxt(beam_file_pol, np.transpose([ells_beam, bl_pol]))

        if os.path.isdir(f"{bdir}/simulated_maps"):
            shutil.rmtree(f"{bdir}/simulated_maps")

        if not args.noise:
            print("---------------------")
            print("Generating dipole template")
            # dipole amplitude, Galactic coordinates of the dipole direction
            # from NPIPE paper, Solar dipole measurements, Table 11
            dipole_amplitude = 3366.6  # [muK_cmb]
            dipole_longitude = np.radians(263.986)  # l [deg]
            dipole_latitude = np.radians(48.247)  # b [deg]
            # Convert Galactic coordinates to Cartesian coordinates
            x = np.cos(dipole_latitude) * np.cos(dipole_longitude)
            y = np.cos(dipole_latitude) * np.sin(dipole_longitude)
            z = np.sin(dipole_latitude)
            # Generate a dipole template (for the two possible nside)
            dipole_template = {}
            dipole_template[1024] = dipole_amplitude * np.dot([x, y, z], hp.pix2vec(1024, np.arange(hp.nside2npix(1024))))  # noqa
            dipole_template[2048] = dipole_amplitude * np.dot([x, y, z], hp.pix2vec(2048, np.arange(hp.nside2npix(2048))))  # noqa

    elif args.wmap:
        convert_to_K = 1.e-3
        bands_dict = {'023': 'K1', '033': 'Ka1'}

        for map_set in meta.map_sets_list:
            if 'wmap' in map_set:
                f = meta.freq_tag_from_map_set(map_set)
                nbundles = meta.n_bundles_from_map_set(map_set)
        bundles = list(range(1, nbundles+1))

        if args.data:
            for map_set in meta.map_sets_list:
                if 'wmap' not in map_set:
                    continue
                f = str(meta.freq_tag_from_map_set(map_set)).zfill(3)

                map_dir = meta.map_dir_from_map_set(map_set)
                meta.timer.start("Download WMAP data")
                url_pref = "https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/1yr/raw"  # noqa

                # Download WMAP single-year maps from URL and store
                print("Downloading WMAP single-year maps")
                for yr in bundles:
                    fname_map = f"wmap_iqumap_r9_yr{yr}_{bands_dict[f]}_v5.fits"  # noqa
                    print("-", fname_map)
                    fname_data_map = f"{map_dir}/{map_set}_bundle{yr-1}_map.fits"  # noqa
                    if os.path.isfile(fname_data_map):
                        print("  data maps already on disk")
                        continue
                    url = f"{url_pref}/{fname_map}"
                    fname_out = f"{map_dir}/{fname_map}"
                    if os.path.isfile(fname_out):
                        print("  data already downloaded, at", map_dir)
                        continue
                    wget.download(url, fname_out)
                    print("\n")

                print("Mask")
                # Download WMAP temperature analysis mask from URL and store
                url_pref = "https://lambda.gsfc.nasa.gov/data/map/dr5/ancillary/masks"  # noqa
                url = f"{url_pref}/wmap_temperature_kq85_analysis_mask_r9_9yr_v5.fits"  # noqa
                fname_mask = f"{output_dir}/masks/wmap_temperature_analysis_mask.fits"  # noqa

                os.makedirs(f"{output_dir}/masks", exist_ok=True)
                if not os.path.isfile(fname_mask):
                    wget.download(url, fname_mask)
                    print("\n")
                mask = mu.read_map(fname_mask, field=['N_OBS'])

                print("Beams")
                print("Downloading WMAP beam window functions")
                url_pref = "https://lambda.gsfc.nasa.gov/data/map/dr5/ancillary/beams"  # noqa

                print(map_set)
                beam_dir = meta.beam_dir_from_map_set(map_set)
                beam_file = f"{beam_dir}/{meta.beam_file_from_map_set(map_set)}"  # noqa

                if os.path.isfile(beam_file):
                    print(f"beams found at {beam_dir}")
                    continue
                if not os.path.isfile(beam_file):
                    url = f"{url_pref}/wmap_ampl_bl_{bands_dict[f]}_9yr_v5p1.txt"  # noqa
                    wget.download(url, beam_file)
                    print("\n")
                # read beams
                # if last ell < 3*nside_out
                # extend beams repeating the last value
                bl = np.zeros(3*nside_out+1)
                l, b, _ = np.loadtxt(beam_file, unpack=True)
                lmax_file = int(l[-1])
                if lmax_file < 3*nside_out:
                    bl[:(lmax_file+1)] = b
                    bl[lmax_file:] = b[-1]
                else:
                    bl = b[:(3*nside_out+1)]
                np.savetxt(beam_file, np.transpose([ells_beam, bl]))
            meta.timer.stop("Download WMAP data")

    if args.data:
        # rotating and downgrading data maps
        # both operations carried out in harmonic space (alms)
        # to conserve harmonic properties
        print("---------------------")
        print("Processing external maps")
        meta.timer.start("Process external maps")
        angles = hp.rotator.coordsys2euler_zyz(coord=["G", "C"])

        for map_set in meta.map_sets_list:
            if args.planck:
                if "planck" not in map_set:
                    continue
                # original nside
                f = str(meta.freq_tag_from_map_set(map_set)).zfill(3)
                nside_in = 1024 if f == '030' else 2048
            elif args.wmap:
                if "wmap" not in map_set:
                    continue
                nside_in = 512

            map_dir = meta.map_dir_from_map_set(map_set)

            dgrade = (nside_in != nside_out)
            if dgrade:  # if downgrading
                # which alms indices to clip before downgrading
                lmax_in = 3*nside_in - 1
                clip_indices = []
                for m in range(lmax_out+1):
                    clip_indices.append(hp.Alm.getidx(lmax_in, np.arange(m, lmax_out+1), m))  # noqa
                clip_indices = np.concatenate(clip_indices)

            for bundle in bundles:
                print("---------------------")
                print(f"- f{f} - bundle {bundle}")
                if args.planck:
                    fname_in = f"{npipe_dir}/npipe6v20{bundle}/npipe6v20{bundle}_{f}_map.fits"  # noqa
                    fname_alms = f"{map_dir}/npipe6v20{bundle}/alms_npipe6v20{bundle}_{f}_map_ns{nside_in}.npz"  # noqa
                    fname_out = f"{map_dir}/{map_set}_bundle{bundles_dict[bundle]}_map.fits"  # noqa
                    os.makedirs(f"{map_dir}/npipe6v20{bundle}", exist_ok=True)
                elif args.wmap:
                    fname_in = f"{map_dir}/wmap_{bands_dict[f]}/wmap_iqumap_r9_yr{bundle}_{bands_dict[f]}_v5.fits"  # noqa
                    fname_alms = f"{map_dir}/wmap_{bands_dict[f]}/alms_wmap_{bands_dict[f]}_yr{bundle}_map_{nside_in}.npz"  # noqa
                    fname_out = f"{map_dir}/{map_set}_bundle{bundle-1}_map.fits"  # noqa
                    os.makedirs(f"{map_dir}/wmap_{bands_dict[f]}",
                                exist_ok=True)

                if os.path.isfile(fname_out):
                    print("maps already processed, found in", map_dir)
                    continue
                if os.path.isfile(fname_alms):
                    print("reading alms at", fname_alms)
                    alm_file = np.load(fname_alms)
                    alm = np.array([alm_file['alm_T'],
                                    alm_file['alm_E'],
                                    alm_file['alm_B']])
                else:
                    print("reading input maps at", fname_in)
                    # read TQU, convert to K
                    map_in = convert_to_K * mu.read_map(fname_in,
                                                        field=[0, 1, 2])
                    if args.planck:
                        print("removing dipole")
                        map_in[0] -= dipole_template[nside_in]
                    elif args.wmap:
                        # remove mean temperature outside mask
                        map_in_masked = hp.ma(map_in[0])
                        map_in_masked.mask = np.logical_not(mask)
                        map_in[0] -= np.mean(map_in_masked)

                    print("computing alms")
                    alm = hp.map2alm(map_in, pol=True, use_pixel_weights=True)
                    del map_in
                    print("writing alms to disk")
                    np.savez(fname_alms, alm_T=alm[0],
                             alm_E=alm[1], alm_B=alm[2])

                if dgrade:
                    print(f"downgrading to nside={nside_out}")
                    # clipping alms at l>lmax_out to avoid artifacts
                    alm = [each[clip_indices] for each in alm]

                print("rotating alms from galactic to equatorial coords")
                hp.rotate_alm(alm, *angles)

                print("projecting alms to maps")
                map_out = hp.alm2map(alm, nside=nside_out, pol=True)

                # mask external maps with SO binary mask
                map_out *= binary_mask

                print("writing maps to disk")
                mu.write_map(fname_out, map_out, dtype=np.float32)

                if args.plots:
                    print("plotting maps")

                    if args.planck:
                        utils.plot_map(map_out,
                                       f"{map_dir}/plots/map_{map_set}_bundle{bundles_dict[bundle]}",  # noqa
                                       title=map_set, TQU=True)
                    elif args.wmap:
                        utils.plot_map(map_out,
                                       f"{map_dir}/plots/map_{map_set}_bundle{bundle-1}",  # noqa
                                       title=map_set, TQU=True)
        if 'm_out' in locals():
            del map_out
        if 'alm' in locals():
            del alm
        meta.timer.stop("Process external maps")

    if args.sims or args.noise:
        print("---------------------")
        print("Processing simulations")
        meta.timer.start("Process simulations")
        nsims = meta.covariance["cov_num_sims"]
        # path to already downgraded and rotated npipe sims
        ext_sims_dir = "/global/cfs/cdirs/sobs/users/cranucci"
        ext_sims_dir += "/npipe" if args.planck else "/wmap"
        ext_sims_dir += f"/nside{nside_out}_coords_eq"

        if os.path.isdir(ext_sims_dir) or args.wmap:
            process_sims = False
        else:
            # process original npipe sims maps
            process_sims = True
            # path to original maps
            ext_sims_dir = npipe_dir

        for sim_id in range(nsims):
            # starting seed from 0200 to 0000 (if processing original maps)
            sim_id_in = sim_id+200 if process_sims else sim_id

            # sims output dirs
            os.makedirs(f"{sims_dir}/{sim_id:04d}", exist_ok=True)

            for map_set in meta.map_sets_list:
                if "planck" not in map_set and "wmap" not in map_set:
                    continue

                if args.noise:
                    # noise output dirs
                    noise_dir = meta.covariance["noise_map_sims_dir"][map_set]
                    noise_dir += f"/{sim_id:04d}"
                    os.makedirs(noise_dir, exist_ok=True)

                f = str(meta.freq_tag_from_map_set(map_set)).zfill(3)
                # original nside
                nside_in = 1024 if f == '030' else 2048
                dgrade = (nside_in != nside_out)

                if process_sims and dgrade:
                    # which alms indices to clip before downgrading
                    lmax_in = 3*nside_in - 1
                    clip_indices = []
                    for m in range(lmax_out+1):
                        clip_indices.append(hp.Alm.getidx(lmax_in, np.arange(m, lmax_out+1), m)) # noqa
                    clip_indices = np.concatenate(clip_indices)

                for bundle in bundles:
                    print("---------------------")
                    print(f"sim {sim_id:04d} - channel {f} - bundle {bundle}")
                    if args.noise:
                        # noise fnames (Planck and WMAP)
                        if args.planck:
                            fname_in = f"{ext_sims_dir}/npipe6v20{bundle}_sim"
                            fname_in += f"/{sim_id_in:04d}/residual"
                            fname_in += f"/residual_npipe6v20{bundle}_{f}_{sim_id_in:04d}.fits"  # noqa
                            fname_out = f"{noise_dir}/{map_set}_bundle{bundles_dict[bundle]}_noise.fits"  # noqa
                        elif args.wmap:
                            fname_in = f"{ext_sims_dir}/noise/{sim_id_in:04d}"
                            fname_in += f"/noise_maps_mK_band{bands_dict[f]}_yr{bundle}.fits"  # noqa
                            fname_out = f"{noise_dir}/{map_set}_bundle{bundle-1}_noise.fits"  # noqa
                    else:
                        # sims fnames (only Planck)
                        fname_in = f"{ext_sims_dir}/npipe6v20{bundle}_sim"
                        fname_in += f"/{sim_id_in:04d}/npipe6v20{bundle}_{f}_map.fits"  # noqa
                        fname_out = f"{sims_dir}/{sim_id:04d}/{map_set}_bundle{bundles_dict[bundle]}_map.fits"  # noqa

                    if os.path.isfile(fname_out):
                        print("sims already processed, found in", sims_dir)
                        continue

                    print("reading input maps at", fname_in)
                    # read TQU, convert to K
                    map_in = convert_to_K * mu.read_map(fname_in,
                                                        field=[0, 1, 2])

                    if process_sims:
                        if not args.noise:
                            # Subtract the dipole (not from noise)
                            map_in[0] -= dipole_template[nside_in]
                        print("computing alms")
                        alm = hp.map2alm(map_in,
                                         pol=True, use_pixel_weights=True)
                        del map_in

                        if dgrade:
                            print(f"downgrading to nside={nside_out}")
                            # clipping alms at l>lmax_out to avoid artifacts
                            alm = [each[clip_indices] for each in alm]
                        print("rotating alms, galactic -> equatorial")
                        hp.rotate_alm(alm, *angles)
                        print("projecting alms to map")
                        map_out = hp.alm2map(alm, nside=nside_out, pol=True)
                    else:
                        map_out = map_in

                    print("masking")
                    map_out *= binary_mask
                    print("writing maps to disk")
                    mu.write_map(fname_out, map_out, dtype=np.float32)

        meta.timer.stop("Process simulations")
        print("---------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-processing external data for the pipeline")
    parser.add_argument("--globals", type=str,
                        help="Path to the yaml with global parameters")
    parser.add_argument("--planck", action="store_true",
                        help="Pass to process Planck maps")
    parser.add_argument("--wmap", action="store_true",
                        help="Pass to process WMAP maps")
    parser.add_argument("--plots", action="store_true",
                        help="Pass to generate plots")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--data", action="store_true")
    mode.add_argument("--sims", action="store_true")
    mode.add_argument("--noise", action="store_true")

    args = parser.parse_args()

    if args.wmap and args.sims:
        print("WMAP simulations not available yet.")
        print("Only noise sims are available.")
        print("exiting...")
    else:
        pre_processer(args)
