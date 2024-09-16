import pickle
import numpy as np
pi = np.pi
import PyMieScatt as ps
import os
import pandas as pd
import csv
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname("/home/dsc/master/yasf_testing"))
sys.path.append(os.path.dirname("/home/dsc/master/"))
sys.path.append(os.path.dirname("/home/dsc/master/virga/"))
sys.path.append(os.path.dirname("/home/dsc/master/frameworks/"))
sys.path.append(os.path.dirname("/Users/dusc/Code/master/"))

from virga.jdi_utils import get_r_grid_w_max

from YASF.yasfpy.particles import Particles
from YASF.yasfpy.initial_field import InitialField
from YASF.yasfpy.parameters import Parameters
from YASF.yasfpy.solver import Solver
from YASF.yasfpy.numerics import Numerics
from YASF.yasfpy.simulation import Simulation
from YASF.yasfpy.optics import Optics
from pathlib import Path
from frameworks.mmf import mmf_parsing
from frameworks.mstm import mstm4
from particle_generator.particle_generator import ParticleGenerator, Particle

VALID_MODES = ["MMF", "MSTM", "YASF"]

mmf_parsing.OPTOOL_BIN_PATH = "/home/dsc/master/optool/optool"
FRACAL_BIN_PATH = "/home/dsc/master/FracVAL/FRACVAL"
NCORES = 20

def calc_mieff(wave_in, nn,kk, radius, rup, fort_calc_mie=False):
    nradii = len(radius)
    nwave = len(wave_in)  # number of wavalength bin centres for calculation

    qext = np.zeros((nwave, nradii))
    qscat = np.zeros((nwave, nradii))
    cos_qscat = np.zeros((nwave, nradii))

    for iwave in range(nwave):
        print("WOOP")
        for irad in range(nradii):

            corerad = 0.0
            corereal = 1.0
            coreimag = 0.0

            wave = wave_in * 1e3  ## converting to nm
            ## averaging over 6 radial bins to avoid fluctuations
                # arr = qext, qsca, qabs, g, qpr, qback, qratio
            start = time.monotonic()
            arr = ps.MieQCoreShell(
                corereal + (1j) * coreimag,
                nn[iwave] + (1j) * kk[iwave],
                wave[iwave],
                dCore=0,
                dShell=2.0 * radius[irad] * 1e7, 
            )
            # print(f"Took {time.monotonic()-start}s !!")

            qext[iwave, irad] = arr[0]
            qscat[iwave, irad] = arr[1]
            cos_qscat[iwave, irad] = arr[3] * arr[1]

    return qext, qscat, cos_qscat

def calc_mieff_new(wave_in, nn,kk, radius, rup):
    nwave = len(wave_in)  # number of wavalength bin centres for calculation

    corerad = 0.0
    corereal = 1.0
    coreimag = 0.0
    qext = np.zeros((nwave,))
    qscat = np.zeros((nwave,))
    cos_qscat = np.zeros((nwave,))
    # FIXME: make sure radii have correct unit

    sub_radii = 6
    wave=wave_in*1e3  ## converting to nm 
    ## averaging over 6 radial bins to avoid fluctuations
    dr5= (( rup - radius ) / 5.)
    rr= radius

    for iwave in range(nwave):
        for isub in range(sub_radii):
            #arr = qext, qsca, qabs, g, qpr, qback, qratio
            arr= ps.MieQCoreShell( corereal+(1j)*coreimag, 
                                    nn[iwave]+(1j)*kk[iwave], 
                                    wave[iwave],dCore=0,dShell=2.0*rr*1e3) # i did everything in µm for optool, so only *1e3 to get nm

            qext[iwave]+= arr[0]
            qscat[iwave]+= arr[1]
            cos_qscat[iwave] += arr[3]*arr[1] 
            rr+=dr5

    return qext, qscat, cos_qscat


def calc_new_mieff(wave_in, nn, kk, radius, rup, fort_calc_mie=False):
    ## Calculates optics by reading refrind files
    thetd = 0.0  # incident wave angle
    n_thetd = 1
    # number of radii sub bins in order to smooth out fringe effects

    # TODO: @dusc:
    # why are multiple radii calculated? how are these results used? Is this procedure compatible with the use of
    # more sophisticated models for particles in the atmoshpere?

    # TODO: @dusc:
    # for every specificed radius, a maximum radius is specified. Then, (sub_radii) different radii between (radius)
    # and (rup) will be used for mie calculation. The resulting cross sections are then averaged.
    # Why is this done? How does this change the results?

    sub_radii = 6

    nradii = len(radius)
    nwave = len(wave_in)  # number of wavalength bin centres for calculation

    qext = np.zeros((nwave, nradii))
    qscat = np.zeros((nwave, nradii))
    cos_qscat = np.zeros((nwave, nradii))

    # compute individual parameters for each gas
    for iwave in range(nwave):
        for irad in range(nradii):
            if irad == 0:
                dr5 = (rup[0] - radius[0]) / 5.0
                rr = radius[0]
            else:
                dr5 = (rup[irad] - rup[irad - 1]) / 5.0
                rr = rup[irad - 1]
            corerad = 0.0
            corereal = 1.0
            coreimag = 0.0
            ## averaging over 6 radial bins to avoid fluctuations
            # this is the default.
            # if no fortran crappy code, use PyMieScatt which does a much faster
            # more robust computation of the Mie parameters
            wave = wave_in * 1e3  ## converting to nm
            ## averaging over 6 radial bins to avoid fluctuations
            for isub in range(sub_radii):
                # arr = qext, qsca, qabs, g, qpr, qback, qratio
                arr = ps.MieQCoreShell(
                    corereal + (1j) * coreimag,
                    nn[iwave] + (1j) * kk[iwave],
                    wave[iwave],
                    dCore=0,
                    dShell=2.0 * rr * 1e7,
                )

                qext[iwave, irad] += arr[0]
                qscat[iwave, irad] += arr[1]
                cos_qscat[iwave, irad] += arr[3] * arr[1]
                rr += dr5

            ## adding to master arrays
            qext[iwave, irad] = qext[iwave, irad] / sub_radii
            qscat[iwave, irad] = qscat[iwave, irad] / sub_radii
            cos_qscat[iwave, irad] = cos_qscat[iwave, irad] / sub_radii

    return qext, qscat, cos_qscat


def get_refrind(igas, directory):
    """
    Reads reference files with wavelength, and refractory indecies.
    This function relies on input files being structured as a 4 column file with
    columns: index, wavelength (micron), nn, kk

    Parameters
    ----------
    igas : str
        Gas name
    directory : str
        Directory were reference files are located.
    """
    filename = os.path.join(directory, igas + ".refrind")
    try:
        _, wave_in, nn, kk = np.loadtxt(open(filename,'rt').readlines(), unpack=True, usecols=[0,1,2,3])#[:-1]
        return wave_in,nn,kk
    except:
        df = pd.read_csv(filename)
        wave_in = df['micron'].values
        nn = df['real'].values
        kk = df['imaginary'].values
        return wave_in,nn,kk


def get_r_grid(r_min=1e-5, n_radii=40):
    """
    Get spacing of radii to run Mie code

    r_min : float
        Minimum radius to compute (cm)

    n_radii : int
        Number of radii to compute
    """
    vrat = 2.2
    pw = 1.0 / 3.0
    f1 = (2.0 * vrat / (1.0 + vrat)) ** pw
    f2 = ((2.0 / (1.0 + vrat)) ** pw) * (vrat ** (pw - 1.0))

    radius = r_min * vrat ** (np.linspace(0, n_radii - 1, n_radii) / 3.0)
    rup = f1 * radius
    dr = f2 * radius

    return radius, rup, dr


def calc_mie_db(gas_name, dir_refrind, dir_out, rmin=1e-8, rmax = 5.4239131e-2, nradii = 60, fort_calc_mie = False):
    """
    Function that calculations new Mie database using PyMieScatt.

    Parameters
    ----------
    gas_name : list, str
        List of names of gasses. Or a single gas name.
        See pyeddy.available() to see which ones are currently available.
    dir_refrind : str
        Directory where you store optical refractive index files that will be created.
    dir_out: str
        Directory where you want to store Mie parameter files. Will be stored as gas_name.Mieff.
        BEWARE FILE OVERWRITES.
    rmin : float , optional
        (Default=1e-5) Units of cm. The minimum radius to compute Mie parameters for.
        Usually 0.1 microns is small enough. However, if you notice your mean particle radius
        is on the low end, you may compute your grid to even lower particle sizes.
    nradii : int, optional
        (Default=40) number of radii points to compute grid on. 40 grid points for exoplanets/BDs
        is generally sufficient.

    Returns
    -------
    Q extinction, Q scattering,  asymmetry * Q scattering, radius grid (cm), wavelength grid (um)

    The Q "efficiency factors" are = cross section / geometric cross section of particle
    """
    if isinstance(gas_name, str):
        gas_name = [gas_name]
    ngas = len(gas_name)

    for i in range(len(gas_name)):
        print("Computing " + gas_name[i])
        # Setup up a particle size grid on first run and calculate single-particle scattering

        # files will be saved in `directory`
        # obtaining refractive index data for each gas
        wave_in, nn, kk = get_refrind(gas_name[i], dir_refrind)
        nwave = len(wave_in)
        print(f"{nwave = }")

        if i == 0:
            # all these files need to be on the same grid
            print(f"{nradii = }")
            print(f"{rmin = }")
            print("default")
            radius, rup, dr = get_r_grid_w_max(r_min=rmin, r_max=rmax, n_radii=nradii)
            print(f"{radius = }")
            print(f"{rup = }")
            print(f"{dr = }")
            # print("w_max")
            # radius, rup, dr = get_r_grid_w_max(r_min=rmin, n_radii=nradii)
            # print(f"{radius = }")
            # print(f"{rup = }")
            # print(f"{dr = }")

            qext_all = np.zeros(shape=(nwave, nradii, ngas))
            qscat_all = np.zeros(shape=(nwave, nradii, ngas))
            cos_qscat_all = np.zeros(shape=(nwave, nradii, ngas))

        # get extinction, scattering, and asymmetry
        # all of these are  [nwave by nradii]
        qext_gas, qscat_gas, cos_qscat_gas = calc_mieff(
            wave_in, nn, kk, radius, rup, fort_calc_mie=fort_calc_mie
        )

        # add to master matrix that contains the per gas Mie stuff
        qext_all[:, :, i], qscat_all[:, :, i], cos_qscat_all[:, :, i] = (
            qext_gas,
            qscat_gas,
            cos_qscat_gas,
        )

        # prepare format for old ass style # @dusc: ayo
        wave = [nwave] + sum([[r] + list(wave_in) for r in radius], [])
        qscat = [nradii] + sum([[np.nan] + list(iscat) for iscat in qscat_gas.T], [])
        qext = [np.nan] + sum([[np.nan] + list(iext) for iext in qext_gas.T], [])
        cos_qscat = [np.nan] + sum(
            [[np.nan] + list(icos) for icos in cos_qscat_gas.T], []
        )
        print(os.path.join(dir_out,gas_name[i]+".mieff"))
        pd.DataFrame(
            {"wave": wave, "qscat": qscat, "qext": qext, "cos_qscat": cos_qscat}
        ).to_csv(
            os.path.join(dir_out, gas_name[i] + ".mieff"),
            sep=" ",
            index=False,
            header=None,
        )
    return qext_all, qscat_all, cos_qscat_all, radius, wave_in


def get_mie(gas, directory):
    """
    Get Mie parameters from old ass formatted files
    """
    df = pd.read_csv(
        os.path.join(directory, gas + ".mieff"),
        names=["wave", "qscat", "qext", "cos_qscat"],
        sep=' ',
    )

    nwave = int(df.iloc[0, 0])
    nradii = int(df.iloc[0, 1])

    # get the radii (all the rows where there the last three rows are nans)
    radii = df.loc[np.isnan(df["qscat"])]["wave"].values

    df = df.dropna()

    assert (
        len(radii) == nradii
    ), "Number of radii specified in header is not the same as number of radii."
    assert (
        nwave * nradii == df.shape[0]
    ), "Number of wavelength specified in header is not the same as number of waves in file"

    # check if incoming wavegrid is in correct order
    sub_array = df['wave'].values[:196]  # Extract the first 196 values
    is_ascending = np.all(np.diff(sub_array) >= 0) # check if going from short to long wavelength

    if is_ascending == False:
        flipped_wave = np.flip(df['wave'].values.reshape(nradii, -1, nwave), axis=2).flatten()
        flipped_qscat = np.flip(df['qscat'].values.reshape(nradii, -1, nwave), axis=2).flatten()
        flipped_qext = np.flip(df['qext'].values.reshape(nradii, -1, nwave), axis=2).flatten()
        flipped_cos_qscat = np.flip(df['cos_qscat'].values.reshape(nradii, -1, nwave), axis=2).flatten()

        df['wave'] = flipped_wave
        df['qscat'] = flipped_qscat
        df['qext'] = flipped_qext
        df['cos_qscat'] = flipped_cos_qscat

    wave = df["wave"].values.reshape((nradii, nwave)).T
    qscat = df["qscat"].values.reshape((nradii, nwave)).T
    qext = df["qext"].values.reshape((nradii, nwave)).T
    cos_qscat = df["cos_qscat"].values.reshape((nradii, nwave)).T

    return qext, qscat, cos_qscat, nwave, radii, wave


def calc_scattering(properties: Particle, gas_name: str, data_dir: Path, mode: str="YASF", store=False, db_name="/home/dsc/virga-data"):
    assert (mode in VALID_MODES), "Only valid modes are 'YASF', 'MSTM' and 'MMF'"

    # ALL UNITS ARE IN CM! OPTOOL NEEDS µM!
    radii = list(np.array(properties.radii) * 1e4) # R_g done here, r_mon done below
    print(f"{radii[24:28] = }")
    print(f"{properties.monomer_size = }")
    print(f"{properties.N = }")
    print(f"{properties.Df = }")
    print(f"{properties.kf = }")
    nradii = len(radii)
    rmin = float(np.min(radii))
    wave_in, _, _ = get_refrind(gas_name, data_dir)

    _, rup, _ = get_r_grid(r_min=rmin*1e-4, n_radii=nradii)
    rup *= 1e4
    nwave = len(wave_in)  # number of wavalength bin centres for calculation
    # time.sleep(15)
    monomer_size = properties.monomer_size * 1e4
    qext = np.zeros((nwave, nradii))
    qscat = np.zeros((nwave, nradii))
    g0 = np.zeros((nwave, nradii))
    cos_qscat = np.zeros((nwave, nradii))

    # if the requested radius not sufficiently larger than the monomer radius, use actual Mie Scattering, aka treat particle as a sphere
    # ill use a seperate function and call that in the requested mode, if the radius does not fulfill the size criterion

    if mode == "YASF":
        # prep yasf

        particle_generator = ParticleGenerator(fracval_bin_path = FRACAL_BIN_PATH)
        for r_idx in range(len(radii)):
            if properties.N[r_idx] < 2:
                refractive_index_table = read_virga_refrinds(gas_name, data_dir)[0].to_numpy()
                qext, qscat, cos_qscat = calc_mieff(wave_in=wave_in,nn=refractive_index_table[:,1],kk=refractive_index_table[:,2],radius=radii,rup=rup)
            else:
                particle_csv = particle_generator.fracval(r_mon=monomer_size,df=properties.Df,N=properties.N[r_idx],r_agg=radii[r_idx], directory=data_dir,kf=properties.kf)
                refractive_index_table = read_virga_refrinds(gas_name, data_dir)
                refractive_index_table = [{"ref_idx": refractive_index_table[0], "material": refractive_index_table[1]}]
                particles, numerics, simulation, optics = prep_yasf(refractive_index_table,particle_csv, wavelength=wave_in)
                q_ext, q_scat, g = run_yasf(particles, numerics, simulation, optics, gas_name, data_dir, wave_in)
                cos_qscat = g*q_scat

            qext[:,r_idx] = q_ext
            qscat[:,r_idx] = q_scat
            cos_qscat[:,r_idx] = cos_qscat

    elif mode == "MMF":
        material = "Enstatite"
        refractive_index_table = read_virga_refrinds(gas_name, data_dir)[0].to_numpy()
        refrinds = np.array([complex(refractive_index_table[i,1],refractive_index_table[i,2]) for i in range(refractive_index_table.shape[0])])
        # refractive_index_table = [{"ref_idx": refractive_index_table[0], "material": refractive_index_table[1]}]
        for r_idx in range(len(radii)):

            print("AWOOGA")
            if properties.N[r_idx] < 2:
                start = time.monotonic()
                q_ext, q_scat, cosqscat = calc_mieff_new(wave_in=wave_in,nn=refractive_index_table[:,1],kk=refractive_index_table[:,2],radius=radii[r_idx],rup=rup[r_idx])
                print(f"Took {time.monotonic()-start}s for radius {radii[r_idx]}, rup = {rup[r_idx]}")
            else:
                print(f"CURRENT RADIUS: {radii[r_idx]}")
                print(f"CURRENT N: {properties.N[r_idx]}")
                print(f"CURRENT R0: {monomer_size}")
                # r_agg != a, use formula provided in optool manual
                a = (properties.N[r_idx]*(monomer_size)**3)**(1/3)
                print(f"CALCULATED a: {a}")
                p = mmf_parsing.run_optool(a=a,a0=monomer_size,refrinds=refrinds,rho=properties.rho,df=properties.Df,kf=properties.kf, wavelengths=wave_in)
                q_scat = p.ksca
                q_ext = p.kext
                cosqscat = p.gsca*q_scat
                # q_ext, q_scat = mmf_parsing.get_efficiencies(p, properties.N[r_idx], properties.rho, Df=properties.Df, kf=properties.kf)
            qext[:,r_idx] = q_ext
            qscat[:,r_idx] = q_scat
            cos_qscat[:,r_idx] = cosqscat
            # g0[:,r_idx] = p.gsca
            
            
    elif mode == "MSTM":
        particle_generator = ParticleGenerator(fracval_bin_path = FRACAL_BIN_PATH)
        for r_idx in range(len(radii)):
            if properties.N[r_idx] < 2:
                refractive_index_table = read_virga_refrinds(gas_name, data_dir)[0].to_numpy()
                qext, qscat, cos_qscat = calc_mieff(wave_in=wave_in,nn=refractive_index_table[:,1],kk=refractive_index_table[:,2],radius=radii,rup=rup)
            else:
                particle_csv = particle_generator.fracval(r_mon=monomer_size,df=properties.Df,N=properties.N[r_idx],r_agg=radii[r_idx], directory=data_dir,kf=properties.kf)
                refractive_index_table = read_virga_refrinds(gas_name, data_dir)
                medium_refractive_index = np.ones_like(wave_in)
                spheres = pd.read_csv(particle_csv, header=None, names=['x', 'y', 'z', 'r', 'm_idx'])
                spheres = spheres.to_numpy()
                output_file = mstm4.run_mstm4(spheres=spheres, refractive_indices=refractive_index_table[0],medium_refractive_index=medium_refractive_index,wavelengths=wave_in,lmax=6,N=NCORES)
                q_ext, q_scat, _, g, _ = mstm4.parse_results(output_file)
                cos_qscat = np.array(g)*np.array(q_scat)

            qext[:,r_idx] = np.array(q_ext)
            qscat[:,r_idx] = np.array(q_scat)
            cos_qscat[:,r_idx] = cos_qscat

    # sanity check
    scat_inp = {}
    scat_inp["wavelengths"] = wave_in
    scat_inp["r_mon"] = properties.monomer_size
    scat_inp["r_agg"] = radii
    scat_inp["refrinds"] = refrinds
    scat_inp["Df"] = properties.Df
    scat_inp["kf"] = properties.kf
    scat_inp["g0"] = p.gsca
    scat_inp["qext"] = qext
    scat_inp["qscat"] = qscat

    with open("SCAT_PROPS_MMF.pickle", "wb") as f:
        pickle.dump(scat_inp, f)

    print("===============================")
    print("===============================")
    print("FROM LIGHT SCAT CALC:")
    print(f"{radii = }")
    print("HOPE THATS COOL WITH YOU")
    print("===============================")
    print("===============================")
    if store:
        with open(os.path.join(db_name, gas_name + f"_kf_{properties.kf}_df_{properties.Df}_rmon_{np.round(monomer_size,2)}_{mode}.mieff"),"a") as f:
            pass
        with open(os.path.join(db_name, gas_name + f"_kf_{properties.kf}_df_{properties.Df}_rmon_{np.round(monomer_size,2)}_{mode}.mieff"),"w") as f:
            writer = csv.writer(f, delimiter =' ')
            writer.writerow([nwave, len(radii)])
            for r in range(len(radii)):
                writer.writerow([radii[r]])
                for w in range(len(wave_in)):
                    writer.writerow([wave_in[w], qscat[w,r], qext[w,r], cos_qscat[w,r]])


    return qext, qscat, cos_qscat, nwave, radii ,wave_in

def load_stored_fractal_scat_props(gas_name: str, properties: Particle, mode: str, data_dir: Path=Path("/home/dsc/virga-data/")):
    r_mon = np.round(properties.monomer_size * 1e4,2)
    file_name = gas_name+f"_kf_{properties.kf}_df_{properties.Df}_rmon_{r_mon}_{mode}.mieff"

    print("================================")
    print(f"LOADING PROPS FROM {file_name}")
    print("================================")
    time.sleep(5)
    df = pd.read_csv(
        os.path.join(data_dir, file_name),
        names=["wave", "qscat", "qext", "cos_qscat"],
        sep=' ',
    )

    nwave = int(df.iloc[0, 0])
    nradii = int(df.iloc[0, 1])

    print("WOOOOOOOOW")
    # get the radii (all the rows where there the last three rows are nans)
    radii = df.loc[np.isnan(df["qscat"])]["wave"].values
    print(f"{radii = }")

    df = df.dropna()

    assert (
        len(radii) == nradii
    ), "Number of radii specified in header is not the same as number of radii."
    assert (
        nwave * nradii == df.shape[0]
    ), "Number of wavelength specified in header is not the same as number of waves in file"

    # check if incoming wavegrid is in correct order
    sub_array = df['wave'].values[:196]  # Extract the first 196 values
    is_ascending = np.all(np.diff(sub_array) >= 0) # check if going from short to long wavelength

    if is_ascending == False:
        flipped_wave = np.flip(df['wave'].values.reshape(nradii, -1, nwave), axis=2).flatten()
        flipped_qscat = np.flip(df['qscat'].values.reshape(nradii, -1, nwave), axis=2).flatten()
        flipped_qext = np.flip(df['qext'].values.reshape(nradii, -1, nwave), axis=2).flatten()
        flipped_cos_qscat = np.flip(df['cos_qscat'].values.reshape(nradii, -1, nwave), axis=2).flatten()

        df['wave'] = flipped_wave
        df['qscat'] = flipped_qscat
        df['qext'] = flipped_qext
        df['cos_qscat'] = flipped_cos_qscat

    wave = df["wave"].values.reshape((nradii, nwave)).T
    qscat = df["qscat"].values.reshape((nradii, nwave)).T
    qext = df["qext"].values.reshape((nradii, nwave)).T
    cos_qscat = df["cos_qscat"].values.reshape((nradii, nwave)).T

    return qext, qscat, cos_qscat, nwave, radii, wave

def read_virga_refrinds(gas_name: str, data_dir: Path):
    path = data_dir / Path(gas_name+".refrind")
    data = pd.read_csv(
        path , delim_whitespace=True, header=None, names=["wavelength", "n", "k"]
    )
    # print(data)

    material = path.name.split(".")[0]
    # print(material)
    return [data, material]

# TODO: might wanna build classes for that in yasf, so that this isnt necessary and one gets
#       more options for parameters to set
def prep_yasf(refractive_index_table: list, particle_csv: Path, wavelength: list[float]):

    spheres = pd.read_csv(particle_csv, header=None, names=['x', 'y', 'z', 'r', 'm_idx'])

    medium_refractive_index = np.ones_like(wavelength)
    lmax = 12

    # load scattering modules
    spheres = spheres.to_numpy()
    particles = Particles(spheres[:,0:3], spheres[:,3], spheres[:,4], refractive_index_table=refractive_index_table)

    initial_field = InitialField(beam_width=0,
                                focal_point=np.array((0,0,0)),
                                polar_angle=0,
                                azimuthal_angle=0,
                                polarization='UNP')

    parameters = Parameters(wavelength=wavelength,
                            medium_refractive_index=medium_refractive_index,
                            particles=particles,
                            initial_field=initial_field)

    solver = Solver(solver_type='lgmres',
                    tolerance=5e-4,
                    max_iter=1000,
                    restart=500)

    numerics = Numerics(lmax = lmax,
                        #  sampling_points_number = [a // 3 for a in [360, 180]],
                        sampling_points_number = [a // 3 for a in [180, 360]],
                        polar_weight_func = lambda x: x**4,
                        particle_distance_resolution = 1,
                        gpu = True,
                        solver = solver)

    simulation = Simulation(parameters, numerics)

    optics = Optics(simulation)
    return particles, numerics, simulation, optics

def run_yasf(particles: Particles, numerics: Numerics, simulation: Simulation, optics: Optics, igas: str, directory: Path, wavelength: list[float]) -> tuple[np.ndarray,np.ndarray, np.ndarray]:
    particles.compute_volume_equivalent_area()
    numerics.compute_spherical_unity_vectors()
    numerics.compute_translation_table()
    simulation.compute_mie_coefficients()
    simulation.compute_initial_field_coefficients()
    simulation.compute_right_hand_side()
    simulation.compute_scattered_field_coefficients()
    optics.compute_cross_sections()
    optics.compute_phase_function_batched()
    optics.compute_asymmetry()


    q_ext = optics.c_ext/particles.geometric_projection
    q_sca = optics.c_sca/particles.geometric_projection
    g = optics.g

    return q_ext, q_sca, g

def get_mie_yasf(igas, directory: Path):
    # bruh
    pass

