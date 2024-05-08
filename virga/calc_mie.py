import numpy as np
pi = np.pi
import PyMieScatt as ps
import os
import pandas as pd
from jdi_utils import get_r_grid_w_max

from YASF.yasfpy.particles import Particles
from YASF.yasfpy.initial_field import InitialField
from YASF.yasfpy.parameters import Parameters
from YASF.yasfpy.solver import Solver
from YASF.yasfpy.numerics import Numerics
from YASF.yasfpy.simulation import Simulation
from YASF.yasfpy.optics import Optics
from .particle_generator import ParticleGenerator, Particle
from pathlib import Path
from frameworks.mmf import mmf_parsing


def calc_mieff(wave_in, nn,kk, radius, rup, fort_calc_mie=False):
    nradii = len(radius)
    nwave = len(wave_in)  # number of wavalength bin centres for calculation

    qext = np.zeros((nwave, nradii))
    qscat = np.zeros((nwave, nradii))
    cos_qscat = np.zeros((nwave, nradii))

    for iwave in range(nwave):
        for irad in range(nradii):

            corerad = 0.0
            corereal = 1.0
            coreimag = 0.0

            wave = wave_in * 1e3  ## converting to nm
            ## averaging over 6 radial bins to avoid fluctuations
                # arr = qext, qsca, qabs, g, qpr, qback, qratio
            arr = ps.MieQCoreShell(
                corereal + (1j) * coreimag,
                nn[iwave] + (1j) * kk[iwave],
                wave[iwave],
                dCore=0,
                dShell=2.0 * radius[irad] * 1e7,
            )

            qext[iwave, irad] = arr[0]
            qscat[iwave, irad] = arr[1]
            cos_qscat[iwave, irad] = arr[3] * arr[1]

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
        sep='\+s',
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


def calc_scattering(properties: Particle, gas_name: str, data_dir: Path, mode: str="YASF"):
    assert (mode == "YASF" or mode == "MMF"), "Only valid modes are 'YASF' and 'MMF'"

    radii = properties.radii
    nradii = len(radii)
    wave_in, _, _ = get_refrind(gas_name, data_dir)
    print(f"{wave_in = }")
    nwave = len(wave_in)  # number of wavalength bin centres for calculation

    qext = np.zeros((nwave, nradii))
    qscat = np.zeros((nwave, nradii))
    cos_qscat = np.zeros((nwave, nradii))

    if mode == "YASF":
        # prep yasf

        particle_generator = ParticleGenerator()
        for r_idx in range(len(radii)):
            particle_csv = particle_generator.aggregate_generator(radius=radii[r_idx],df=properties.Df,N=properties.N[r_idx], directory=data_dir,kf=properties.kf)
            refractive_index_table = read_virga_refrinds(gas_name, data_dir)
            refractive_index_table = [{"ref_idx": refractive_index_table[0], "material": refractive_index_table[1]}]
            particles, numerics, simulation, optics = prep_yasf(refractive_index_table,particle_csv, wavelength=wave_in)
            q_ext, q_scat, g = run_yasf(particles, numerics, simulation, optics, gas_name, data_dir, wave_in)

            qext[:,r_idx] = q_ext
            qscat[:,r_idx] = q_scat
            cos_qscat[:,r_idx] = g*q_scat

    elif mode == "MMF":
        material = "Enstatite"
        refractive_index_table = read_virga_refrinds(gas_name, data_dir)
        # refractive_index_table = [{"ref_idx": refractive_index_table[0], "material": refractive_index_table[1]}]
        for r_idx in range(len(radii)):
            p = mmf_parsing.run_optool(N=properties.N[r_idx],a0=properties.monomer_size,refrinds=refractive_index_table[0],rho=properties.rho,df=properties.Df,kf=properties.kf, wavelengths=wave_in)
            q_ext, q_scat = mmf_parsing.get_efficiencies(p, properties.N[r_idx], properties.rho)
            qext[:,r_idx] = q_ext
            qscat[:,r_idx] = q_scat
            cos_qscat[:,r_idx] = p.gsca*q_scat

    return qext, qscat, cos_qscat, nwave, radii ,wave_in

def read_virga_refrinds(gas_name: str, data_dir: Path):
    path = data_dir / Path(gas_name+".refrind")
    data = pd.read_csv(
        path , delim_whitespace=True, header=0, names=["wavelength", "n", "k"]
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
