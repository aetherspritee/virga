import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname("/home/dsc/master/"))
sys.path.append(os.path.dirname("/home/dsc/master/virga/"))

import pandas as pd
import numpy as np
from scipy import optimize
from virga.virga.root_functions import qvs_below_model
from virga.virga import gas_properties
from virga.virga import pvaps
import matplotlib.pyplot as plt
from bokeh.io import output_notebook
from virga.virga.direct_mmr_solver import direct_solver
from virga.virga.justplotit import find_nearest_1d
from virga.virga.calc_mie import calc_scattering, get_r_grid, calc_mie_db, get_mie, load_stored_fractal_scat_props
from virga.virga.layer import layer, layer_fractal
from particle_generator.particle_generator import Particle

class Atmosphere:
    def __init__(
        self,
        condensibles,
        fsed=0.5,
        b=1,
        eps=1e-2,
        mh=1,
        mmw=2.2,
        sig=2.0,
        param="const",
        verbose=True,
        supsat=0,
        gas_mmr=None,
    ):
        """
        Parameters
        ----------
        condensibles : list of str
            list of gases for which to consider as cloud species
        fsed : float
            Sedimentation efficiency coefficient. Jupiter ~3-6. Hot Jupiters ~ 0.1-1.
        b : float
            Denominator of exponential in sedimentation efficiency  (if param is 'exp')
        eps: float
            Minimum value of fsed function (if param=exp)
        mh : float
            metalicity
        mmw : float
            MMW of the atmosphere
        sig : float
            Width of the log normal distribution for the particle sizes
        param : str
            fsed parameterisation
            'const' (constant), 'exp' (exponential density derivation)
        verbose : bool
            Prints out warning statements throughout

        """
        self.mh = mh
        self.mmw = mmw
        self.condensibles = condensibles
        self.fsed = fsed
        self.b = b
        self.sig = sig
        self.param = param
        self.eps = eps
        self.verbose = verbose
        # grab constants
        self.constants()
        self.supsat = supsat
        self.gas_mmr = gas_mmr
        if isinstance(gas_mmr, type(None)):
            self.gas_mmr = {igas:None for igas in condensibles}
        else:
            self.gas_mmr = gas_mmr

    def constants(self):
        #   Depth of the Lennard-Jones potential well for the atmosphere
        # Used in the viscocity calculation (units are K) (Rosner, 2000)
        #   (78.6 for air, 71.4 for N2, 59.7 for H2)
        self.eps_k = 59.7
        #   diameter of atmospheric molecule (cm) (Rosner, 2000)
        #   (3.711e-8 for air, 3.798e-8 for N2, 2.827e-8 for H2)
        self.d_molecule = 2.827e-8

        # specific heat factor of the atmosphere
        # 7/2 comes from ideal gas assumption of permanent diatomic gas
        # e.g. h2, o2, n2, air, no, co
        # this technically does increase slowly toward higher temperatures (>~700K)
        self.c_p_factor = 7.0 / 2.0

        self.R_GAS = 8.3143e7
        self.AVOGADRO = 6.02e23
        self.K_BOLTZ = self.R_GAS / self.AVOGADRO

    def ptk(
        self,
        df=None,
        filename=None,
        kz_min=1e5,
        constant_kz=None,
        latent_heat=False,
        convective_overshoot=None,
        Teff=None,
        alpha_pressure=None,
        **pd_kwargs,
    ):
        """
        Read in file or define dataframe.

        Parameters
        ----------
        df : dataframe or dict
            Dataframe with "pressure"(bars),"temperature"(K). MUST have at least two
            columns with names "pressure" and "temperature".
            Optional columns include the eddy diffusion "kz" in cm^2/s CGS units, and
            the convective heat flux 'chf' also in cgs (e.g. sigma_csg T^4)
        filename : str
            Filename read in. Will be read in with pd.read_csv and should
            result in two named headers "pressure"(bars),"temperature"(K).
            Optional columns include the eddy diffusion "kz" in cm^2/s CGS units, and
            the convective heat flux 'chf' also in cgs (e.g. sigma_csg T^4)
            Use pd_kwargs to ensure file is read in properly.
        kz_min : float, optional
            Minimum Kz value. This will reset everything below kz_min to kz_min.
            Default = 1e5 cm2/s
        constant_kz : float, optional
            Constant value for kz, if kz is supplied in df or filename,
            it will inheret that value and not use this constant_value
            Default = None
        latent_heat : bool
            optional, Default = False. The latent heat factors into the mixing length.
            When False, the mixing length goes as the scale height
            When True, the mixing length is scaled by the latent heat
        convective_overshoot : float
            Optional, Default is None. But the default value used in
            Ackerman & Marley 2001 is 1./3. If you are unsure of what to pick, start
            there. This is only used when the
            This is ONLY used when a chf (convective heat flux) is supplied
        Teff : float, optional
            Effective temperature. If None (default), Teff set to temperature at 1 bar
        alpha_pressure : float
            Pressure at which we want fsed=alpha for variable fsed calculation
        pd_kwargs : kwargs
            Pandas key words for file read in.
            If reading old style eddysed files, you would need:
            skiprows=3, delim_whitespace=True, header=None, names=["ind","pressure","temperature","kz"]
        """
        # first read in dataframe, dict or file and sort by pressure
        if not isinstance(df, type(None)):
            if isinstance(df, dict):
                df = pd.DataFrame(df)
            df = df.sort_values("pressure")
        elif not isinstance(filename, type(None)):
            df = pd.read_csv(filename, **pd_kwargs)
            df = df.sort_values("pressure")

        # convert bars to dyne/cm^2
        self.p_level = np.array(df["pressure"]) * 1e6
        self.t_level = np.array(df["temperature"])
        print(f"P-T Profile {len(self.p_level) = }, {len(self.t_level) = }")
        print(self.p_level)
        print(self.t_level)
        plt.yscale("log")
        plt.gca().invert_yaxis()
        plt.plot(self.t_level[::-1], self.p_level[::-1])
        plt.show()
        if alpha_pressure is None:
            self.alpha_pressure = min(df["pressure"])
        else:
            self.alpha_pressure = alpha_pressure
        self.get_atmo_parameters()
        self.get_kz_mixl(df, constant_kz, latent_heat, convective_overshoot, kz_min)

        # Teff
        if isinstance(Teff, type(None)):
            onebar = (np.abs(self.p_level / 1e6 - 1.0)).argmin()
            self.Teff = self.t_level[onebar]
        else:
            self.Teff = Teff

    def get_atmo_parameters(self):
        """Defines all the atmospheric parameters needed for the calculation

        Note: Some of this is repeated in the layer() function.
        This is done on purpose because layer is used to get a "virtual"
        layer off the user input's grid. These parameters are also
        needed, though, to get the initial mixing parameters from the user.
        Therefore, we put them in both places.
        """
        #   specific gas constant for atmosphere (erg/K/g)
        self.r_atmos = self.R_GAS / self.mmw

        # pressure thickess
        dlnp = np.log(self.p_level[1:] / self.p_level[0:-1])  # USED IN LAYER
        self.p_layer = 0.5 * (self.p_level[1:] + self.p_level[0:-1])  # USED IN LAYER

        #   temperature gradient - we use this for the sub layering
        self.dtdlnp = (self.t_level[0:-1] - self.t_level[1:]) / dlnp  # USED IN LAYER

        # get temperatures at layers
        self.t_layer = (
            self.t_level[1:] + np.log(self.p_level[1:] / self.p_layer) * self.dtdlnp
        )  # USED IN LAYER

        # lapse ratio used for kz calculation if user asks for
        # us ot calculate it based on convective heat flux
        self.lapse_ratio = (
            (self.t_level[1:] - self.t_level[0:-1])
            / dlnp
            / (self.t_layer / self.c_p_factor)
        )

        #   atmospheric density (g/cm^3)
        self.rho_atmos = self.p_layer / (self.r_atmos * self.t_layer)

        # scale height (todo=make this gravity dependent)
        self.scale_h = self.r_atmos * self.t_layer / self.g

        # specific heat of atmosphere
        self.c_p = self.c_p_factor * self.r_atmos

        # get altitudes
        self.dz_pmid = self.scale_h * np.log(self.p_level[1:] / self.p_layer)

        self.dz_layer = self.scale_h * dlnp

        self.z_top = np.concatenate(([0], np.cumsum(self.dz_layer[::-1])))[::-1]

        self.z = self.z_top[1:] + self.dz_pmid

        # altitude to set fsed = alpha
        p_alpha = find_nearest_1d(self.p_layer / 1e6, self.alpha_pressure)
        z_temp = np.cumsum(self.dz_layer[::-1])[::-1]
        self.z_alpha = z_temp[p_alpha]

    def get_kz_mixl(self, df, constant_kz, latent_heat, convective_overshoot, kz_min):
        """
        Computes kz profile and mixing length given user input. In brief the options are:

        1) Input Kz
        2) Input constant kz
        3) Input convective heat flux (supply chf in df)
        3a) Input convective heat flux, correct for latent heat (supply chf in df and set latent_heat=True)
        and/or 3b) Input convective heat flux, correct for convective overshoot (supply chf, convective_overshoot=1/3)
        4) Set kz_min to prevent kz from going too low (any of the above and set kz_min~1e5)

        Parameters
        ----------
        df : dataframe or dict
            Dataframe from input with "pressure"(bars),"temperature"(K). MUST have at least two
            columns with names "pressure" and "temperature".
            Optional columns include the eddy diffusion "kz" in cm^2/s CGS units, and
            the convective heat flux 'chf' also in cgs (e.g. sigma_csg T^4)
        constant_kz : float
            Constant value for kz, if kz is supplied in df or filename,
            it will inheret that value and not use this constant_value
            Default = None
        latent_heat : bool
            optional, Default = False. The latent heat factors into the mixing length.
            When False, the mixing length goes as the scale height
            When True, the mixing length is scaled by the latent heat
        convective_overshoot : float
            Optional, Default is None. But the default value used in
            Ackerman & Marley 2001 is 1./3. If you are unsure of what to pick, start
            there. This is only used when the
            This is ONLY used when a chf (convective heat flux) is supplied
        kz_min : float
            Minimum Kz value. This will reset everything below kz_min to kz_min.
            Default = 1e5 cm2/s
        """

        # MIXING LENGTH ASSUMPTIONS
        if latent_heat:
            #   convective mixing length scale (cm): no less than 1/10 scale height
            self.mixl = (
                np.array([np.max([0.1, ilr]) for ilr in self.lapse_ratio])
                * self.scale_h
            )
        else:
            # convective mixing length is the scale height
            self.mixl = 1 * self.scale_h

        # KZ OPTIONS

        #   option 1) the user has supplied it in their file or dictionary
        if "kz" in df.keys():
            if df.loc[df["kz"] < kz_min].shape[0] > 0:
                df.loc[df["kz"] < kz_min] = kz_min
                if self.verbose:
                    print(
                        "Overwriting some Kz values to minimum value set by kz_min \n \
                    You can always turn off these warnings by setting verbose=False"
                    )
            kz_level = np.array(df["kz"])
            self.kz = 0.5 * (kz_level[1:] + kz_level[0:-1])
            self.chf = None

        #   option 2) the user wants a constant value
        elif not isinstance(constant_kz, type(None)):
            self.kz = np.zeros(df.shape[0] - 1) + constant_kz
            self.chf = None

        #   option 3) the user wants to compute kz based on a convective heat flux
        elif "chf" in df.keys():
            self.chf = np.array(df["chf"])

            # CONVECTIVE OVERSHOOT ON OR OFF
            #     sets the minimum allowed heat flux in a layer by assuming some overshoot
            #     the default value of 1/3 is arbitrary, allowing convective flux to fall faster than
            #     pressure scale height
            if not isinstance(convective_overshoot, type(None)):
                used = False
                nz = len(self.p_layer)
                for iz in range(nz - 1, -1, -1):
                    ratio_min = (
                        (convective_overshoot) * self.p_level[iz] / self.p_level[iz + 1]
                    )
                    if self.chf[iz] < ratio_min * self.chf[iz + 1]:
                        self.chf[iz] = self.chf[iz + 1] * ratio_min
                        used = True
                if self.verbose:
                    print(
                        """Convective overshoot was turned on. The convective heat flux
                    has been adjusted such that it is not allowed to decrease more than {0}
                    the pressure. This number is set with the convective_overshoot parameter.
                    It can be disabled with convective_overshoot=None. To turn
                    off these messages set verbose=False in Atmosphere""".format(
                            convective_overshoot
                        )
                    )

            #   vertical eddy diffusion coefficient (cm^2/s)
            #   from Gierasch and Conrath (1985)
            gc_kzz = (
                (1.0 / 3.0)
                * self.scale_h
                * (self.mixl / self.scale_h) ** (4.0 / 3.0)
                * ((self.r_atmos * self.chf[1:]) / (self.rho_atmos * self.c_p))
                ** (1.0 / 3.0)
            )

            self.kz = [np.max([i, kz_min]) for i in gc_kzz]
        else:
            raise Exception(
                "Users can define kz by: \n \
            1) Adding 'kz' as a column or key to your dataframe dict, or file \n \
            2) Defining constant-w-altitude kz through the constant_kz input \n  \
            3) Adding 'chf', the conective heat flux as a column to your \
            dataframe, dict or file."
            )

    def gravity(
        self,
        gravity=None,
        gravity_unit=None,
        radius=None,
        radius_unit=None,
        mass=None,
        mass_unit=None,
    ):
        """
        Get gravity based on mass and radius, or gravity inputs

        Parameters
        ----------
        gravity : float
            (Optional) Gravity of planet
        gravity_unit : astropy.unit
            (Optional) Unit of Gravity
        radius : float
            (Optional) radius of planet MUST be specified for thermal emission!
        radius_unit : astropy.unit
            (Optional) Unit of radius
        mass : float
            (Optional) mass of planet
        mass_unit : astropy.unit
            (Optional) Unit of mass
        """
        if (mass is not None) and (radius is not None):
            m = (mass * mass_unit).to(u.g)
            r = (radius * radius_unit).to(u.cm)
            g = (c.G.cgs * m / (r**2)).value
            self.g = g
            self.gravity_unit = "cm/(s**2)"
        elif gravity is not None:
            g = (gravity * gravity_unit).to("cm/(s**2)")
            g = g.value
            self.g = g
            self.gravity_unit = "cm/(s**2)"
        else:
            raise Exception(
                "Need to specify gravity or radius and mass + additional units"
            )

    def kz(self, df=None, constant_kz=None, chf=None, kz_min=1e5, latent_heat=False):
        """
        Define Kz in CGS. Should be on same grid as pressure. This overwrites whatever was
        defined in get_pt ! Users can define kz by:
            1) Defining a DataFrame with keys 'pressure' (in bars), and 'kz'
            2) Defining constant kz
            3) Supplying a convective heat flux and prescription for latent_heat

        Parameters
        ----------
        df : pandas.DataFrame, dict
            Dataframe or dictionary with 'kz' as one of the fields.
        constant_kz : float
            Constant value for kz in units of cm^2/s
        chf : ndarray
            Convective heat flux in cgs units (e.g. sigma T^4). This will be used to compute
            the kzz using the methodology of Gierasch and Conrath (1985)
        latent_heat : bool
            optional, Default = False. The latent heat factors into the mixing length.
            When False, the mixing length goes as the scale height
            When True, the mixing length is scaled by the latent heat
            This is ONLY used when a chf (convective heat flux) is supplied
        """
        return "Depricating this function. Please use ptk instead. It has identical functionality."
        if not isinstance(df, type(None)):
            # will not need any convective heat flux
            self.chf = None
            # reset to minimun value if specified by the user
            if df.loc[df["kz"] < kz_min].shape[0] > 0:
                df.loc[df["kz"] < kz_min] = kz_min
                print("Overwriting some Kz values to minimum value set by kz_min")
            self.kz = np.array(df["kz"])
            # make sure pressure and kz are the same size
            if len(self.kz) != len(self.pressure):
                raise Exception("Kzz and pressure are not the same length")

        elif not isinstance(constant_kz, type(None)):
            # will not need any convective heat flux
            self.chf = None
            self.kz = constant_kz
            if self.kz < kz_min:
                self.kz = kz_min
                print("Overwriting kz constant value to minimum value set by kz_min")

        elif not isinstance(chf, type(None)):

            def g_c_85(scale_h, r_atmos, chf, rho_atmos, c_p, lapse_ratio):
                #   convective mixing length scale (cm): no less than 1/10 scale height
                if latent_heat:
                    mixl = np.max(0.1, lapse_ratio) * scale_h
                else:
                    mixl = scale_h
                #   vertical eddy diffusion coefficient (cm^2/s)
                #   from Gierasch and Conrath (1985)
                gc_kzz = (
                    (1.0 / 3.0)
                    * scale_h
                    * (mixl / scale_h) ** (4.0 / 3.0)
                    * ((r_atmos * chf) / (rho_atmos * c_p)) ** (1.0 / 3.0)
                )
                return np.max(gc_kzz, kz_min), mixl

            self.kz = g_c_85
            self.chf = chf

    def compute(self, directory=None, as_dict=True):
        """
        Parameters
        ----------
        atmo : class
            `Atmosphere` class
        directory : str, optional
            Directory string that describes where refrind files are
        as_dict : bool
            Default = True, option to view full output as dictionary

        Returns
        -------
        dict
            When as_dict=True. Dictionary output that contains full output. See tutorials for explanation of all output.
        opd, w0, g0
            Extinction per layer, single scattering abledo, asymmetry parameter,
            All are ndarrays that are nlayer by nwave
        """
        print("??????????")
        run = compute(self, directory=directory, as_dict=as_dict)
        return run

    def compute_yasf(self):
        run = compute_yasf(self)
        return run


def compute_yasf(
    atmo: Atmosphere,
    directory=None,
    og_vfall=True,
    particle_props: Particle = Particle(),
    mode = "YASF",
    store_scat_props = False,
    load_scat_props = True,
):
    """
    Just like `compute`, but using YASF for numerical light scattering of fractal particles.
    """
    results = {}

    mmw = atmo.mmw
    mh = atmo.mh
    condensibles = atmo.condensibles

    ngas = len(condensibles)

    gas_mw = np.zeros(ngas)
    gas_mmr = np.zeros(ngas)
    rho_p = np.zeros(ngas)

    # scale-height for fsed taken at Teff (default: temp at 1bar)
    H = atmo.r_atmos * atmo.Teff / atmo.g


    # Next, calculate size and concentration
    # of condensates in balance between eddy diffusion and sedimentation

    # qc = condensate mixing ratio, qt = condensate+gas mr, rg = mean radius,
    # reff = droplet eff radius, ndz = column dens of condensate,
    # qc_path = vertical path of condensate

    fsed_in = atmo.fsed

    assert directory != None , "Need a directory for now"
    rmin, nradii = get_radii_tentatively(directory, condensibles[0])
    # TODO: alternative selection of radii


    results["condensibles"] = condensibles
    for i, igas in zip(range(ngas), condensibles):
        run_gas = getattr(gas_properties, igas)
        gas_mw[i], gas_mmr[i], rho_p[i] = run_gas(mmw, mh=mh, gas_mmr=atmo.gas_mmr[igas])

        # TODO: currently only works with precalculated values
        # qext_test, qscat_test, g_qscat_test, radius_test, wave_in_test = calc_mie_db(
        #     [igas], directory, directory, rmin=1e-5, nradii=10
        # )
        # determine which radii to use, might wanna move this somewhere else


        radii, _, _ = get_r_grid(rmin, n_radii=nradii)
        # comment out for faster testing
        # radii = radii[-2:-1]

        particle_properties = Particle(list(radii),particle_props.monomer_size, particle_props.Df, particle_props.kf)
        if mode == "YASF":
            print(f"I WILL BUILD A PARTICLE WITH {particle_properties.N} monomers!!")

        # TODO: Adjust inputs here!
        # TODO: Add func for MMF here aswell
        if not load_scat_props:
            qext_gas, qscat_gas, cos_qscat_gas, nwave, radius, wave_in = calc_scattering(particle_properties, igas, directory, mode=mode, store=store_scat_props)
        else:
            qext_gas, qscat_gas, cos_qscat_gas, nwave, radius, wave_in = load_stored_fractal_scat_props(gas_name=igas, mode=mode)

        print(f"{qext_gas = }")
        print(f"{qscat_gas = }")
        print(f"{nwave = }")
        print(f"{radius = }")

        if i == 0:
            nradii = len(radius)
            rmin = np.min(radius)
            results["rmin"] = rmin
            radius, rup, dr = get_r_grid(rmin, n_radii=nradii)
            qext = np.zeros((nwave, nradii, ngas))
            qscat = np.zeros((nwave, nradii, ngas))
            cos_qscat = np.zeros((nwave, nradii, ngas))

        # add to master matrix that contains the per gas Mie stuff
        qext[:, :, i], qscat[:, :, i], cos_qscat[:, :, i] = (
            qext_gas,
            qscat_gas,
            cos_qscat_gas,
        )

    z_cld = None  # temporary fix

    qc, qt, rg, reff, ndz, qc_path, mixl, z_cld = eddysed_fractal(
        atmo.t_level,
        atmo.p_level,
        atmo.t_layer,
        atmo.p_layer,
        condensibles,
        gas_mw,
        gas_mmr,
        rho_p,
        mmw,
        atmo.g,
        atmo.kz,
        atmo.mixl,
        fsed_in,
        atmo.b,
        atmo.eps,
        atmo.scale_h,
        atmo.z_top,
        atmo.z_alpha,
        min(atmo.z),
        atmo.param,
        mh,
        atmo.sig,
        rmin,
        nradii,
        atmo.d_molecule,
        atmo.eps_k,
        atmo.c_p_factor,
        og_vfall,
        supsat=atmo.supsat,
        verbose=atmo.verbose,
        do_virtual=True, # TODO: make this available in function as arg
        r_mon=particle_properties.monomer_size,
        Df=particle_properties.Df,
        kf=particle_properties.kf,
    )
    pres_out = atmo.p_layer
    temp_out = atmo.t_layer
    z_out = atmo.z


    print("Starting optical calculations")
    opd, w0, g0, opd_gas = calc_optics(
        nwave,
        qc,
        qt,
        rg,
        reff,
        ndz,
        radius,
        dr,
        qext,
        qscat,
        cos_qscat,
        atmo.sig,
        rmin,
        nradii,
        verbose=False,
    )

    if atmo.param == "exp":
        fsed_out = fsed_in * np.exp((atmo.z - atmo.z_alpha) / atmo.b) + atmo.eps
    else:
        fsed_out = fsed_in
    return create_dict(
        qc,
        qt,
        rg,
        reff,
        ndz,
        opd,
        w0,
        g0,
        opd_gas,
        wave_in,
        pres_out,
        temp_out,
        condensibles,
        mh,
        mmw,
        fsed_out,
        atmo.sig,
        nradii,
        rmin,
        z_out,
        atmo.dz_layer,
        mixl,
        atmo.kz,
        atmo.scale_h,
        z_cld,
    )

def get_radii_tentatively(directory: str, gas: str):
    df = pd.read_csv(
        os.path.join(directory, gas + ".mieff"),
        names=["wave", "qscat", "qext", "cos_qscat"],
        delim_whitespace=True,
    )

    nradii = int(df.iloc[0, 1])

    radii = df.loc[np.isnan(df["qscat"])]["wave"].values
    rmin = np.min(radii)
    nradii = len(radii)
    return rmin, nradii

def export_results(atmo: Atmosphere,fsed_in, results:dict,as_dict: bool=True):
    if as_dict:
        if atmo.param == "exp":
            fsed_out = fsed_in * np.exp((atmo.z - atmo.z_alpha) / atmo.b) + atmo.eps
        else:
            fsed_out = fsed_in
        return create_dict(
            results["qc"],
            results["qt"],
            results["rg"],
            results["reff"],
            results["ndz"],
            results["opd"],
            results["w0"],
            results["g0"],
            results["opd_gas"],
            results["wave_in"],
            results["pres_out"],
            results["temp_out"],
            results["condensibles"],
            results["mh"],
            results["mmw"],
            fsed_out,
            atmo.sig,
            results["nradii"],
            results["rmin"],
            results["z_out"],
            atmo.dz_layer,
            results["mixl"],
            atmo.kz,
            atmo.scale_h,
            results["z_cld"],
        )
    else:
        return results["opd"], results["w0"], results["g0"],

def compute(
    atmo,
    directory=None,
    as_dict=True,
    og_solver=True,
    direct_tol=1e-15,
    refine_TP=True,
    og_vfall=True,
    analytical_rg=True,
    do_virtual=True,
):
    """
    Top level program to run eddysed. Requires running `Atmosphere` class
    before running this.

    Parameters
    ----------
    atmo : class
        `Atmosphere` class
    directory : str, optional
        Directory string that describes where refrind files are
    as_dict : bool, optional
        Default = False. Option to view full output as dictionary
    og_solver : bool, optional
        Default=True. BETA. Contact developers before changing to False.
         Option to change mmr solver (True = original eddysed, False = new direct solver)
    direct_tol : float , optional
        Only used if og_solver =False. Default = True.
        Tolerance for direct solver
    refine_TP : bool, optional
        Only used if og_solver =False.
        Option to refine temperature-pressure profile for direct solver
    og_vfall : bool, optional
        Option to use original A&M or new Khan-Richardson method for finding vfall
    analytical_rg : bool, optional
        Only used if og_solver =False.
        Option to use analytical expression for rg, or alternatively deduce rg from calculation
        Calculation option will be most useful for future
        inclusions of alternative particle size distributions
    do_virtual : bool
        If the user adds an upper bound pressure that is too low. There are cases where a cloud wants to
        form off the grid towards higher pressures. This enables that.

    Returns
    -------
    opd, w0, g0
        Extinction per layer, single scattering abledo, asymmetry parameter,
        All are ndarrays that are nlayer by nwave
    dict
        Dictionary output that contains full output. See tutorials for explanation of all output.
    """
    mmw = atmo.mmw
    mh = atmo.mh
    condensibles = atmo.condensibles

    ngas = len(condensibles)

    gas_mw = np.zeros(ngas)
    gas_mmr = np.zeros(ngas)
    rho_p = np.zeros(ngas)

    # scale-height for fsed taken at Teff (default: temp at 1bar)
    H = atmo.r_atmos * atmo.Teff / atmo.g

    #### First we need to either grab or compute Mie coefficients ####
    print(ngas)
    print(condensibles)
    for i, igas in zip(range(ngas), condensibles):
        # Get gas properties including gas mean molecular weight,
        # gas mixing ratio, and the density
        run_gas = getattr(gas_properties, igas)
        gas_mw[i], gas_mmr[i], rho_p[i] = run_gas(mmw, mh=mh, gas_mmr=atmo.gas_mmr)

        # Get mie files that are already saved in
        # directory
        # eventually we will replace this with nice database

        qext_test, qscat_test, g_qscat_test, radius_test, wave_in_test = calc_mie_db(
            [igas], directory, directory, rmin=1e-5, nradii=10
        )

        qext_gas, qscat_gas, cos_qscat_gas, nwave, radius, wave_in = get_mie(
            igas, directory
        )
        print("PRE:")
        print(radius)
        if i == 0:
            nradii = len(radius)
            rmin = np.min(radius)
            radius, rup, dr = get_r_grid(rmin, n_radii=nradii)
            qext = np.zeros((nwave, nradii, ngas))
            qscat = np.zeros((nwave, nradii, ngas))
            cos_qscat = np.zeros((nwave, nradii, ngas))

        print("POST:")
        print(radius)
        # add to master matrix that contains the per gas Mie stuff
        qext[:, :, i], qscat[:, :, i], cos_qscat[:, :, i] = (
            qext_gas,
            qscat_gas,
            cos_qscat_gas,
        )

    # Next, calculate size and concentration
    # of condensates in balance between eddy diffusion and sedimentation

    # qc = condensate mixing ratio, qt = condensate+gas mr, rg = mean radius,
    # reff = droplet eff radius, ndz = column dens of condensate,
    # qc_path = vertical path of condensate

    #   run original eddysed code
    if og_solver:
        # here atmo.param describes the parameterization used for the variable fsed methodology
        if atmo.param == "exp":
            # the formalism of this is detailed in Rooney et al. 2021
            atmo.b = 6 * atmo.b * H  # using constant scale-height in fsed
            fsed_in = atmo.fsed - atmo.eps
        elif atmo.param == "const":
            fsed_in = atmo.fsed

        # @dusc: FeelsGoodMan Clap
        qc, qt, rg, reff, ndz, qc_path, mixl, z_cld = eddysed(
            atmo.t_level,
            atmo.p_level,
            atmo.t_layer,
            atmo.p_layer,
            condensibles,
            gas_mw,
            gas_mmr,
            rho_p,
            mmw,
            atmo.g,
            atmo.kz,
            atmo.mixl,
            fsed_in,
            atmo.b,
            atmo.eps,
            atmo.scale_h,
            atmo.z_top,
            atmo.z_alpha,
            min(atmo.z),
            atmo.param,
            mh,
            atmo.sig,
            rmin,
            nradii,
            atmo.d_molecule,
            atmo.eps_k,
            atmo.c_p_factor,
            og_vfall,
            supsat=atmo.supsat,
            verbose=atmo.verbose,
            do_virtual=do_virtual,
        )
        pres_out = atmo.p_layer
        temp_out = atmo.t_layer
        z_out = atmo.z

    #   run new, direct solver
    else:
        fsed_in = atmo.fsed
        z_cld = None  # temporary fix
        qc, qt, rg, reff, ndz, qc_path, pres_out, temp_out, z_out, mixl = direct_solver(
            atmo.t_layer,
            atmo.p_layer,
            condensibles,
            gas_mw,
            gas_mmr,
            rho_p,
            mmw,
            atmo.g,
            atmo.kz,
            atmo.fsed,
            mh,
            atmo.sig,
            rmin,
            nradii,
            atmo.d_molecule,
            atmo.eps_k,
            atmo.c_p_factor,
            direct_tol,
            refine_TP,
            og_vfall,
            analytical_rg,
        )

    # Finally, calculate spectrally-resolved profiles of optical depth, single-scattering
    # albedo, and asymmetry parameter.
    print("Starting optical calculations")
    opd, w0, g0, opd_gas = calc_optics(
        nwave,
        qc,
        qt,
        rg,
        reff,
        ndz,
        radius,
        dr,
        qext,
        qscat,
        cos_qscat,
        atmo.sig,
        rmin,
        nradii,
        verbose=False,
    )

    if as_dict:
        if atmo.param == "exp":
            fsed_out = fsed_in * np.exp((atmo.z - atmo.z_alpha) / atmo.b) + atmo.eps
        else:
            fsed_out = fsed_in
        return create_dict(
            qc,
            qt,
            rg,
            reff,
            ndz,
            opd,
            w0,
            g0,
            opd_gas,
            wave_in,
            pres_out,
            temp_out,
            condensibles,
            mh,
            mmw,
            fsed_out,
            atmo.sig,
            nradii,
            rmin,
            z_out,
            atmo.dz_layer,
            mixl,
            atmo.kz,
            atmo.scale_h,
            z_cld,
        )
    else:
        return opd, w0, g0


def create_dict(
    qc,
    qt,
    rg,
    reff,
    ndz,
    opd,
    w0,
    g0,
    opd_gas,
    wave,
    pressure,
    temperature,
    gas_names,
    mh,
    mmw,
    fsed,
    sig,
    nrad,
    rmin,
    z,
    dz_layer,
    mixl,
    kz,
    scale_h,
    z_cld,
):
    if len(wave.shape) < 2:
        wave = wave[:,np.newaxis]
    return {
        "pressure": pressure / 1e6,
        "pressure_unit": "bar",
        "temperature": temperature,
        "temperature_unit": "kelvin",
        "wave_in": wave[:, 0],
        "wave_unit": "micron",
        "condensate_mmr": qc,
        "cond_plus_gas_mmr": qt,
        "mean_particle_r": rg * 1e4,
        "droplet_eff_r": reff * 1e4,
        "r_units": "micron",
        "column_density": ndz,
        "column_density_unit": "#/cm^2",
        "opd_per_layer": opd,
        "single_scattering": w0,
        "asymmetry": g0,
        "opd_by_gas": opd_gas,
        "condensibles": gas_names,
        "mh": mh,
        # "scalar_inputs": {'mh':mh, 'mmw':mmw,'fsed':fsed, 'sig':sig,'nrad':nrad,'rmin':rmin},
        "scalar_inputs": {"mh": mh, "mmw": mmw, "sig": sig, "nrad": nrad, "rmin": rmin},
        "fsed": fsed,
        "altitude": z,
        "layer_thickness": dz_layer,
        "z_unit": "cm",
        "mixing_length": mixl,
        "mixing_length_unit": "cm",
        "kz": kz,
        "kz_unit": "cm^2/s",
        "scale_height": scale_h,
        "cloud_deck": z_cld,
    }


def calc_optics(
    nwave,
    qc,
    qt,
    rg,
    reff,
    ndz,
    radius,
    dr,
    qext,
    qscat,
    cos_qscat,
    sig,
    rmin,
    nrad,
    verbose=False,
):
    """
    Calculate spectrally-resolved profiles of optical depth, single-scattering
    albedo, and asymmetry parameter.

    Parameters
    ----------
    nwave : int
        Number of wave points
    qc : ndarray
        Condensate mixing ratio
    qt : ndarray
        Gas + condensate mixing ratio
    rg : ndarray
        Geometric mean radius of condensate
    reff : ndarray
        Effective (area-weighted) radius of condensate (cm)
    ndz : ndarray
        Column density of particle concentration in layer (#/cm^2)
    radius : ndarray
        Radius bin centers (cm)
    dr : ndarray
        Width of radius bins (cm)
    qscat : ndarray
        Scattering efficiency
    qext : ndarray
        Extinction efficiency
    cos_qscat : ndarray
        qscat-weighted <cos (scattering angle)>
    sig : float
        Width of the log normal particle distribution
    verbose: bool
        print out warnings or not


    Returns
    -------
    opd : ndarray
        extinction optical depth due to all condensates in layer
    w0 : ndarray
        single scattering albedo
    g0 : ndarray
        asymmetry parameter = Q_scat wtd avg of <cos theta>
    opd_gas : ndarray
        cumulative (from top) opd by condensing vapor as geometric conservative scatterers
    """

    PI = np.pi
    nz = qc.shape[0]
    ngas = qc.shape[1]
    nrad = len(radius)

    opd_layer = np.zeros((nz, ngas))
    scat_gas = np.zeros((nz, nwave, ngas))
    ext_gas = np.zeros((nz, nwave, ngas))
    cqs_gas = np.zeros((nz, nwave, ngas))
    opd = np.zeros((nz, nwave))
    opd_gas = np.zeros((nz, ngas))
    w0 = np.zeros((nz, nwave))
    g0 = np.zeros((nz, nwave))
    warning = ""
    for iz in range(nz):
        for igas in range(ngas):
            # Optical depth for conservative geometric scatterers
            if ndz[iz, igas] > 0:
                if np.log10(rg[iz, igas]) < np.log10(rmin) + 0.75 * sig:
                    warning0 = f"Take caution in analyzing results. There have been a calculated particle radii off the Mie grid, which has a min radius of {rmin}cm and distribution of {sig}. The following errors:"
                    warning += (
                        "{0}cm for the {1}th gas at the {2}th grid point; ".format(
                            str(rg[iz, igas]), str(igas), str(iz)
                        )
                    )

                r2 = rg[iz, igas] ** 2 * np.exp(2 * np.log(sig) ** 2)
                opd_layer[iz, igas] = 2.0 * PI * r2 * ndz[iz, igas]

                #  Calculate normalization factor (forces lognormal sum = 1.0)
                # TODO: @dusc: whats this
                rsig = sig
                norm = 0.0
                for irad in range(nrad):
                    rr = radius[irad]
                    arg1 = dr[irad] / (np.sqrt(2.0 * PI) * rr * np.log(rsig))
                    arg2 = -np.log(rr / rg[iz, igas]) ** 2 / (2 * np.log(rsig) ** 2) # lognormal dist
                    norm = norm + arg1 * np.exp(arg2)
                    # print (rr, rg[iz,igas],rsig,arg1,arg2)

                # normalization
                norm = ndz[iz, igas] / norm

                # @dusc: this is some sort of integration over the individual layers using the particle distributions
                # and number densities of particles
                for irad in range(nrad):
                    rr = radius[irad]
                    arg1 = dr[irad] / (np.sqrt(2.0 * PI) * np.log(rsig))
                    # print(f"{arg1 = }")
                    arg2 = -np.log(rr / rg[iz, igas]) ** 2 / (2 * np.log(rsig) ** 2)
                    # print(f"{arg2 = }")
                    # TODO: have a look at this, whats the value of pir2ndz?
                    # if its tiny for the very large particles we can limit the number of required radii to calc
                    pir2ndz = norm * PI * rr * arg1 * np.exp(arg2) # rr*pi* PDF, what is this?
                    for iwave in range(nwave):
                        scat_gas[iz, iwave, igas] = (
                            scat_gas[iz, iwave, igas]
                            + qscat[iwave, irad, igas] * pir2ndz
                        )
                        ext_gas[iz, iwave, igas] = (
                            ext_gas[iz, iwave, igas] + qext[iwave, irad, igas] * pir2ndz
                        )
                        cqs_gas[iz, iwave, igas] = (
                            cqs_gas[iz, iwave, igas]
                            + cos_qscat[iwave, irad, igas] * pir2ndz
                        )

                    # TO DO ADD IN CLOUD SUBLAYER KLUGE LATER

    for igas in range(ngas):
        for iz in range(nz-1,-1,-1):

            if np.sum(ext_gas[iz,:,igas]) > 0:
                ibot = iz
                break
            if iz == 0:
                ibot=0
        #print(igas,ibot)
        if ibot >= nz -2:
            print("Not doing sublayer as cloud deck at the bottom of pressure grid")

        else:
            opd_layer[ibot+1,igas] = opd_layer[ibot,igas]*0.1
            scat_gas[ibot+1,:,igas] = scat_gas[ibot,:,igas]*0.1
            ext_gas[ibot+1,:,igas] = ext_gas[ibot,:,igas]*0.1
            cqs_gas[ibot+1,:,igas] = cqs_gas[ibot,:,igas]*0.1
            opd_layer[ibot+2,igas] = opd_layer[ibot,igas]*0.05
            scat_gas[ibot+2,:,igas] = scat_gas[ibot,:,igas]*0.05
            ext_gas[ibot+2,:,igas] = ext_gas[ibot,:,igas]*0.05
            cqs_gas[ibot+2,:,igas] = cqs_gas[ibot,:,igas]*0.05

        # Sum over gases and compute spectral optical depth profile etc
        for iz in range(nz):
            for iwave in range(nwave):
                opd_scat = 0.0
                opd_ext = 0.0
                cos_qs = 0.0
                for igas in range(ngas):
                    opd_scat = opd_scat + scat_gas[iz, iwave, igas]
                    opd_ext = opd_ext + ext_gas[iz, iwave, igas]
                    cos_qs = cos_qs + cqs_gas[iz, iwave, igas]

                    if opd_scat > 0.0:
                        opd[iz, iwave] = opd_ext
                        w0[iz, iwave] = opd_scat / opd_ext
                        # if w0[iz,iwave]>1:
                        #    w0[iz,iwave]=1.
                        g0[iz, iwave] = cos_qs / opd_scat

    # cumulative optical depths for conservative geometric scatterers
    opd_tot = 0.0

    for igas in range(ngas):
        opd_gas[0, igas] = opd_layer[0, igas]

        for iz in range(1, nz):
            opd_gas[iz, igas] = opd_gas[iz - 1, igas] + opd_layer[iz, igas]
    if (warning != "") & (verbose):
        print(warning0 + warning + " Turn off warnings by setting verbose=False.")
    return opd, w0, g0, opd_gas


def eddysed_fractal(
    t_top,
    p_top,
    t_mid,
    p_mid,
    condensibles,
    gas_mw,
    gas_mmr,
    rho_p,
    mw_atmos,
    gravity,
    kz,
    mixl,
    fsed,
    b,
    eps,
    scale_h,
    z_top,
    z_alpha,
    z_min,
    param,
    mh,
    sig,
    rmin,
    nrad,
    d_molecule,
    eps_k,
    c_p_factor,
    og_vfall=True,
    do_virtual=True,
    supsat=0,
    verbose=True,
    r_mon=0.01,
    Df: float=1.8,
    kf: float=1.0,
):
    """
    Given an atmosphere and condensates, calculate size and concentration
    of condensates in balance between eddy diffusion and sedimentation.

    Parameters
    ----------
    t_top : ndarray
        Temperature at each layer (K)
    p_top : ndarray
        Pressure at each layer (dyn/cm^2)
    t_mid : ndarray
        Temperature at each midpoint (K)
    p_mid : ndarray
        Pressure at each midpoint (dyn/cm^2)
    condensibles : ndarray or list of str
        List or array of condensible gas names
    gas_mw : ndarray
        Array of gas mean molecular weight from `gas_properties`
    gas_mmr : ndarray
        Array of gas mixing ratio from `gas_properties`
    rho_p : float
        density of condensed vapor (g/cm^3)
    mw_atmos : float
        Mean molecular weight of the atmosphere
    gravity : float
        Gravity of planet cgs
    kz : float or ndarray
        Kzz in cgs, either float or ndarray depending of whether or not
        it is set as input
    fsed : float
        Sedimentation efficiency coefficient, unitless
    b : float
        Denominator of exponential in sedimentation efficiency  (if param is 'exp')
    eps: float
        Minimum value of fsed function (if param=exp)
    scale_h : float
        Scale height of the atmosphere
    z_top : float
        Altitude at each layer
    z_alpha : float
        Altitude at which fsed=alpha for variable fsed calculation
    param : str
        fsed parameterisation
        'const' (constant), 'exp' (exponential density derivation)
    mh : float
        Atmospheric metallicity in NON log units (e.g. 1 for 1x solar)
    sig : float
        Width of the log normal particle distribution
    d_molecule : float
        diameter of atmospheric molecule (cm) (Rosner, 2000)
        (3.711e-8 for air, 3.798e-8 for N2, 2.827e-8 for H2)
        Set in Atmosphere constants
    eps_k : float
        Depth of the Lennard-Jones potential well for the atmosphere
        Used in the viscocity calculation (units are K) (Rosner, 2000)
    c_p_factor : float
        specific heat of atmosphere (erg/K/g) . Usually 7/2 for ideal gas
        diatomic molecules (e.g. H2, N2). Technically does slowly rise with
        increasing temperature
    og_vfall : bool , optional
        optional, default = True. True does the original fall velocity calculation.
        False does the updated one which runs a tad slower but is more consistent.
        The main effect of turning on False is particle sizes in the upper atmosphere
        that are slightly bigger.
    do_virtual : bool,optional
        optional, Default = True which adds a virtual layer if the
        species condenses below the model domain.
    supsat : float, optional
        Default = 0 , Saturation factor (after condensation)

    Returns
    -------
    qc : ndarray
        condenstate mixing ratio (g/g)
    qt : ndarray
        gas + condensate mixing ratio (g/g)
    rg : ndarray
        geometric mean radius of condensate  cm
    reff : ndarray
        droplet effective radius (second moment of size distrib, cm)
    ndz : ndarray
        number column density of condensate (cm^-3)
    qc_path : ndarray
        vertical path of condensate
    """
    # default for everything is false, will fill in as True as we go

    did_gas_condense = [False for i in condensibles]
    t_bot = t_top[-1]
    p_bot = p_top[-1]
    z_bot = z_top[-1]
    ngas = len(condensibles)
    nz = len(t_mid)
    qc = np.zeros((nz, ngas))
    qt = np.zeros((nz, ngas))
    rg = np.zeros((nz, ngas))
    reff = np.zeros((nz, ngas))
    ndz = np.zeros((nz, ngas))
    fsed_layer = np.zeros((nz, ngas))
    qc_path = np.zeros(ngas)
    z_cld_out = np.zeros(ngas)

    for i, igas in zip(range(ngas), condensibles):
        q_below = gas_mmr[i]

        # include decrease in condensate mixing ratio below model domain
        if do_virtual:
            z_cld = None
            qvs_factor = (supsat + 1) * gas_mw[i] / mw_atmos
            get_pvap = getattr(pvaps, igas)
            if igas in ['Mg2SiO4','CaTiO3','CaAl12O19','FakeHaze','H2SO4','KhareHaze','SteamHaze300K','SteamHaze400K']:
                pvap = get_pvap(t_bot, p_bot, mh=mh)
            else:
                pvap = get_pvap(t_bot, mh=mh)

            qvs = qvs_factor * pvap / p_bot
            if qvs <= q_below:
                # find the pressure at cloud base
                #   parameters for finding root
                p_lo = p_bot
                p_hi = p_bot * 1e3

                # temperature gradient
                dtdlnp = (t_top[-2] - t_bot) / np.log(p_bot / p_top[-2])

                #   load parameters into qvs_below common block
                qv_dtdlnp = dtdlnp
                qv_p = p_bot
                qv_t = t_bot
                qv_gas_name = igas
                qv_factor = qvs_factor

                try:
                    p_base = optimize.root_scalar(
                        qvs_below_model,
                        bracket=[p_lo, p_hi],
                        method="brentq",
                        args=(
                            qv_dtdlnp,
                            qv_p,
                            qv_t,
                            qv_factor,
                            qv_gas_name,
                            mh,
                            q_below,
                        ),
                    )  # , xtol = 1e-20)

                    if verbose:
                        print("Virtual Cloud Found: " + qv_gas_name)
                    root_was_found = True
                except ValueError:
                    root_was_found = False

                if root_was_found:
                    # Yes, the gas did condense (below the grid)
                    did_gas_condense[i] = True

                    p_base = p_base.root
                    t_base = t_bot + np.log(p_bot / p_base) * dtdlnp
                    z_base = z_bot + scale_h[-1] * np.log(p_bot / p_base)

                    #   Calculate temperature and pressure below bottom layer
                    #   by adding a virtual layer

                    p_layer_virtual = 0.5 * (p_bot + p_base)
                    t_layer_virtual = t_bot + np.log10(p_bot / p_layer_virtual) * dtdlnp

                    # we just need to overwrite
                    # q_below from this output for the next routine
                    (
                        qc_v,
                        qt_v,
                        rg_v,
                        reff_v,
                        ndz_v,
                        q_below,
                        z_cld,
                        fsed_layer_v,
                    ) = layer_fractal(
                        igas,
                        rho_p[i],
                        # t,p layers, then t.p levels below and above
                        t_layer_virtual,
                        p_layer_virtual,
                        t_bot,
                        t_base,
                        p_bot,
                        p_base,
                        kz[-1],
                        mixl[-1],
                        gravity,
                        mw_atmos,
                        gas_mw[i],
                        q_below,
                        supsat,
                        fsed,
                        b,
                        eps,
                        z_bot,
                        z_base,
                        z_alpha,
                        z_min,
                        param,
                        sig,
                        mh,
                        rmin,
                        nrad,
                        d_molecule,
                        eps_k,
                        c_p_factor,  # all scalaers
                        og_vfall,
                        z_cld,
                        r_mon=r_mon,
                        Df=Df,
                        kf=kf,
                    )

        z_cld = None
        for iz in range(nz - 1, -1, -1):  # goes from BOA to TOA
            (
                qc[iz, i],
                qt[iz, i],
                rg[iz, i],
                reff[iz, i],
                ndz[iz, i],
                q_below,
                z_cld,
                fsed_layer[iz, i],
            ) = layer_fractal(
                igas,
                rho_p[i],
                # t,p layers, then t.p levels below and above
                t_mid[iz],
                p_mid[iz],
                t_top[iz],
                t_top[iz + 1],
                p_top[iz],
                p_top[iz + 1],
                kz[iz],
                mixl[iz],
                gravity,
                mw_atmos,
                gas_mw[i],
                q_below,
                supsat,
                fsed,
                b,
                eps,
                z_top[iz],
                z_top[iz + 1],
                z_alpha,
                z_min,
                param,
                sig,
                mh,
                rmin,
                nrad,
                d_molecule,
                eps_k,
                c_p_factor,  # all scalars
                og_vfall,
                z_cld,
                r_mon=r_mon,
                Df=Df,
                kf=kf,
            )

            qc_path[i] = qc_path[i] + qc[iz, i] * (p_top[iz + 1] - p_top[iz]) / gravity
        z_cld_out[i] = z_cld

    return qc, qt, rg, reff, ndz, qc_path, mixl, z_cld_out


def eddysed(
    t_top,
    p_top,
    t_mid,
    p_mid,
    condensibles,
    gas_mw,
    gas_mmr,
    rho_p,
    mw_atmos,
    gravity,
    kz,
    mixl,
    fsed,
    b,
    eps,
    scale_h,
    z_top,
    z_alpha,
    z_min,
    param,
    mh,
    sig,
    rmin,
    nrad,
    d_molecule,
    eps_k,
    c_p_factor,
    og_vfall=True,
    do_virtual=True,
    supsat=0,
    verbose=True,
):
    """
    Given an atmosphere and condensates, calculate size and concentration
    of condensates in balance between eddy diffusion and sedimentation.

    Parameters
    ----------
    t_top : ndarray
        Temperature at each layer (K)
    p_top : ndarray
        Pressure at each layer (dyn/cm^2)
    t_mid : ndarray
        Temperature at each midpoint (K)
    p_mid : ndarray
        Pressure at each midpoint (dyn/cm^2)
    condensibles : ndarray or list of str
        List or array of condensible gas names
    gas_mw : ndarray
        Array of gas mean molecular weight from `gas_properties`
    gas_mmr : ndarray
        Array of gas mixing ratio from `gas_properties`
    rho_p : float
        density of condensed vapor (g/cm^3)
    mw_atmos : float
        Mean molecular weight of the atmosphere
    gravity : float
        Gravity of planet cgs
    kz : float or ndarray
        Kzz in cgs, either float or ndarray depending of whether or not
        it is set as input
    fsed : float
        Sedimentation efficiency coefficient, unitless
    b : float
        Denominator of exponential in sedimentation efficiency  (if param is 'exp')
    eps: float
        Minimum value of fsed function (if param=exp)
    scale_h : float
        Scale height of the atmosphere
    z_top : float
        Altitude at each layer
    z_alpha : float
        Altitude at which fsed=alpha for variable fsed calculation
    param : str
        fsed parameterisation
        'const' (constant), 'exp' (exponential density derivation)
    mh : float
        Atmospheric metallicity in NON log units (e.g. 1 for 1x solar)
    sig : float
        Width of the log normal particle distribution
    d_molecule : float
        diameter of atmospheric molecule (cm) (Rosner, 2000)
        (3.711e-8 for air, 3.798e-8 for N2, 2.827e-8 for H2)
        Set in Atmosphere constants
    eps_k : float
        Depth of the Lennard-Jones potential well for the atmosphere
        Used in the viscocity calculation (units are K) (Rosner, 2000)
    c_p_factor : float
        specific heat of atmosphere (erg/K/g) . Usually 7/2 for ideal gas
        diatomic molecules (e.g. H2, N2). Technically does slowly rise with
        increasing temperature
    og_vfall : bool , optional
        optional, default = True. True does the original fall velocity calculation.
        False does the updated one which runs a tad slower but is more consistent.
        The main effect of turning on False is particle sizes in the upper atmosphere
        that are slightly bigger.
    do_virtual : bool,optional
        optional, Default = True which adds a virtual layer if the
        species condenses below the model domain.
    supsat : float, optional
        Default = 0 , Saturation factor (after condensation)

    Returns
    -------
    qc : ndarray
        condenstate mixing ratio (g/g)
    qt : ndarray
        gas + condensate mixing ratio (g/g)
    rg : ndarray
        geometric mean radius of condensate  cm
    reff : ndarray
        droplet effective radius (second moment of size distrib, cm)
    ndz : ndarray
        number column density of condensate (cm^-3)
    qc_path : ndarray
        vertical path of condensate
    """
    # default for everything is false, will fill in as True as we go

    did_gas_condense = [False for i in condensibles]
    t_bot = t_top[-1]
    p_bot = p_top[-1]
    z_bot = z_top[-1]
    ngas = len(condensibles)
    nz = len(t_mid)
    qc = np.zeros((nz, ngas))
    qt = np.zeros((nz, ngas))
    rg = np.zeros((nz, ngas))
    reff = np.zeros((nz, ngas))
    ndz = np.zeros((nz, ngas))
    fsed_layer = np.zeros((nz, ngas))
    qc_path = np.zeros(ngas)
    z_cld_out = np.zeros(ngas)

    for i, igas in zip(range(ngas), condensibles):
        q_below = gas_mmr[i]

        # include decrease in condensate mixing ratio below model domain
        if do_virtual:
            z_cld = None
            qvs_factor = (supsat + 1) * gas_mw[i] / mw_atmos
            get_pvap = getattr(pvaps, igas)
            if igas in ['Mg2SiO4','CaTiO3','CaAl12O19','FakeHaze','H2SO4','KhareHaze','SteamHaze300K','SteamHaze400K']:
                pvap = get_pvap(t_bot, p_bot, mh=mh)
            else:
                pvap = get_pvap(t_bot, mh=mh)

            qvs = qvs_factor * pvap / p_bot
            if qvs <= q_below:
                # find the pressure at cloud base
                #   parameters for finding root
                p_lo = p_bot
                p_hi = p_bot * 1e3

                # temperature gradient
                dtdlnp = (t_top[-2] - t_bot) / np.log(p_bot / p_top[-2])

                #   load parameters into qvs_below common block
                qv_dtdlnp = dtdlnp
                qv_p = p_bot
                qv_t = t_bot
                qv_gas_name = igas
                qv_factor = qvs_factor

                try:
                    p_base = optimize.root_scalar(
                        qvs_below_model,
                        bracket=[p_lo, p_hi],
                        method="brentq",
                        args=(
                            qv_dtdlnp,
                            qv_p,
                            qv_t,
                            qv_factor,
                            qv_gas_name,
                            mh,
                            q_below,
                        ),
                    )  # , xtol = 1e-20)

                    if verbose:
                        print("Virtual Cloud Found: " + qv_gas_name)
                    root_was_found = True
                except ValueError:
                    root_was_found = False

                if root_was_found:
                    # Yes, the gas did condense (below the grid)
                    did_gas_condense[i] = True

                    p_base = p_base.root
                    t_base = t_bot + np.log(p_bot / p_base) * dtdlnp
                    z_base = z_bot + scale_h[-1] * np.log(p_bot / p_base)

                    #   Calculate temperature and pressure below bottom layer
                    #   by adding a virtual layer

                    p_layer_virtual = 0.5 * (p_bot + p_base)
                    t_layer_virtual = t_bot + np.log10(p_bot / p_layer_virtual) * dtdlnp

                    # we just need to overwrite
                    # q_below from this output for the next routine
                    (
                        qc_v,
                        qt_v,
                        rg_v,
                        reff_v,
                        ndz_v,
                        q_below,
                        z_cld,
                        fsed_layer_v,
                    ) = layer(
                        igas,
                        rho_p[i],
                        # t,p layers, then t.p levels below and above
                        t_layer_virtual,
                        p_layer_virtual,
                        t_bot,
                        t_base,
                        p_bot,
                        p_base,
                        kz[-1],
                        mixl[-1],
                        gravity,
                        mw_atmos,
                        gas_mw[i],
                        q_below,
                        supsat,
                        fsed,
                        b,
                        eps,
                        z_bot,
                        z_base,
                        z_alpha,
                        z_min,
                        param,
                        sig,
                        mh,
                        rmin,
                        nrad,
                        d_molecule,
                        eps_k,
                        c_p_factor,  # all scalaers
                        og_vfall,
                        z_cld,
                    )

        z_cld = None
        for iz in range(nz - 1, -1, -1):  # goes from BOA to TOA
            (
                qc[iz, i],
                qt[iz, i],
                rg[iz, i],
                reff[iz, i],
                ndz[iz, i],
                q_below,
                z_cld,
                fsed_layer[iz, i],
            ) = layer(
                igas,
                rho_p[i],
                # t,p layers, then t.p levels below and above
                t_mid[iz],
                p_mid[iz],
                t_top[iz],
                t_top[iz + 1],
                p_top[iz],
                p_top[iz + 1],
                kz[iz],
                mixl[iz],
                gravity,
                mw_atmos,
                gas_mw[i],
                q_below,
                supsat,
                fsed,
                b,
                eps,
                z_top[iz],
                z_top[iz + 1],
                z_alpha,
                z_min,
                param,
                sig,
                mh,
                rmin,
                nrad,
                d_molecule,
                eps_k,
                c_p_factor,  # all scalars
                og_vfall,
                z_cld,
            )

            qc_path[i] = qc_path[i] + qc[iz, i] * (p_top[iz + 1] - p_top[iz]) / gravity
        z_cld_out[i] = z_cld

    return qc, qt, rg, reff, ndz, qc_path, mixl, z_cld_out


def calc_optics_user_r_dist(wave_in, ndz, radius, radius_unit, r_distribution, qext, qscat ,cos_qscat,  verbose=False):
    """
    Calculate spectrally-resolved profiles of optical depth, single-scattering
    albedo, and asymmetry parameter for a user-input particle radius distribution
    Parameters
    ----------
    wave_in : ndarray
        your wavelength grid in microns
    ndz : float
        Column density of total particle concentration (#/cm^2)
            Note: set to whatever, it's your free knob
            ---- this does not directly translate to something physical because it's for all particles in your slab
            May have to use values of 1e8 or so
    radius : ndarray
        Radius bin values - the range of particle sizes of interest. Maybe measured in the lab,
        Ensure radius_unit is specified
    radius_unit : astropy.unit.Units
        Astropy compatible unit
    qscat : ndarray
        Scattering efficiency
    qext : ndarray
        Extinction efficiency
    cos_qscat : ndarray
        qscat-weighted <cos (scattering angle)>
    r_distribution : ndarray
        the radius distribution in each bin. Maybe measured from the lab, generated from microphysics, etc.
        Should integrate to 1.
    verbose: bool
        print out warnings or not
    Returns
    -------
    opd : ndarray
        extinction optical depth due to all condensates in layer
    w0 : ndarray
        single scattering albedo
    g0 : ndarray
        asymmetry parameter = Q_scat wtd avg of <cos theta>
    """

    radius = (radius*radius_unit).to(u.cm)
    radius = radius.value

    wavenumber_grid = 1e4/wave_in
    wavenumber_grid = np.array([item[0] for item in wavenumber_grid])
    nwave = len(wavenumber_grid)
    PI=np.pi
    nrad = len(radius) ## where radius is the radius grid of the particle size distribution

    scat= np.zeros((nwave))
    ext = np.zeros((nwave))
    cqs = np.zeros((nwave))

    opd = np.zeros((nwave))
    w0 = np.zeros((nwave))
    g0 = np.zeros((nwave))

    opd_scat = 0.
    opd_ext = 0.
    cos_qs = 0.

        #  Calculate normalization factor
    for irad in range(nrad):
            rr = radius[irad] # the get the radius at each grid point, this is in nanometers

            each_r_bin = ndz * (r_distribution[irad]) # weight the radius bin by the distribution
            pir2ndz = PI * rr**2 * each_r_bin # find the weighted cross section

            for iwave in range(nwave):
                scat[iwave] = scat[iwave] + qscat[iwave,irad]*pir2ndz
                ext[iwave] = ext[iwave] + qext[iwave,irad]*pir2ndz
                cqs[iwave] = cqs[iwave] + cos_qscat[iwave,irad]*pir2ndz


                    # calculate the spectral optical depth profile etc
    for iwave in range(nwave):
            opd_scat = 0.
            opd_ext = 0.
            cos_qs = 0.

            opd_scat = opd_scat + scat[iwave]
            opd_ext = opd_ext + ext[iwave]
            cos_qs = cos_qs + cqs[iwave]


            if( opd_scat > 0. ):
                            opd[iwave] = opd_ext
                            w0[iwave] = opd_scat / opd_ext
                            g0[iwave] = cos_qs / opd_scat

    return opd, w0, g0, wavenumber_grid
