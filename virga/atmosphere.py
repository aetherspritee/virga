#!/usr/bin/env python3
import astropy.constants as c
import astropy.units as u

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from justdoit import compute, compute_yasf
from justplotit import find_nearest_1d

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
