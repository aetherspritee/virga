import pandas as pd
import numpy as np
import os
from scipy import optimize
from atmosphere import Atmosphere
from root_functions import qvs_below_model
import gas_properties
import pvaps

from direct_mmr_solver import direct_solver


def compute_yasf(
    atmo: Atmosphere,
    directory=None,
    as_dict=True,
    direct_tol=1e-15,
    refine_TP=True,
    og_vfall=True,
    analytical_rg=True,
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

    results["z_cld"] = None  # temporary fix
    results["qc"], results["qt"], results["rg"], results["reff"], results["ndz"], results["qc_path"], results["pres_out"], results["temp_out"], results["z_out"], results["mixl"] = direct_solver(
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

    results["condensibles"] = condensibles
    for i, igas in zip(range(ngas), condensibles):
        run_gas = getattr(gas_properties, igas)
        gas_mw[i], gas_mmr[i], rho_p[i] = run_gas(mmw, mh=mh, gas_mmr=atmo.gas_mmr)

        # TODO: currently only works with precalculated values
        # qext_test, qscat_test, g_qscat_test, radius_test, wave_in_test = calc_mie_db(
        #     [igas], directory, directory, rmin=1e-5, nradii=10
        # )

        qext_gas, qscat_gas, cos_qscat_gas, nwave, radius, wave_in = get_mie_yasf(igas, directory)

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

    print("Starting optical calculations")
    results["opd"], results["w0"], results["g0"], results["opd_gas"] = calc_optics(
        nwave,
        results["qc"],
        results["qt"],
        results["rg"],
        results["reff"],
        results["ndz"],
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
    export_results(atmo, fsed_in, results)

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
        print("im here lmao")
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

        if i == 0:
            nradii = len(radius)
            rmin = np.min(radius)
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
    return {
        "pressure": pressure / 1e6,
        "pressure_unit": "bar",
        "temperature": temperature,
        "temperature_unit": "kelvin",
        "wave": wave[:, 0],
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
    verbose,
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
                    arg2 = -np.log(rr / rg[iz, igas]) ** 2 / (2 * np.log(rsig) ** 2)
                    norm = norm + arg1 * np.exp(arg2)
                    # print (rr, rg[iz,igas],rsig,arg1,arg2)

                # normalization
                norm = ndz[iz, igas] / norm

                # TODO: @dusc: check ssa formulae to try and understand what this monstrosity is
                for irad in range(nrad):
                    rr = radius[irad]
                    # @dusc: this refers to
                    arg1 = dr[irad] / (np.sqrt(2.0 * PI) * np.log(rsig))
                    arg2 = -np.log(rr / rg[iz, igas]) ** 2 / (2 * np.log(rsig) ** 2)
                    pir2ndz = norm * PI * rr * arg1 * np.exp(arg2)
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
            if igas == "Mg2SiO4":
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
