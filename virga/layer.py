#!/usr/bin/env python3
import numpy as np
import pvaps
from scipy import optimize
from calc_mie import get_r_grid
from root_functions import (
    vfall,
    var_vfall,
    vfall_find_root,
    vfall_find_root_fractal,
    qvs_below_model,
    solve_force_balance,
)

def layer(
    gas_name,
    rho_p,
    t_layer,
    p_layer,
    t_top,
    t_bot,
    p_top,
    p_bot,
    kz,
    mixl,
    gravity,
    mw_atmos,
    gas_mw,
    q_below,
    supsat,
    fsed,
    b,
    eps,
    z_top,
    z_bot,
    z_alpha,
    z_min,
    param,
    sig,
    mh,
    rmin,
    nrad,
    d_molecule,
    eps_k,
    c_p_factor,
    og_vfall,
    z_cld,
):
    """
    Calculate layer condensate properties by iterating on optical depth
    in one model layer (convering on optical depth over sublayers)

    gas_name : str
        Name of condenstante
    rho_p : float
        density of condensed vapor (g/cm^3)
    t_layer : float
        Temperature of layer mid-pt (K)
    p_layer : float
        Pressure of layer mid-pt (dyne/cm^2)
    t_top : float
        Temperature at top of layer (K)
    t_bot : float
        Temperature at botton of layer (K)
    p_top : float
        Pressure at top of layer (dyne/cm2)
    p_bot : float
        Pressure at botton of layer
    kz : float
        eddy diffusion coefficient (cm^2/s)
    mixl : float
        Mixing length (cm)
    gravity : float
        Gravity of planet cgs
    mw_atmos : float
        Molecular weight of the atmosphere
    gas_mw : float
        Gas molecular weight
    q_below : float
        total mixing ratio (vapor+condensate) below layer (g/g)
    supsat : float
        Super saturation factor
    fsed : float
        Sedimentation efficiency coefficient (unitless)
    b : float
        Denominator of exponential in sedimentation efficiency  (if param is 'exp')
    eps: float
        Minimum value of fsed function (if param=exp)
    z_top : float
        Altitude at top of layer
    z_bot : float
        Altitude at bottom of layer
    z_alpha : float
        Altitude at which fsed=alpha for variable fsed calculation
    param : str
        fsed parameterisation
        'const' (constant), 'exp' (exponential density derivation)
    sig : float
        Width of the log normal particle distribution
    mh : float
        Metallicity NON log soar (1=1xSolar)
    rmin : float
        Minium radius on grid (cm)
    nrad : int
        Number of radii on Mie grid
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
    og_vfall : bool
        Use original or new vfall calculation

    Returns
    -------
    qc_layer : ndarray
        condenstate mixing ratio (g/g)
    qt_layer : ndarray
        gas + condensate mixing ratio (g/g)
    rg_layer : ndarray
        geometric mean radius of condensate  cm
    reff_layer : ndarray
        droplet effective radius (second moment of size distrib, cm)
    ndz_layer : ndarray
        number column density of condensate (cm^-3)
    q_below : ndarray
        total mixing ratio (vapor+condensate) below layer (g/g)
    """
    #   universal gas constant (erg/mol/K)
    nsub_max = 128
    R_GAS = 8.3143e7
    AVOGADRO = 6.02e23
    K_BOLTZ = R_GAS / AVOGADRO
    PI = np.pi
    #   Number of levels of grid refinement used
    nsub = 1

    #   specific gas constant for atmosphere (erg/K/g)
    r_atmos = R_GAS / mw_atmos

    # specific gas constant for cloud (erg/K/g)
    r_cloud = R_GAS / gas_mw

    #   specific heat of atmosphere (erg/K/g)
    c_p = c_p_factor * r_atmos

    #   pressure thickness of layer
    dp_layer = p_bot - p_top
    dlnp = np.log(p_bot / p_top)

    #   temperature gradient
    dtdlnp = (t_top - t_bot) / dlnp
    lapse_ratio = (t_bot - t_top) / dlnp / (t_layer / c_p_factor)

    #   atmospheric density (g/cm^3)
    rho_atmos = p_layer / (r_atmos * t_layer)

    #   atmospheric scale height (cm)
    scale_h = r_atmos * t_layer / gravity

    #   convective velocity scale (cm/s) from mixing length theory
    w_convect = kz / mixl

    #   atmospheric number density (molecules/cm^3)
    n_atmos = p_layer / (K_BOLTZ * t_layer)

    #   atmospheric mean free path (cm)
    mfp = 1.0 / (np.sqrt(2.0) * n_atmos * PI * d_molecule**2)

    # atmospheric viscosity (dyne s/cm^2)
    # EQN B2 in A & M 2001, originally from Rosner+2000
    # Rosner, D. E. 2000, Transport Processes in Chemically Reacting Flow Systems (Dover: Mineola)
    visc = (
        5.0
        / 16.0
        * np.sqrt(PI * K_BOLTZ * t_layer * (mw_atmos / AVOGADRO))
        / (PI * d_molecule**2)
        / (1.22 * (t_layer / eps_k) ** (-0.16))
    )

    #   --------------------------------------------------------------------
    #   Top of convergence loop
    converge = False
    while not converge:
        #   Zero cumulative values
        qc_layer = 0.0
        qt_layer = 0.0
        ndz_layer = 0.0
        opd_layer = 0.0

        #   total mixing ratio and pressure at bottom of sub-layer

        qt_bot_sub = q_below
        p_bot_sub = p_bot
        z_bot_sub = z_bot

        # SUBALYER
        dp_sub = dp_layer / nsub

        for isub in range(nsub):
            qt_below = qt_bot_sub
            p_top_sub = p_bot_sub - dp_sub
            dz_sub = scale_h * np.log(p_bot_sub / p_top_sub)  # width of layer
            p_sub = 0.5 * (p_bot_sub + p_top_sub)
            #################### CHECK #####################
            z_top_sub = z_bot_sub + dz_sub
            z_sub = z_bot_sub + scale_h * np.log(p_bot_sub / p_sub)  # midpoint of layer
            ################################################
            t_sub = t_bot + np.log(p_bot / p_sub) * dtdlnp
            (
                qt_top,
                qc_sub,
                qt_sub,
                rg_sub,
                reff_sub,
                ndz_sub,
                z_cld,
                fsed_layer,
            ) = calc_qc(
                gas_name,
                supsat,
                t_sub,
                p_sub,
                r_atmos,
                r_cloud,
                qt_below,
                mixl,
                dz_sub,
                gravity,
                mw_atmos,
                mfp,
                visc,
                rho_p,
                w_convect,
                fsed,
                b,
                eps,
                param,
                z_bot_sub,
                z_sub,
                z_alpha,
                z_min,
                sig,
                mh,
                rmin,
                nrad,
                og_vfall,
                z_cld,
            )

            #   vertical sums
            qc_layer = qc_layer + qc_sub * dp_sub / gravity
            qt_layer = qt_layer + qt_sub * dp_sub / gravity
            ndz_layer = ndz_layer + ndz_sub

            if reff_sub > 0.0:
                opd_layer = opd_layer + 1.5 * qc_sub * dp_sub / gravity / (
                    rho_p * reff_sub
                )

            #   Increment values at bottom of sub-layer

            qt_bot_sub = qt_top
            p_bot_sub = p_top_sub
            z_bot_sub = z_top_sub

        #    Check convergence on optical depth
        if nsub_max == 1:
            converge = True
        elif nsub == 1:
            opd_test = opd_layer
        elif (opd_layer == 0.0) or (nsub >= nsub_max):
            converge = True
        elif abs(1.0 - opd_test / opd_layer) <= 1e-2:
            converge = True
        else:
            opd_test = opd_layer

        nsub = nsub * 2
    #   Update properties at bottom of next layer

    q_below = qt_top

    # Get layer averages

    if opd_layer > 0.0:
        reff_layer = 1.5 * qc_layer / (rho_p * opd_layer)
        lnsig2 = 0.5 * np.log(sig) ** 2
        rg_layer = reff_layer * np.exp(-5 * lnsig2)
    else:
        reff_layer = 0.0
        rg_layer = 0.0

    qc_layer = qc_layer * gravity / dp_layer
    qt_layer = qt_layer * gravity / dp_layer

    return (
        qc_layer,
        qt_layer,
        rg_layer,
        reff_layer,
        ndz_layer,
        q_below,
        z_cld,
        fsed_layer,
    )


def calc_qc(
    gas_name,
    supsat,
    t_layer,
    p_layer,
    r_atmos,
    r_cloud,
    q_below,
    mixl,
    dz_layer,
    gravity,
    mw_atmos,
    mfp,
    visc,
    rho_p,
    w_convect,
    fsed,
    b,
    eps,
    param,
    z_bot,
    z_layer,
    z_alpha,
    z_min,
    sig,
    mh,
    rmin,
    nrad,
    og_vfall=True,
    z_cld=None,
):
    """
    Calculate condensate optical depth and effective radius for a layer,
    assuming geometric scatterers.

    gas_name : str
        Name of condensate
    supsat : float
        Super saturation factor
    t_layer : float
        Temperature of layer mid-pt (K)
    p_layer : float
        Pressure of layer mid-pt (dyne/cm^2)
    r_atmos : float
        specific gas constant for atmosphere (erg/K/g)
    r_cloud : float
        specific gas constant for cloud species (erg/K/g)
    q_below : float
        total mixing ratio (vapor+condensate) below layer (g/g)
    mxl : float
        convective mixing length scale (cm): no less than 1/10 scale height
    dz_layer : float
        Thickness of layer cm
    gravity : float
        Gravity of planet cgs
    mw_atmos : float
        Molecular weight of the atmosphere
    mfp : float
        atmospheric mean free path (cm)
    visc : float
        atmospheric viscosity (dyne s/cm^2)
    rho_p : float
        density of condensed vapor (g/cm^3)
    w_convect : float
        convective velocity scale (cm/s)
    fsed : float
        Sedimentation efficiency coefficient (unitless)
    b : float
        Denominator of exponential in sedimentation efficiency  (if param is 'exp')
    eps: float
        Minimum value of fsed function (if param=exp)
    z_bot : float
        Altitude at bottom of layer
    z_layer : float
        Altitude of midpoint of layer (cm)
    z_alpha : float
        Altitude at which fsed=alpha for variable fsed calculation
    param : str
        fsed parameterisation
        'const' (constant), 'exp' (exponential density derivation)
    sig : float
        Width of the log normal particle distrubtion
    mh : float
        Metallicity NON log solar (1 = 1x solar)
    rmin : float
        Minium radius on grid (cm)
    nrad : int
        Number of radii on Mie grid

    Returns
    -------
    qt_top : float
        gas + condensate mixing ratio at top of layer(g/g)
    qc_layer : float
        condenstate mixing ratio (g/g)
    qt_layer : float
        gas + condensate mixing ratio (g/g)
    rg_layer : float
        geometric mean radius of condensate  cm
    reff_layer : float
        droplet effective radius (second moment of size distrib, cm)
    ndz_layer : float
        number column density of condensate (cm^-3)
    """

    get_pvap = getattr(pvaps, gas_name)
    if gas_name in ['Mg2SiO4','CaTiO3','CaAl12O19','FakeHaze','H2SO4','KhareHaze','SteamHaze300K','SteamHaze400K']:
        pvap = get_pvap(t_layer, p_layer, mh=mh)
    else:
        pvap = get_pvap(t_layer, mh=mh)

    fs = supsat + 1

    #   atmospheric density (g/cm^3)
    rho_atmos = p_layer / (r_atmos * t_layer)

    #   mass mixing ratio of saturated vapor (g/g)
    qvs = fs * pvap / ((r_cloud) * t_layer) / rho_atmos

    #   --------------------------------------------------------------------
    #   Layer is cloud free

    if q_below < qvs:
        qt_layer = q_below
        qt_top = q_below
        qc_layer = 0.0
        rg_layer = 0.0
        reff_layer = 0.0
        ndz_layer = 0.0
        z_cld = z_cld
        fsed_mid = 0

    else:
        #   --------------------------------------------------------------------
        #   Cloudy layer: first calculate qt and qc at top of layer,
        #   then calculate layer averages
        if isinstance(z_cld, type(None)):
            z_cld = z_bot
        else:
            z_cld = z_cld

        #   range of mixing ratios to search (g/g)
        qhi = q_below
        qlo = qhi / 1e3

        #   load parameters into advdiff common block

        ad_qbelow = q_below
        ad_qvs = qvs
        ad_mixl = mixl
        ad_dz = dz_layer
        ad_rainf = fsed

        #   Find total vapor mixing ratio at top of layer
        # find_root = True
        # while find_root:
        #    try:
        #        qt_top = optimize.root_scalar(advdiff, bracket=[qlo, qhi], method='brentq',
        #                    args=(ad_qbelow,ad_qvs, ad_mixl,ad_dz ,ad_rainf,
        #                        z_bot, b, eps, param)
        #                        )#, xtol = 1e-20)
        #        find_root = False
        #    except ValueError:
        #        qlo = qlo/10
        #
        # qt_top = qt_top.root
        if param == "const":
            qt_top = qvs + (q_below - qvs) * np.exp(-fsed * dz_layer / mixl)
        elif param == "exp":
            fs = fsed / np.exp(z_alpha / b)
            qt_top = qvs + (q_below - qvs) * np.exp(
                -b * fs / mixl * np.exp(z_bot / b) * (np.exp(dz_layer / b) - 1)
                + eps * dz_layer / mixl
            )

        #   Use trapezoid rule (for now) to calculate layer averages
        #   -- should integrate exponential
        qt_layer = 0.5 * (q_below + qt_top)

        #   Find total condensate mixing ratio
        qc_layer = np.max([0.0, qt_layer - qvs])

        #   --------------------------------------------------------------------
        #   Find <rw> corresponding to <w_convect> using function vfall()

        #   range of particle radii to search (cm)
        rlo = 1.0e-10
        rhi = 10.0

        #   precision of vfall solution (cm/s)
        find_root = True
        while find_root:
            try:
                if og_vfall:
                    rw_temp = optimize.root_scalar(
                        vfall_find_root,
                        bracket=[rlo, rhi],
                        method="brentq",
                        args=(
                            gravity,
                            mw_atmos,
                            mfp,
                            visc,
                            t_layer,
                            p_layer,
                            rho_p,
                            w_convect,
                        ),
                    )
                else:
                    rw_temp = solve_force_balance(
                        "rw",
                        w_convect,
                        gravity,
                        mw_atmos,
                        mfp,
                        visc,
                        t_layer,
                        p_layer,
                        rho_p,
                        rlo,
                        rhi,
                    )
                find_root = False
            except ValueError:
                rlo = rlo / 10
                rhi = rhi * 10

        # fall velocity particle radius
        if og_vfall:
            rw_layer = rw_temp.root
        else:
            rw_layer = rw_temp

        #   geometric std dev of lognormal size distribution
        lnsig2 = 0.5 * np.log(sig) ** 2
        #   sigma floor for the purpose of alpha calculation
        sig_alpha = np.max([1.1, sig])

        #   find alpha for power law fit vf = w(r/rw)^alpha
        def pow_law(r, alpha):
            return np.log(w_convect) + alpha * np.log(r / rw_layer)

        r_, rup, dr = get_r_grid(r_min=rmin, n_radii=nrad)
        vfall_temp = []
        for j in range(len(r_)):
            if og_vfall:
                vfall_temp.append(
                    vfall(r_[j], gravity, mw_atmos, mfp, visc, t_layer, p_layer, rho_p)
                )
            else:
                vlo = 1e0
                vhi = 1e6
                find_root = True
                while find_root:
                    try:
                        vfall_temp.append(
                            solve_force_balance(
                                "vfall",
                                r_[j],
                                gravity,
                                mw_atmos,
                                mfp,
                                visc,
                                t_layer,
                                p_layer,
                                rho_p,
                                vlo,
                                vhi,
                            )
                        )
                        find_root = False
                    except ValueError:
                        vlo = vlo / 10
                        vhi = vhi * 10

        pars, cov = optimize.curve_fit(
            f=pow_law,
            xdata=r_,
            ydata=np.log(vfall_temp),
            p0=[0],
            bounds=(-np.inf, np.inf),
        )
        alpha = pars[0]

        #   fsed at middle of layer
        if param == "exp":
            fsed_mid = fs * np.exp(z_layer / b) + eps
        else:  # 'const'
            fsed_mid = fsed

        #     EQN. 13 A&M
        #   geometric mean radius of lognormal size distribution
        rg_layer = fsed_mid ** (1.0 / alpha) * rw_layer * np.exp(-(alpha + 6) * lnsig2)

        #   droplet effective radius (cm)
        reff_layer = rg_layer * np.exp(5 * lnsig2)

        #      EQN. 14 A&M
        #   column droplet number concentration (cm^-2)
        ndz_layer = (
            3
            * rho_atmos
            * qc_layer
            * dz_layer
            / (4 * np.pi * rho_p * rg_layer**3)
            * np.exp(-9 * lnsig2)
        )

    return qt_top, qc_layer, qt_layer, rg_layer, reff_layer, ndz_layer, z_cld, fsed_mid


def layer_fractal(
    gas_name,
    rho_p,
    t_layer,
    p_layer,
    t_top,
    t_bot,
    p_top,
    p_bot,
    kz,
    mixl,
    gravity,
    mw_atmos,
    gas_mw,
    q_below,
    supsat,
    fsed,
    b,
    eps,
    z_top,
    z_bot,
    z_alpha,
    z_min,
    param,
    sig,
    mh,
    rmin,
    nrad,
    d_molecule,
    eps_k,
    c_p_factor,
    og_vfall,
    z_cld,
    r_mon,
    Df,
    kf,
):
    """
    Calculate layer condensate properties by iterating on optical depth
    in one model layer (convering on optical depth over sublayers)

    gas_name : str
        Name of condenstante
    rho_p : float
        density of condensed vapor (g/cm^3)
    t_layer : float
        Temperature of layer mid-pt (K)
    p_layer : float
        Pressure of layer mid-pt (dyne/cm^2)
    t_top : float
        Temperature at top of layer (K)
    t_bot : float
        Temperature at botton of layer (K)
    p_top : float
        Pressure at top of layer (dyne/cm2)
    p_bot : float
        Pressure at botton of layer
    kz : float
        eddy diffusion coefficient (cm^2/s)
    mixl : float
        Mixing length (cm)
    gravity : float
        Gravity of planet cgs
    mw_atmos : float
        Molecular weight of the atmosphere
    gas_mw : float
        Gas molecular weight
    q_below : float
        total mixing ratio (vapor+condensate) below layer (g/g)
    supsat : float
        Super saturation factor
    fsed : float
        Sedimentation efficiency coefficient (unitless)
    b : float
        Denominator of exponential in sedimentation efficiency  (if param is 'exp')
    eps: float
        Minimum value of fsed function (if param=exp)
    z_top : float
        Altitude at top of layer
    z_bot : float
        Altitude at bottom of layer
    z_alpha : float
        Altitude at which fsed=alpha for variable fsed calculation
    param : str
        fsed parameterisation
        'const' (constant), 'exp' (exponential density derivation)
    sig : float
        Width of the log normal particle distribution
    mh : float
        Metallicity NON log soar (1=1xSolar)
    rmin : float
        Minium radius on grid (cm)
    nrad : int
        Number of radii on Mie grid
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
    og_vfall : bool
        Use original or new vfall calculation

    Returns
    -------
    qc_layer : ndarray
        condenstate mixing ratio (g/g)
    qt_layer : ndarray
        gas + condensate mixing ratio (g/g)
    rg_layer : ndarray
        geometric mean radius of condensate  cm
    reff_layer : ndarray
        droplet effective radius (second moment of size distrib, cm)
    ndz_layer : ndarray
        number column density of condensate (cm^-3)
    q_below : ndarray
        total mixing ratio (vapor+condensate) below layer (g/g)
    """
    #   universal gas constant (erg/mol/K)
    nsub_max = 128
    R_GAS = 8.3143e7
    AVOGADRO = 6.02e23
    K_BOLTZ = R_GAS / AVOGADRO
    PI = np.pi
    #   Number of levels of grid refinement used
    nsub = 1

    #   specific gas constant for atmosphere (erg/K/g)
    r_atmos = R_GAS / mw_atmos

    # specific gas constant for cloud (erg/K/g)
    r_cloud = R_GAS / gas_mw

    #   specific heat of atmosphere (erg/K/g)
    c_p = c_p_factor * r_atmos

    #   pressure thickness of layer
    dp_layer = p_bot - p_top
    dlnp = np.log(p_bot / p_top)

    #   temperature gradient
    dtdlnp = (t_top - t_bot) / dlnp
    lapse_ratio = (t_bot - t_top) / dlnp / (t_layer / c_p_factor)

    #   atmospheric density (g/cm^3)
    rho_atmos = p_layer / (r_atmos * t_layer)

    #   atmospheric scale height (cm)
    scale_h = r_atmos * t_layer / gravity

    #   convective velocity scale (cm/s) from mixing length theory
    w_convect = kz / mixl

    #   atmospheric number density (molecules/cm^3)
    n_atmos = p_layer / (K_BOLTZ * t_layer)

    #   atmospheric mean free path (cm)
    mfp = 1.0 / (np.sqrt(2.0) * n_atmos * PI * d_molecule**2)

    # atmospheric viscosity (dyne s/cm^2)
    # EQN B2 in A & M 2001, originally from Rosner+2000
    # Rosner, D. E. 2000, Transport Processes in Chemically Reacting Flow Systems (Dover: Mineola)
    visc = (
        5.0
        / 16.0
        * np.sqrt(PI * K_BOLTZ * t_layer * (mw_atmos / AVOGADRO))
        / (PI * d_molecule**2)
        / (1.22 * (t_layer / eps_k) ** (-0.16))
    )

    #   --------------------------------------------------------------------
    #   Top of convergence loop
    converge = False
    while not converge:
        #   Zero cumulative values
        qc_layer = 0.0
        qt_layer = 0.0
        ndz_layer = 0.0
        opd_layer = 0.0

        #   total mixing ratio and pressure at bottom of sub-layer

        qt_bot_sub = q_below
        p_bot_sub = p_bot
        z_bot_sub = z_bot

        # SUBALYER
        dp_sub = dp_layer / nsub

        for isub in range(nsub):
            qt_below = qt_bot_sub
            p_top_sub = p_bot_sub - dp_sub
            dz_sub = scale_h * np.log(p_bot_sub / p_top_sub)  # width of layer
            p_sub = 0.5 * (p_bot_sub + p_top_sub)
            #################### CHECK #####################
            z_top_sub = z_bot_sub + dz_sub
            z_sub = z_bot_sub + scale_h * np.log(p_bot_sub / p_sub)  # midpoint of layer
            ################################################
            t_sub = t_bot + np.log(p_bot / p_sub) * dtdlnp
            (
                qt_top,
                qc_sub,
                qt_sub,
                rg_sub,
                reff_sub,
                ndz_sub,
                z_cld,
                fsed_layer,
            ) = calc_qc_fractal(
                gas_name,
                supsat,
                t_sub,
                p_sub,
                r_atmos,
                r_cloud,
                qt_below,
                mixl,
                dz_sub,
                gravity,
                mw_atmos,
                mfp,
                visc,
                rho_p,
                w_convect,
                fsed,
                b,
                eps,
                param,
                z_bot_sub,
                z_sub,
                z_alpha,
                z_min,
                sig,
                mh,
                rmin,
                nrad,
                og_vfall,
                z_cld,
                r_mon=r_mon,
                Df=Df,
                kf=kf,
            )

            #   vertical sums
            qc_layer = qc_layer + qc_sub * dp_sub / gravity
            qt_layer = qt_layer + qt_sub * dp_sub / gravity
            ndz_layer = ndz_layer + ndz_sub

            if reff_sub > 0.0:
                opd_layer = opd_layer + 1.5 * qc_sub * dp_sub / gravity / (
                    rho_p * reff_sub
                )

            #   Increment values at bottom of sub-layer

            qt_bot_sub = qt_top
            p_bot_sub = p_top_sub
            z_bot_sub = z_top_sub

        #    Check convergence on optical depth
        if nsub_max == 1:
            converge = True
        elif nsub == 1:
            opd_test = opd_layer
        elif (opd_layer == 0.0) or (nsub >= nsub_max):
            converge = True
        elif abs(1.0 - opd_test / opd_layer) <= 1e-2:
            converge = True
        else:
            opd_test = opd_layer

        nsub = nsub * 2
    #   Update properties at bottom of next layer

    q_below = qt_top

    # Get layer averages

    if opd_layer > 0.0:
        reff_layer = 1.5 * qc_layer / (rho_p * opd_layer)
        lnsig2 = 0.5 * np.log(sig) ** 2
        rg_layer = reff_layer * np.exp(-5 * lnsig2)
    else:
        reff_layer = 0.0
        rg_layer = 0.0

    qc_layer = qc_layer * gravity / dp_layer
    qt_layer = qt_layer * gravity / dp_layer

    return (
        qc_layer,
        qt_layer,
        rg_layer,
        reff_layer,
        ndz_layer,
        q_below,
        z_cld,
        fsed_layer,
    )

def calc_qc_fractal(
    gas_name,
    supsat,
    t_layer,
    p_layer,
    r_atmos,
    r_cloud,
    q_below,
    mixl,
    dz_layer,
    gravity,
    mw_atmos,
    mfp,
    visc,
    rho_p,
    w_convect,
    fsed,
    b,
    eps,
    param,
    z_bot,
    z_layer,
    z_alpha,
    z_min,
    sig,
    mh,
    rmin,
    nrad,
    og_vfall=True,
    z_cld=None,
    r_mon=0.01,
    Df=1.8,
    kf=1.0,
):
    """
    Calculate condensate optical depth and effective radius for a layer,
    assuming geometric scatterers.

    gas_name : str
        Name of condensate
    supsat : float
        Super saturation factor
    t_layer : float
        Temperature of layer mid-pt (K)
    p_layer : float
        Pressure of layer mid-pt (dyne/cm^2)
    r_atmos : float
        specific gas constant for atmosphere (erg/K/g)
    r_cloud : float
        specific gas constant for cloud species (erg/K/g)
    q_below : float
        total mixing ratio (vapor+condensate) below layer (g/g)
    mxl : float
        convective mixing length scale (cm): no less than 1/10 scale height
    dz_layer : float
        Thickness of layer cm
    gravity : float
        Gravity of planet cgs
    mw_atmos : float
        Molecular weight of the atmosphere
    mfp : float
        atmospheric mean free path (cm)
    visc : float
        atmospheric viscosity (dyne s/cm^2)
    rho_p : float
        density of condensed vapor (g/cm^3)
    w_convect : float
        convective velocity scale (cm/s)
    fsed : float
        Sedimentation efficiency coefficient (unitless)
    b : float
        Denominator of exponential in sedimentation efficiency  (if param is 'exp')
    eps: float
        Minimum value of fsed function (if param=exp)
    z_bot : float
        Altitude at bottom of layer
    z_layer : float
        Altitude of midpoint of layer (cm)
    z_alpha : float
        Altitude at which fsed=alpha for variable fsed calculation
    param : str
        fsed parameterisation
        'const' (constant), 'exp' (exponential density derivation)
    sig : float
        Width of the log normal particle distrubtion
    mh : float
        Metallicity NON log solar (1 = 1x solar)
    rmin : float
        Minium radius on grid (cm)
    nrad : int
        Number of radii on Mie grid

    Returns
    -------
    qt_top : float
        gas + condensate mixing ratio at top of layer(g/g)
    qc_layer : float
        condenstate mixing ratio (g/g)
    qt_layer : float
        gas + condensate mixing ratio (g/g)
    rg_layer : float
        geometric mean radius of condensate  cm
    reff_layer : float
        droplet effective radius (second moment of size distrib, cm)
    ndz_layer : float
        number column density of condensate (cm^-3)
    """

    get_pvap = getattr(pvaps, gas_name)
    if gas_name in ['Mg2SiO4','CaTiO3','CaAl12O19','FakeHaze','H2SO4','KhareHaze','SteamHaze300K','SteamHaze400K']:
        pvap = get_pvap(t_layer, p_layer, mh=mh)
    else:
        pvap = get_pvap(t_layer, mh=mh)

    fs = supsat + 1

    #   atmospheric density (g/cm^3)
    rho_atmos = p_layer / (r_atmos * t_layer)

    #   mass mixing ratio of saturated vapor (g/g)
    qvs = fs * pvap / ((r_cloud) * t_layer) / rho_atmos

    #   --------------------------------------------------------------------
    #   Layer is cloud free

    if q_below < qvs:
        qt_layer = q_below
        qt_top = q_below
        qc_layer = 0.0
        rg_layer = 0.0
        reff_layer = 0.0
        ndz_layer = 0.0
        z_cld = z_cld
        fsed_mid = 0

    else:
        #   --------------------------------------------------------------------
        #   Cloudy layer: first calculate qt and qc at top of layer,
        #   then calculate layer averages
        if isinstance(z_cld, type(None)):
            z_cld = z_bot
        else:
            z_cld = z_cld

        #   range of mixing ratios to search (g/g)
        qhi = q_below
        qlo = qhi / 1e3

        #   load parameters into advdiff common block

        ad_qbelow = q_below
        ad_qvs = qvs
        ad_mixl = mixl
        ad_dz = dz_layer
        ad_rainf = fsed

        #   Find total vapor mixing ratio at top of layer
        # find_root = True
        # while find_root:
        #    try:
        #        qt_top = optimize.root_scalar(advdiff, bracket=[qlo, qhi], method='brentq',
        #                    args=(ad_qbelow,ad_qvs, ad_mixl,ad_dz ,ad_rainf,
        #                        z_bot, b, eps, param)
        #                        )#, xtol = 1e-20)
        #        find_root = False
        #    except ValueError:
        #        qlo = qlo/10
        #
        # qt_top = qt_top.root
        if param == "const":
            qt_top = qvs + (q_below - qvs) * np.exp(-fsed * dz_layer / mixl)
        elif param == "exp":
            fs = fsed / np.exp(z_alpha / b)
            qt_top = qvs + (q_below - qvs) * np.exp(
                -b * fs / mixl * np.exp(z_bot / b) * (np.exp(dz_layer / b) - 1)
                + eps * dz_layer / mixl
            )

        #   Use trapezoid rule (for now) to calculate layer averages
        #   -- should integrate exponential
        qt_layer = 0.5 * (q_below + qt_top)

        #   Find total condensate mixing ratio
        qc_layer = np.max([0.0, qt_layer - qvs])

        #   --------------------------------------------------------------------
        #   Find <rw> corresponding to <w_convect> using function vfall()

        #   range of particle radii to search (cm)
        # FIXME: im not sure this is a smart idea, but i think its fine
        rlo = 10*r_mon
        rhi = 10.0

        #   precision of vfall solution (cm/s)
        find_root = True
        while find_root:
            try:
                if og_vfall:
                    rw_temp = optimize.root_scalar(
                        vfall_find_root_fractal,
                        bracket=[rlo, rhi],
                        method="brentq",
                        args=(
                            gravity,
                            mw_atmos,
                            mfp,
                            visc,
                            t_layer,
                            p_layer,
                            rho_p,
                            w_convect,
                            Df,
                            r_mon,
                            kf,
                        ),
                    )
                else:
                    rw_temp = solve_force_balance(
                        "rw",
                        w_convect,
                        gravity,
                        mw_atmos,
                        mfp,
                        visc,
                        t_layer,
                        p_layer,
                        rho_p,
                        rlo,
                        rhi,
                    )
                find_root = False
            except ValueError:
                rlo = rlo / 10
                rhi = rhi * 10

        # fall velocity particle radius
        if og_vfall:
            rw_layer = rw_temp.root
            print(f"{rw_layer = }")
        else:
            rw_layer = rw_temp

        #   geometric std dev of lognormal size distribution
        lnsig2 = 0.5 * np.log(sig) ** 2
        #   sigma floor for the purpose of alpha calculation
        sig_alpha = np.max([1.1, sig])

        #   find alpha for power law fit vf = w(r/rw)^alpha
        def pow_law(r, alpha):
            return np.log(w_convect) + alpha * np.log(r / rw_layer)

        r_, _, _ = get_r_grid(r_min=rmin, n_radii=nrad)
        vfall_temp = []
        for j in range(len(r_)):
            if og_vfall:
                vfall_temp.append(
                    var_vfall(r_[j], gravity, mw_atmos, mfp, visc, t_layer, p_layer, rho_p, mode="fractal", r_mon=r_mon,kf=kf, Df=Df)
                )
            else:
                vlo = 1e0
                vhi = 1e6
                find_root = True
                while find_root:
                    try:
                        vfall_temp.append(
                            solve_force_balance(
                                "vfall",
                                r_[j],
                                gravity,
                                mw_atmos,
                                mfp,
                                visc,
                                t_layer,
                                p_layer,
                                rho_p,
                                vlo,
                                vhi,
                            )
                        )
                        find_root = False
                    except ValueError:
                        vlo = vlo / 10
                        vhi = vhi * 10

        # TODO: CHECK ALL OF THIS!
        pars, cov = optimize.curve_fit(
            f=pow_law,
            xdata=r_,
            ydata=np.log(vfall_temp),
            p0=[0],
            bounds=(-np.inf, np.inf),
        )
        alpha = pars[0]

        #   fsed at middle of layer
        if param == "exp":
            fsed_mid = fs * np.exp(z_layer / b) + eps
        else:  # 'const'
            fsed_mid = fsed

        #     EQN. 13 A&M
        #   geometric mean radius of lognormal size distribution
        rg_layer = fsed_mid ** (1.0 / alpha) * rw_layer * np.exp(-(alpha + 6) * lnsig2)

        #   droplet effective radius (cm)
        reff_layer = rg_layer * np.exp(5 * lnsig2)

        #      EQN. 14 A&M
        #   column droplet number concentration (cm^-2)
        ndz_layer = (
            3
            * rho_atmos
            * qc_layer
            * dz_layer
            / (4 * np.pi * rho_p * rg_layer**3)
            * np.exp(-9 * lnsig2)
        )

    return qt_top, qc_layer, qt_layer, rg_layer, reff_layer, ndz_layer, z_cld, fsed_mid
