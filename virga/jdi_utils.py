#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from scipy import optimize
import pvaps
import gas_properties
from justplotit import plot_format

from root_functions import find_cond_t

def get_r_grid_w_max(r_min=1e-8, r_max=5.4239131e-2, n_radii=60):
    """
    Get spacing of radii to run Mie code

    r_min : float
            Minimum radius to compute (cm)
    r_max : float
            Maximum radius to compute (cm)
    n_radii : int
            Number of radii to compute
    """

    radius = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
    rat = radius[1] / radius[0]
    rup = 2 * rat / (rat + 1) * radius
    dr = np.zeros(rup.shape)
    dr[1:] = rup[1:] - rup[:-1]
    dr[0] = dr[1] ** 2 / dr[2]

    return radius, rup, dr


def recommend_gas(
    pressure,
    temperature,
    mh,
    mmw,
    plot=False,
    return_plot=False,
    legend="inside",
    **plot_kwargs,
):
    """
    Recommends condensate species for a users calculation.

    Parameters
    ----------
    pressure : ndarray, list
        Pressure grid for user's pressure-temperature profile. Unit=bars
    temperature : ndarray, list
        Temperature grid (should be on same grid as pressure input). Unit=Kelvin
    mh : float
        Metallicity in NOT log units. Solar =1
    mmw : float
        Mean molecular weight of the atmosphere. Solar = 2.2
    plot : bool, optional
        Default is False. Plots condensation curves against PT profile to
        demonstrate why it has chose the gases it did.
    plot_kwargs : kwargs
        Plotting kwargs for bokeh figure

    Returns
    -------
    ndarray, ndarray
        pressure (bars), condensation temperature (Kelvin)
    """
    if plot:
        from bokeh.plotting import figure, show
        from bokeh.models import Legend
        from bokeh.palettes import magma

        plot_kwargs["y_range"] = plot_kwargs.get("y_range", [1e2, 1e-3])
        plot_kwargs["height"] = plot_kwargs.get("height", 400)
        plot_kwargs["width"] = plot_kwargs.get("width", 600)
        plot_kwargs["x_axis_label"] = plot_kwargs.get("x_axis_label", "Temperature (K)")
        plot_kwargs["y_axis_label"] = plot_kwargs.get("y_axis_label", "Pressure (bars)")
        plot_kwargs["y_axis_type"] = plot_kwargs.get("y_axis_type", "log")
        fig = figure(**plot_kwargs)

    all_gases = available()
    cond_ts = []
    recommend = []
    line_widths = []
    for gas_name in all_gases:  # case sensitive names
        # grab p,t from eddysed
        cond_p, t = condensation_t(gas_name, mh, mmw)
        cond_ts += [t]

        interp_cond_t = np.interp(pressure, cond_p, t)

        diff_curve = interp_cond_t - temperature

        if (len(diff_curve[diff_curve > 0]) > 0) & (
            len(diff_curve[diff_curve < 0]) > 0
        ):
            recommend += [gas_name]
            line_widths += [5]
        else:
            line_widths += [1]

    if plot:
        legend_it = []
        ngas = len(all_gases)
        cols = magma(ngas)
        if legend == "inside":
            fig.line(
                temperature,
                pressure,
                legend_label="User",
                color="black",
                line_width=5,
                line_dash="dashed",
            )
            for i in range(ngas):
                fig.line(
                    cond_ts[i],
                    cond_p,
                    legend_label=all_gases[i],
                    color=cols[i],
                    line_width=line_widths[i],
                )
        else:
            f = fig.line(
                temperature, pressure, color="black", line_width=5, line_dash="dashed"
            )
            legend_it.append(("input profile", [f]))
            for i in range(ngas):
                f = fig.line(
                    cond_ts[i], cond_p, color=cols[i], line_width=line_widths[i]
                )
                legend_it.append((all_gases[i], [f]))

        if legend == "outside":
            legend = Legend(items=legend_it, location=(0, 0))
            legend.click_policy = "mute"
            fig.add_layout(legend, "right")

        plot_format(fig)
        if return_plot:
            return recommend, fig
        else:
            show(fig)
            return recommend


def condensation_t(gas_name, mh, mmw, pressure=np.logspace(-6, 2, 20)):
    """
    Find condensation curve for any planet given a pressure. These are computed
    based on pressure vapor curves defined in pvaps.py.

    Default is to compute condensation temperature on a pressure grid

    Parameters
    ----------
    gas_name : str
        Name of gas, which is case sensitive. See print_available to see which
        gases are available.
    mh : float
        Metallicity in NOT log units. Solar =1
    mmw : float
        Mean molecular weight of the atmosphere. Solar = 2.2
    pressure : ndarray, list, float, optional
        Grid of pressures (bars) to compute condensation temperatures on.
        Default = np.logspace(-3,2,20)

    Returns
    -------
    ndarray, ndarray
        pressure (bars), condensation temperature (Kelvin)
    """
    if isinstance(pressure, (float, int)):
        pressure = [pressure]
    temps = []
    for p in pressure:
        temp = optimize.root_scalar(
            find_cond_t,
            bracket=[10, 10000],
            method="brentq",
            args=(p, mh, mmw, gas_name),
        )
        temps += [temp.root]
    return np.array(pressure), np.array(temps)


def hot_jupiter():
    directory = os.path.join(os.path.dirname(__file__), "reference", "hj.pt")

    df = pd.read_csv(
        directory,
        delim_whitespace=True,
        usecols=[1, 2, 3],
        names=["pressure", "temperature", "kz"],
        skiprows=1,
    )
    df.loc[df["pressure"] > 12.8, "temperature"] = np.linspace(
        1822, 2100, df.loc[df["pressure"] > 12.8].shape[0]
    )
    return df


def brown_dwarf():
    directory = os.path.join(
        os.path.dirname(__file__), "reference", "t1000g100nc_m0.0.dat"
    )

    df = pd.read_csv(
        directory,
        skiprows=1,
        delim_whitespace=True,
        header=None,
        usecols=[1, 2, 3],
        names=["pressure", "temperature", "chf"],
    )
    return df


def picaso_format(opd, w0, g0):
    df = pd.DataFrame(
        index=[i for i in range(opd.shape[0] * opd.shape[1])],
        columns=["lvl", "w", "opd", "w0", "g0"],
    )
    i = 0
    LVL = []
    WV, OPD, WW0, GG0 = [], [], [], []
    for j in range(opd.shape[0]):
        for w in range(opd.shape[1]):
            LVL += [j + 1]
            WV += [w + 1]
            OPD += [opd[j, w]]
            WW0 += [w0[j, w]]
            GG0 += [g0[j, w]]
    df.iloc[:, 0] = LVL
    df.iloc[:, 1] = WV
    df.iloc[:, 2] = OPD
    df.iloc[:, 3] = WW0
    df.iloc[:, 4] = GG0
    return df


def available():
    """
    Print all available gas condensates
    """
    pvs = [i for i in dir(pvaps) if i != "np" and "_" not in i]
    gas_p = [i for i in dir(gas_properties) if i != "np" and "_" not in i]
    return list(np.intersect1d(gas_p, pvs))