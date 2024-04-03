import numpy as np
import pandas as pd
import astropy.units as u
import justdoit as jdi
import justplotit as jpi
import matplotlib.pyplot as plt
import time
from bokeh.plotting import show, figure
from direct_mmr_solver import generate_altitude
import jdi_utils

#   locate data
mieff_directory = "/home/dsc/virga-data"
fsed = 1
b = 2
eps = 10
Mark_data = False
refine_TP = False
quick_stop = False
generate = False

if Mark_data:
    TP_directory = "~/Documents/codes/all-data/Mark_data/"
    filenames = [
        "t1000g1000nc_m0.0.dat",
        "t1500g1000nc_m0.0.dat",
        "t1700g1000f3_m0.0k.dat",
        "t200g3160nc_m0.0.dat",
        "t2400g3160nc_m0.0.dat",
    ]
    filename = TP_directory + filenames[1]

    #   define atmosphere properties
    df = pd.read_csv(filename, delim_whitespace=True, usecols=[1, 2], header=None)
    df.columns = ["pressure", "temperature"]
    pressure = np.array(df["pressure"])[1:]
    temperature = np.array(df["temperature"])[1:]
    grav = df["pressure"][0] * 100
    kz = 1e9

    metallicity = 1  # atmospheric metallicity relative to Solar
    mean_molecular_weight = 2.2  # atmospheric mean molecular weight
    # get pyeddy recommendation for which gases to run
    recommended_gases = jdi.recommend_gas(
        pressure, temperature, metallicity, mean_molecular_weight
    )

    a = jdi.Atmosphere(
        [recommended_gases[0]],
        fsed=fsed,
        mh=metallicity,
        mmw=mean_molecular_weight,
        b=b,
    )
    a.gravity(gravity=grav, gravity_unit=u.Unit("cm/(s**2)"))
    a.ptk(df=pd.DataFrame({"pressure": pressure, "temperature": temperature, "kz": kz}))

else:
    metallicity = 1  # atmospheric metallicity relative to Solar
    mean_molecular_weight = 2.2  # atmospheric mean molecular weight

    # set the run
    # a = jdi.Atmosphere(['MnS','Cr','MgSiO3','Fe'],
    a = jdi.Atmosphere(
        ["MnS"], fsed=fsed, mh=metallicity, mmw=mean_molecular_weight, b=b  # , 'Cr'],
    )

    # set the planet gravity
    grav = 7.460
    a.gravity(gravity=grav, gravity_unit=u.Unit("m/(s**2)"))

    if generate:
        df = jid_utils.hot_jupiter()
        pres = np.array(df["pressure"])
        temp = np.array(df["temperature"])
        kz = np.array(df["kz"])
        gravity = grav * 100
        print("initial number of pressure values = ", len(pres))

        plt.ylim(pres[len(pres) - 1], pres[0])
        plt.loglog(temp, pres, label="initial")

        z, pres, P_z, temp, T_z, T_P, kz = generate_altitude(
            pres, temp, kz, gravity, mean_molecular_weight, refine_TP
        )
        print("refined number of pressure values = ", len(pres))

        a.ptk(df=pd.DataFrame({"pressure": pres, "temperature": temp, "kz": kz}))

        plt.loglog(temp, pres, "--", label="refined")
        plt.ylabel("pressure")
        plt.xlabel("temperature")
        plt.legend(loc="best")
        plt.savefig("temperature_profile.png")
        plt.show()

    else:
        # Get preset pt profile for testing
        a.ptk(df=jdi_utils.hot_jupiter())

#   verify original and new solvers give same mixing ratios
fig1, ax1 = plt.subplots()

radii = [0.1]

all_out_og = jdi.compute(a, as_dict=True, directory=mieff_directory)
all_out_yasf = jdi.compute_yasf(a, as_dict=True, directory=mieff_directory, radii=radii)

print(f"{all_out_og = }")
print(f"{all_out_yasf = }")
