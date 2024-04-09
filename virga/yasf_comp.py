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
    df = jdi_utils.hot_jupiter()
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
radius =[1.00000000e-05, 2.59970240e-05, 6.75845259e-05, 1.75699655e-04, 4.56766814e-04, 1.18745779e-03, 3.08703686e-03, 8.02537715e-03, 2.08635923e-02, 5.42391310e-02]

all_out_og = jdi.compute(a, as_dict=True,og_solver=False, directory=mieff_directory)
all_out_yasf = jdi.compute_yasf(a, as_dict=True, directory=mieff_directory, radii=radius)

print(f"{all_out_og['wave_in'] = }")
print(f"{all_out_yasf['wave_in'] = }")

print(f"{all_out_og['mean_particle_r'] = }")
print(f"{all_out_yasf['mean_particle_r'] = }")

print(f"{all_out_og['droplet_eff_r'] = }")
print(f"{all_out_yasf['droplet_eff_r'] = }")


print(f"{all_out_og['asymmetry'] = }")
print(f"{all_out_yasf['asymmetry'] = }")
