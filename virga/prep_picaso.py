# import os
# os.environ['picaso_refdata'] = "/Users/dusc/Code/picaso/reference"
# os.environ['PYSYN_CDBS'] = "/Users/dsc/grp/redcat/trds/"

import warnings
warnings.filterwarnings('ignore')
#main programs
from picaso import justdoit as pj
from virga import justdoit as vj
#plot tools
from picaso import justplotit as picplt
from virga import justplotit as cldplt
from bokeh.plotting import show

import astropy.units as u
import pandas as pd


opacity = pj.opannection(wave_range=[0.3,1])

sum_planet = pj.inputs()
sum_planet.phase_angle(0) #radians
sum_planet.gravity(gravity=25, gravity_unit=u.Unit('m/(s**2)')) #any astropy units available
sum_planet.star(opacity, 5000,0,4.0) #opacity db, pysynphot database, temp, metallicity, logg

df_atmo = pd.read_csv(pj.jupiter_pt(), delim_whitespace=True)
#you will have to add kz to the picaso profile
df_atmo['kz'] = [1e9]*df_atmo.shape[0]

metallicity = 1 #atmospheric metallicity relative to Solar
mean_molecular_weight = 2.2 # atmospheric mean molecular weight
directory ='/Users/dusc/virga-data/'

#business as usual
sum_planet.atmosphere(df=df_atmo)

#let's get the cloud free spectrum for reference
cloud_free = sum_planet.spectrum(opacity)

x_cld_free, y_cld_free = pj.mean_regrid(cloud_free['wavenumber'], cloud_free['albedo'], R=150)

#we can get the same full output from the virga run
cld_out = sum_planet.virga(['H2O'],directory, fsed=1,mh=metallicity,
                           mmw = mean_molecular_weight, full_output=True)
out = sum_planet.spectrum(opacity, full_output=True)

x_cldy, y_cldy = pj.mean_regrid(out['wavenumber'], out['albedo'], R=150)

show(picplt.spectrum([x_cld_free, x_cldy],
                     [y_cld_free, y_cldy],plot_width=500, plot_height=300,
                  legend=['Cloud Free','Cloudy']))

show(picplt.photon_attenuation(out['full_output'], plot_width=500, plot_height=300))

print(f"{cld_out = }")
fig, dndr = cldplt.radii(cld_out,at_pressure=0.1)
show(fig)

hot_atmo = df_atmo
hot_atmo['temperature'] = hot_atmo['temperature'] + 600

#remember we can use recommend_gas function to look at what the condensation curves look like
recommended = vj.recommend_gas(hot_atmo['pressure'], hot_atmo['temperature'], metallicity,mean_molecular_weight,
                              plot=True)
