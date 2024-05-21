#!/usr/bin/env python3

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname("/home/dsc/master/dla-particles/"))

from particle import Particle as PParticle
import numpy as np
from pathlib import Path
import csv, time, subprocess, re, math
# yuh

AGG_GEN_BIN_PATH = "/home/dsc/aggregate_generator/aggregate_gen_main"
FRACVAL_BIN_PATH = "/home/dsc/FracVAL/FRACVAL"

class Particle():
    def __init__(self, radii: list=[10, 20, 30, 40], monomer_size: float=0.1,Df: float=1.8, kf: float=1.0,rho: float=3.2) -> None:
        self.radii = radii
        self.monomer_size = monomer_size
        self.Df = Df
        self.kf = kf
        N = list(kf * (np.array(radii)/monomer_size)**Df)
        self.N = [int(i) for i in N]
        self.rho = rho

class ParticleGenerator():
    def __init__(self, fracval_bin_path = FRACVAL_BIN_PATH, agg_gen_bin_path = AGG_GEN_BIN_PATH) -> None:
        self.fracval_bin_path = fracval_bin_path
        self.agg_gen_bin_path = agg_gen_bin_path

    def mie_sphere(self,radius: float, refrind_type_idx: int, directory: Path) -> Path:
        file_name = f"mie_sphere_{radius}_nm.csv"
        with open(Path(directory) / Path(file_name), "a"):
            pass
        with open(Path(directory) / Path(file_name), "w") as f:
            writer = csv.writer(f)
            writer.writerow([0.0,0.0,0.0,radius,refrind_type_idx])
        return Path(directory) / Path(file_name)

    def aggregate_generator(self, radius: float, df: float, N: int, directory: Path,kf: float = 1.0, n1: int=4) -> Path:
        # n1 = math.ceil(N/(2**layer))
        layer = np.log2(N/n1)
        print(f"{layer = }")
        pr = subprocess.Popen([self.agg_gen_bin_path, f"{n1}", f"{layer}", f"{kf}", f"{df}", "1"])
        pr.communicate()
        # time.sleep(5)

        # parse results
        p = PParticle(with_seed=False)
        p.import_particle_out(f"agg0_N{N}_kf{kf}_Df{df}.out")
        # p.visualize()

        # scale particle and prep
        p.scale(radius*1e-4) # scale to µm, radius is given in cm
        csv_name = f"agg_gen_{N}_{radius}_{kf}_{df}.csv"
        # check if already exists, else add version
        if os.path.isfile(directory / Path(csv_name)):
            print("File exists already, adding version tag")
            num_of_files = 0
            for f in os.listdir(directory):
                num_of_files += len(re.findall(f"agg_gen_{N}_{radius}_{kf}_{df}",f))
            csv_name = f"agg_gen_{N}_{radius}_{kf}_{df}_v{num_of_files}.csv"
        print(f"{str(directory / Path(csv_name)) = }")
        p.export_particle(path = str(directory/Path(csv_name)))

        # clean up results
        files = os.listdir()
        to_delete = [i for i in files if re.match("agg[0-9]", i)]
        for file in to_delete:
            print(file)
            os.remove(file)

        return directory/Path(csv_name)

    def fracval(self,r_mon: float, df: float, kf: float, N: int, r_agg: float, directory: Path, nlim = 256) -> Path:
        if N > nlim:
            # i dont want to die while this shit runs, so limiting the number of maximum monomers in a given particle
            N = nlim
            r_mon = (kf/N * r_agg**df)**(1/df)
        pr = subprocess.Popen([self.fracval_bin_path, f"{N}", f"{1}", f"{df}", f"{kf}"])
        pr.communicate()

        # parse results
        p = PParticle(with_seed=False)
        Nf = f"{N:08d}"
        Na = f"{1:08d}"
        p.import_particle_out2(f'FRACVAL_N_{Nf}_Agg_{Na}.dat')
        # p.visualize()

        # scale particle and prep
        p.scale(r_mon*1e-4) # scale to µm
        csv_name = f"fracval_{N}_{r_mon}_{kf}_{df}.csv"
        # check if already exists, else add version
        if os.path.isfile(directory / Path(csv_name)):
            print("File exists already, adding version tag")
            num_of_files = 0
            for f in os.listdir(directory):
                num_of_files += len(re.findall(f"fracval_{N}_{r_mon}_{kf}_{df}",f))
            csv_name = f"fracval_{N}_{r_mon}_{kf}_{df}_v{num_of_files}.csv"
        print(f"{str(directory / Path(csv_name)) = }")
        p.export_particle(path = str(directory/Path(csv_name)))

        # clean up results
        files = os.listdir()
        to_delete = [i for i in files if re.match("agg[0-9]", i)]
        for file in to_delete:
            print(file)
            os.remove(file)

        return directory/Path(csv_name)
