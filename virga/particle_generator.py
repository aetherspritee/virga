#!/usr/bin/env python3

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname("/home/dsc/master/dla-particles/"))

from particle import Particle
import numpy as np
from pathlib import Path
import csv, time, subprocess, re, math
# yuh

AGG_GEN_BIN_PATH = "/home/dsc/aggregate_generator/aggregate_gen_main"

class ParticleGenerator():
    def __init__(self) -> None:
        pass

    def mie_sphere(self,radius: float, refrind_type_idx: int, directory: Path) -> Path:
        file_name = f"mie_sphere_{radius}_nm.csv"
        with open(Path(directory) / Path(file_name), "a"):
            pass
        with open(Path(directory) / Path(file_name), "w") as f:
            writer = csv.writer(f)
            writer.writerow([0.0,0.0,0.0,radius,refrind_type_idx])
        return Path(directory) / Path(file_name)

    def aggregate_generator(self, radius: float, refrind: complex, df: float, N: int, directory: Path, n1: int=4) -> Path:
        # n1 = math.ceil(N/(2**layer))
        layer = np.log2(N/n1)
        print(f"{layer = }")
        pr = subprocess.Popen([AGG_GEN_BIN_PATH, f"{n1}", f"{layer}", "1.0", f"{df}", "1"])
        pr.communicate()
        # time.sleep(5)

        # parse results
        p = Particle(with_seed=False)
        p.import_particle_out(f"agg0_N{N}_kf1.0_Df{df}.out")
        # p.visualize()

        # scale particle and prep
        p.scale(radius*1e-9) # scale to nm
        csv_name = f"agg_gen_{N}_{radius}_{df}.csv"
        print(f"{str(directory / Path(csv_name)) = }")
        p.export_particle(path = str(directory/Path(csv_name)))

        # clean up results
        files = os.listdir()
        to_delete = [i for i in files if re.match("agg[0-9]", i)]
        for file in to_delete:
            print(file)
            os.remove(file)

        return directory/Path(csv_name)
