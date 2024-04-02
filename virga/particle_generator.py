#!/usr/bin/env python3
from pathlib import Path
import csv
# yuh

class ParticleGenerator():
    def __init__(self) -> None:
        pass

    def mie_sphere(self,radius: float, refrind: complex, directory: Path) -> Path:
        file_name = f"mie_sphere_{radius}_nm.csv"
        with open(Path(directory) / Path(file_name), "a"):
            pass
        with open(Path(directory) / Path(file_name), "w") as f:
            writer = csv.writer(f)
            writer.writerow([0.0,0.0,0.0,radius,refrind])
        return Path(directory) / Path(file_name)
