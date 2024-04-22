#!/usr/bin/env python3
from particle_generator import ParticleGenerator
from pathlib import Path

p = ParticleGenerator()
res = p.aggregate_generator(radius=1.4,refrind=1.0+0.0j, df=1.7, N=256, directory=Path("/home/dsc/master/virga"))
print(res)
