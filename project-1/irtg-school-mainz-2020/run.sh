#!/bin/bash

lmpexec=$HOME/lammps-stable_3Mar2020/build/lmp
ncor=2

mpirun -np ${ncor} ${lmpexec} < start.lmp > out.lmp
