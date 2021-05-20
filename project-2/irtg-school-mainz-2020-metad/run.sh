#!/bin/bash

lmpexec=${PLUMED_INSTALL_DIR}/bin/lmp
ncor=2

mpirun -np ${ncor} ${lmpexec} < start.lmp > out.lmp
