#!/bin/bash

set -e

LAMMPS_VERSION=3Mar2020
Make_CPUs=4

wget https://github.com/lammps/lammps/archive/stable_${LAMMPS_VERSION}.zip
unzip stable_${LAMMPS_VERSION}.zip 
rm -f stable_${LAMMPS_VERSION}.zip 
cd lammps-stable_${LAMMPS_VERSION}

mkdir build
cd build
cmake -DPKG_MANYBODY=yes \
      -DPKG_KSPACE=yes \
      -DPKG_MOLECULE=yes \
      -DPKG_RIGID=yes \
    	../cmake
make VERBOSE=1 -j ${Make_CPUs}
