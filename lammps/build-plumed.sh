#!/bin/bash
set -e

INSTALL_DIR=
BuildMPI=yes
PLUMED_VERSION=2.6.2
Make_CPUs=2

wget https://github.com/plumed/plumed2/archive/v${PLUMED_VERSION}.zip
unzip v${PLUMED_VERSION}.zip
rm -f  v${PLUMED_VERSION}.zip

cd plumed2-${PLUMED_VERSION}

if [[  -z ${INSTALL_DIR} ]];
then 
  INSTALL_DIR=${PWD}/install 
fi

./configure --prefix=${INSTALL_DIR}
make -j ${Make_CPUs}
make install 
cd ..

cat << EOF > source-plumed.sh
export PLUMED_INSTALL_DIR=${INSTALL_DIR}
export PATH=\${PLUMED_INSTALL_DIR}/bin:\${PATH}
export INCLUDE=\${PLUMED_INSTALL_DIR}/include:\${INCLUDE}
export CPATH=\${PLUMED_INSTALL_DIR}/include:\${CPATH}
export LIBRARY_PATH=\${PLUMED_INSTALL_DIR}/lib:\${LIBRARY_PATH}
export LD_LIBRARY_PATH=\${PLUMED_INSTALL_DIR}/lib:\${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=\${PLUMED_INSTALL_DIR}/lib/pkgconfig:\${PKG_CONFIG_PATH}
export PLUMED_KERNEL=\${PLUMED_INSTALL_DIR}/lib/libplumedKernel.so
EOF
chmod a+x source-plumed.sh
echo ""
echo "build-plumed.sh: PLUMED has been installed in ${INSTALL_DIR}"
echo "build-plumed.sh:: You need to use "source source-plumed.sh" to load all the variables related to PLUMED" 
