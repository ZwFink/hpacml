#!/bin/bash

prefix=$1
threads=$2
current_dir=$(pwd)

NOCOLOR='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
LIGHTGRAY='\033[0;37m'
DARKGRAY='\033[1;30m'
LIGHTRED='\033[1;31m'
LIGHTGREEN='\033[1;32m'
YELLOW='\033[1;33m'
LIGHTBLUE='\033[1;34m'
LIGHTPURPLE='\033[1;35m'
LIGHTCYAN='\033[1;36m'
WHITE='\033[1;37m'

clang_bin=$prefix/bin/clang
approx_runtime_lib=/dev/null


if [ ! -f $clang_bin ]; then
  mkdir -p build_compiler
  mkdir -p $prefix
  pushd build_compiler
  cmake -G Ninja \
    -DCMAKE_INSTALL_PREFIX=$prefix \
    -DLLVM_CCACHE_BUILD='Off'\
    -DCMAKE_EXPORT_COMPILE_COMMANDS='On'\
    -DCMAKE_BUILD_TYPE='RelWithDebInfo' \
    -DLLVM_FORCE_ENABLE_STATS='On' \
    -DLLVM_ENABLE_PROJECTS='clang' \
    -DCMAKE_C_COMPILER='clang' \
    -DCMAKE_CXX_COMPILER='clang++' \
    -DLLVM_ENABLE_RUNTIMES='openmp' \
    -DLLVM_OPTIMIZED_TABLEGEN='On' \
    -DCLANG_BUILD_EXAMPLES='On' \
    -DBUILD_SHARED_LIBS='On' \
    -DLLVM_ENABLE_ASSERTIONS='On' \
    ../llvm

    ninja -j $threads
    ninja -j $threads install 
    popd
    echo "#!/bin/bash" > hpac_env.sh
    echo "export PATH=$prefix/bin/:\$PATH" >> hpac_env.sh
    echo "export LD_LIBRARY_PATH=$prefix/lib/:\$LD_LIBRARY_PATH" >> hpac_env.sh
    echo "export CC=clang" >> hpac_env.sh
    echo "export CPP=clang++" >> hpac_env.sh
fi

torch_d=`spack location -i py-torch`
#hdf5_d=`spack location -i hdf5`
hdf5_d=/sw/spack/delta-2022-03/apps/hdf5/1.13.1-gcc-11.2.0-jb4yx45
echo HDF5 directory: $hdf5_d

torch_d=$(echo $torch_d/lib/python3.*/site-packages/torch/share/cmake/Torch)
echo Torch directory: $torch_d

gpuarchsm=$(python3 approx/approx_utilities/detect_arch.py $prefix)
gpuarch=$(echo $gpuarchsm | cut -d ';' -f 1)
gpusm=$(echo $gpuarchsm | cut -d ';' -f 2)

echo "export HPAC_GPU_ARCH=$gpuarch" >> hpac_env.sh
echo "export HPAC_GPU_SM=$gpusm" >> hpac_env.sh

if [ ! $? -eq 0 ]; then
  echo "ERROR: No GPU architecture targets found, exiting..."
  exit 1
else
  echo "Building for GPU architecture $gpuarch, compute capability $gpusm"
fi
source hpac_env.sh

if [ ! -f $approx_runtime_lib ]; then
  mkdir build_hpac
  pushd build_hpac
  CC=clang CPP=clang++ cmake -G Ninja \
      -DCMAKE_INSTALL_PREFIX=$prefix \
      -DLLVM_EXTERNAL_CLANG_SOURCE_DIR=${current_dir}/clang/ \
      -DPACKAGE_VERSION=17 \
      -DCMAKE_EXPORT_COMPILE_COMMANDS='On'\
      -DCMAKE_C_COMPILER='clang' \
      -DCMAKE_CXX_COMPILER='clang++' \
    -DCMAKE_BUILD_TYPE='RelWithDebInfo' \
    -DCAFFE2_USE_CUDNN='On' \
      -DTorch_DIR=$torch_d \
      -DHDF5_Dir=$hdf5_d \
     ../approx
    ninja -j $threads
    ninja -j $threads install
    popd
fi
exit
