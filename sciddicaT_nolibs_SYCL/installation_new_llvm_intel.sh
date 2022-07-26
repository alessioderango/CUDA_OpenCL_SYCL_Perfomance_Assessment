export DPCPP_HOME=~/sycl_workspace
mkdir $DPCPP_HOME
cd $DPCPP_HOME
git clone https://github.com/intel/llvm -b sycl
export PATH=/usr/lib/jvm/default/bin:/opt/mpich/bin:/usr/bin/site_perl:/usr/bin/vendor_perl:/usr/bin/core_perl:/usr/bin
export LD_LIBRARY_PATH=
CUDA_LIB_PATH=/opt/cuda/lib64/stubs CC=gcc CXX=g++ python $DPCPP_HOME/llvm/buildbot/configure.py --cuda --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=/opt/cuda"
CUDA_LIB_PATH=/opt/cuda/lib64/stubs CC=gcc CXX=g++ python $DPCPP_HOME/llvm/buildbot/compile.py