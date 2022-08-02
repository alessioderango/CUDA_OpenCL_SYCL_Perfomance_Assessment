#!/bin/bash

if [ "$1" != "-gpu" ] || [ "$2" == "" ] || [ "$3" != "-block_size" ] || [ "$4" == "" ]
then
  echo "$1 $2 $3 $4 $5 $6 $7 $8 $9 $10 $11 $12"
  echo "Usage: sh -gpu <gpu_id>"
  echo ""
  echo "Examples:"
  echo "  test on gpu 0 with block size 16x16:"
  echo "    bash $0 -gpu 0 -block_size 16 -type standard -folder_name standard -cuda_executable sciddicaTcudaStraightforward -sycl_executable sciddicaTsycl_dpcpp"
  echo ""
  echo "  test on gpu 1 with block_size 8x8:"
  echo "    bash $0 -gpu 1 -block_size 8 -type standard -folder_name standard -cuda_executable sciddicaTcudaStraightforward -sycl_executable sciddicaTsycl_dpcpp"
  echo ""
  exit 0
fi

#standard
bash run_profile_nsight.sh -gpu $2 -block_size $4 -type standard -folder_name standard -cuda_executable sciddicaTcudaStraightforward -sycl_executable sciddicaTsycl_dpcpp

bash run_profile_nsight.sh -gpu $2 -block_size $4 -type standard -folder_name standard_with_halo -cuda_executable sciddicaTcudawithHaloCells -sycl_executable sciddicaTsycl_dpcpp_with_halo_cells

bash run_profile_nsight.sh -gpu $2 -block_size $4 -type standard -folder_name standard_without_halo -cuda_executable sciddicaTcudaWithoutHaloCells -sycl_executable sciddicaTsycl_dpcpp_without_halo_cells

#stresstest
bash run_profile_nsight.sh -gpu $2 -block_size $4 -type stresstest -folder_name stresstest -cuda_executable sciddicaTcudaStraightforward -sycl_executable sciddicaTsycl_dpcpp

bash run_profile_nsight.sh -gpu $2 -block_size $4 -type stresstest -folder_name stresstest_with_halo -cuda_executable sciddicaTcudawithHaloCells -sycl_executable sciddicaTsycl_dpcpp_with_halo_cells

bash run_profile_nsight.sh -gpu $2 -block_size $4 -type stresstest -folder_name stresstest_without_halo -cuda_executable sciddicaTcudaWithoutHaloCells -sycl_executable sciddicaTsycl_dpcpp_without_halo_cells
