#!/bin/bash

if [ "$1" != "-gpu" ] || [ "$2" == "" ] || [ "$3" != "-block_size" ] || [ "$4" == "" ]
then
  echo ""
  echo "Usage: sh -gpu <gpu_id>"
  echo ""
  echo "Examples:"
  echo "  test on gpu 0 with block size 16x16:"
  echo "    sh $0 -gpu 0 -block_size 16"
  echo ""
  echo "  test on gpu 1 with block_size 8x8:"
  echo "    sh $0 -gpu 1 -block_size 8"
  echo ""
  exit 0
fi

HOSTNAME=`echo $HOSTNAME`
echo "HOSTNAME $HOSTNAME"
source ./setenv_$HOSTNAME.txt
export CUDA_VISIBLE_DEVICES=$2

OutputDir=./LogProfile_with_halo_gpu_$2_block_size_$4_hostname_${HOSTNAME}_`date +"%Y-%m-%d_%T"`
mkdir -p ${OutputDir}
OutputDirResults=$OutputDir/Results
mkdir -p ${OutputDirResults}

delimiter=";"

logfileprofilmetricsCUDA=./${OutputDir}/log_sciddicaTcuda_naive_oi.csv
logfileprofilruntimeCUDA=./${OutputDir}/log_sciddicaTcuda_naive_flop.csv
executableDirCUDA=./sciddicaT_nolibs_CUDA/
executableStrCUDA=" ./sciddicaTcudawithHaloCells ../data/tessina_header.txt ../data/tessina_dem.txt ../data/tessina_source.txt ./tessina_output_cuda 100 $4 $4 && cd .."
executableStrRuntimeCUDA=" ./sciddicaTcudawithHaloCells ../data/tessina_header.txt ../data/tessina_dem.txt ../data/tessina_source.txt ./tessina_output_cuda 4000 $4 $4 && cd .."


logfileprofilmetricsSYCL=./${OutputDir}/log_sciddicaTsycl_naive_oi.csv
logfileprofilruntimeSYCL=./${OutputDir}/log_sciddicaTsycl_naive_flop.csv
executableDirSYCL=./sciddicaT_nolibs_SYCL/
executableStrSYCL=" ./sciddicaTsycl_dpcpp_with_halo_cells ../data/tessina_header.txt ../data/tessina_dem.txt ../data/tessina_source.txt ./tessina_output_OpenCL 100 $4 $4 && cd .."
executableStrRuntimeSYCL=" ./sciddicaTsycl_dpcpp_with_halo_cells ../data/tessina_header.txt ../data/tessina_dem.txt ../data/tessina_source.txt ./tessina_output_OpenCL 4000 $4 $4 && cd .."


declare -a kernelsCUDA=( sciddicaTFlowsComputation sciddicaTWidthUpdate sciddicaTResetFlows )
declare -a kernelsSYCL=( Compute Update Reset )
declare -a metrics=( flop_count_dp flop_count_sp flop_count_hp gld_transactions gst_transactions atomic_transactions local_load_transactions local_store_transactions shared_load_transactions shared_store_transactions l2_read_transactions l2_write_transactions dram_read_transactions dram_write_transactions )

echo "Profiling sciddicaTcuda_naive"
echo "..."

########## CUDA         #############

profileStrCUDA="cd ${executableDirCUDA} && nvprof --log-file ../${logfileprofilmetricsCUDA} --csv "
for m in ${metrics[@]}
do
    profileStrCUDA+="--metrics $m "
done
profileStrCUDA+=$executableStrCUDA
echo "Running $profileStrCUDA"
eval ${profileStrCUDA}
echo ""


profileStrRuntimeCUDA="cd ${executableDirCUDA} && nvprof --log-file ../${logfileprofilruntimeCUDA} --csv --print-gpu-summary ${executableStrRuntimeCUDA}"
echo "Running $profileStrRuntimeCUDA"
eval $profileStrRuntimeCUDA
echo ""

########## END  CUDA    #############

########## SYCL         #############

profileStrSYCL="cd ${executableDirSYCL} && nvprof --log-file ../${logfileprofilmetricsSYCL} --csv "
for m in ${metrics[@]}
do
    profileStrSYCL+="--metrics $m "
done
profileStrSYCL+=$executableStrSYCL
echo "Running $profileStrSYCL"
eval $profileStrSYCL
echo ""

profileStrRuntimeSYCL="cd ${executableDirSYCL} && nvprof --log-file ../${logfileprofilruntimeSYCL} --csv --print-gpu-summary ${executableStrRuntimeSYCL}"
echo "Running $profileStrRuntimeSYCL"
eval $profileStrRuntimeSYCL
echo " "

########## END SYCL      #############

echo "Done"
echo "Results saved in ${OutputDir}/Results"

###############  CUDA NVPROF ##################
for i in "${kernelsCUDA[@]}"
do 
    resultsFileNameKernel="${OutputDirResults}/results_cuda_${i}.csv"
    echo "${i}${delimiter}Min${delimiter}Max${delimiter}Avg" > $resultsFileNameKernel
    outputKernel=$(grep -hnr "${i}" ${logfileprofilmetricsCUDA})
    tmpFileName="${OutputDirResults}/tmpFile.txt"
    echo "" > "${tmpFileName}"
    for t in "${outputKernel[@]}"; 
    do
        echo "${t} \n">> "${tmpFileName}"
    done 
    for m in "${metrics[@]}"
    do
        outputMetric=$(grep -hnr "${m}" ${tmpFileName})
        len=${#outputMetric[@]}
        #delete endline
        for o in "${!outputMetric[@]}"
        do
            outputMetric[o]=$( echo "${outputMetric[o]}" | tr -d '\\n' )
        done
        r=${outputMetric[@]}
        dd=(${r//,/ })
        echo "${m}$delimiter${dd[-3]}$delimiter${dd[-2]}$delimiter${dd[-1]}$delimiter" >> $resultsFileNameKernel
    done
    rm $tmpFileName
done

for i in "${kernelsCUDA[@]}"
do 
    resultsFileNameKernel="${OutputDirResults}/results_cuda_${i}.csv"
    unitsProfile=$(grep -hnr "%" ${logfileprofilruntimeCUDA})
    r=${unitsProfile[@]}
    dd=(${r//,/ })
    echo "${i}${delimiter}time(${dd[-4]})${delimiter}Avg(${dd[-3]})${delimiter}Min(${dd[-2]})${delimiter}Max(${dd[-1]})${delimiter}" >> $resultsFileNameKernel 
    timebyprofile=$(grep -hnr "${i}" ${logfileprofilruntimeCUDA})
    r1=${timebyprofile[@]}
    for o in "${!timebyprofile[@]}"
    do
        timebyprofile[o]=$( echo "${timebyprofile[o]}" | tr -d '\\n' )
    done
    dd1=(${r1//,/ })
    echo "${i}$delimiter${dd1[3]}$delimiter${dd1[5]}$delimiter${dd1[6]}$delimiter${dd1[7]}$delimiter" >> $resultsFileNameKernel
done
############### END CUDA NVPROF ##################


###############  SYCL DPCPP NVPROF ##################

for i in "${kernelsSYCL[@]}"
do 
    resultsFileNameKernel="${OutputDirResults}/results_sycl_${i}.csv"
    echo "${i}${delimiter}Min${delimiter}Max${delimiter}Avg" > $resultsFileNameKernel
    outputKernel=$(grep -hnr "${i}" ${logfileprofilmetricsSYCL})
    tmpFileName="${OutputDirResults}/tmpFile.txt"
    echo "" > "${tmpFileName}"
    for t in "${outputKernel[@]}"; 
    do
        echo "${t} \n">> "${tmpFileName}"
    done 
    for m in "${metrics[@]}"
    do
        outputMetric=$(grep -hnr "${m}" ${tmpFileName})
        len=${#outputMetric[@]}
        #delete endline
        for o in "${!outputMetric[@]}"
        do
            outputMetric[o]=$( echo "${outputMetric[o]}" | tr -d '\\n' )
        done
        r=${outputMetric[@]}
        dd=(${r//,/ })
        echo "${m}$delimiter${dd[-3]}$delimiter${dd[-2]}$delimiter${dd[-1]}$delimiter" >> $resultsFileNameKernel
    done
    rm $tmpFileName
done

for i in "${kernelsSYCL[@]}"
do 
    resultsFileNameKernel="${OutputDirResults}/results_sycl_${i}.csv"
    unitsProfile=$(grep -hnr "%" ${logfileprofilruntimeSYCL})
    r=${unitsProfile[@]}
    dd=(${r//,/ })
    echo "${i}${delimiter}time(${dd[-4]})${delimiter}Avg(${dd[-3]})${delimiter}Min(${dd[-2]})${delimiter}Max(${dd[-1]})${delimiter}" >> $resultsFileNameKernel 
    timebyprofile=$(grep -hnr "${i}" ${logfileprofilruntimeSYCL})
    r1=${timebyprofile[@]}
    for o in "${!timebyprofile[@]}"
    do
        timebyprofile[o]=$( echo "${timebyprofile[o]}" | tr -d '\\n' )
    done
    dd1=(${r1//,/ })
    echo "${i}$delimiter${dd1[3]}$delimiter${dd1[5]}$delimiter${dd1[6]}$delimiter${dd1[7]}$delimiter" >> $resultsFileNameKernel
done

###############  END SYCL DPCPP NVPROF ##################
